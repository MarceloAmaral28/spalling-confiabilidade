"""Inferência e confiabilidade do cenário 08Var/XGB do notebook 31."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass
class Evaluation:
    probability: float
    predicted_class: int
    cl: float
    al: float
    reliable_cl: bool
    reliable_al: bool
    reliable: bool
    inputs: dict[str, float]
    outside_ranges: list[str]
    matching_records: int
    neighbors: pd.DataFrame


class SpallingModel:
    """Carrega apenas o modelo e a base operacional; não treina modelos."""

    def __init__(self, assets: str | Path):
        self.assets = Path(assets)
        self.metadata = json.loads((self.assets / 'metadata.json').read_text(encoding='utf-8'))
        for name, expected in self.metadata['sha256'].items():
            actual = hashlib.sha256((self.assets / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Artefato inconsistente: {name}. Restaure o pacote publicado.')
        self.features = self.metadata['features']
        self.columns = [feature['column'] for feature in self.features]
        self.parameters = self.metadata['parameters']
        self.reference = pd.read_csv(self.assets / 'reference.csv', float_precision='round_trip')
        if list(self.reference.columns) != ['idx_original', 'Reference', *self.columns, 'output', 'correct_oof']:
            raise ValueError('Colunas da base operacional não correspondem ao modelo.')
        if len(self.reference) != self.metadata['n_samples']:
            raise ValueError('Número de observações diferente da versão publicada.')
        if self.reference['idx_original'].duplicated().any() or self.reference.isna().any().any():
            raise ValueError('Base operacional tem identificadores repetidos ou valores ausentes.')
        for name in ('output', 'correct_oof'):
            if not self.reference[name].isin([0, 1]).all():
                raise ValueError(f'Coluna binária inválida: {name}.')
        self.x_train = self.reference[self.columns].to_numpy(dtype=float)
        if not np.isfinite(self.x_train).all():
            raise ValueError('Base operacional contém valores não finitos.')
        self.scaler = StandardScaler().fit(self.x_train)
        self.x_scaled = self.scaler.transform(self.x_train)
        k_den = self.parameters['k_den']
        if len(self.x_scaled) <= k_den:
            raise ValueError('Base insuficiente para o filtro de densidade.')
        distances, _ = KDTree(self.x_scaled).query(self.x_scaled, k=k_den + 1)
        self.radii = distances[:, -1]
        self.density_radius = float(np.quantile(self.radii, 1 - self.parameters['alpha_den']))
        self.dense_mask = self.radii <= self.density_radius
        self.dense_rows = self.reference.loc[self.dense_mask].reset_index(drop=True)
        self.tree = KDTree(self.x_scaled[self.dense_mask])
        self.k = min(self.parameters['k'], len(self.dense_rows))
        self.model = xgb.Booster()
        self.model.load_model(self.assets / 'model.ubj')
        self.model.set_param({'nthread': 1})
        if self.model.feature_names != self.columns:
            raise ValueError('Ordem dos atributos incompatível com o modelo treinado.')

    def validate_inputs(self, inputs: dict[str, float]) -> np.ndarray:
        if set(inputs) != set(self.columns):
            raise ValueError('Informe os oito atributos da ferramenta.')
        values = np.asarray([inputs[column] for column in self.columns], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Preencha todos os campos com números finitos.')
        for feature, value in zip(self.features, values):
            if value < 0 or (feature['strictly_positive'] and value <= 0):
                raise ValueError(f"{feature['label']}: informe um valor {'maior que' if feature['strictly_positive'] else 'maior ou igual a'} zero.")
            if feature.get('physical_max') is not None and value > feature['physical_max']:
                raise ValueError(f"{feature['label']}: valor acima do limite físico informado.")
        return values.reshape(1, -1)

    def evaluate(self, inputs: dict[str, float]) -> Evaluation:
        values = self.validate_inputs(inputs)
        matrix = xgb.DMatrix(values, feature_names=self.columns)
        probability = float(self.model.predict(matrix)[0])
        predicted_class = int(probability >= self.parameters['prediction_threshold'])
        scaled = self.scaler.transform(values)
        distances, indices = self.tree.query(scaled, k=self.k)
        distances, indices = distances[0], indices[0]
        weights = 1.0 / (distances + self.parameters['distance_epsilon'])
        neighbors = self.dense_rows.iloc[indices].copy().reset_index(drop=True)
        agreement = neighbors['output'].to_numpy(dtype=int) == predicted_class
        correct = neighbors['correct_oof'].to_numpy(dtype=int)
        cl = float(np.sum(weights * agreement) / np.sum(weights))
        al = float(np.sum(weights * correct) / np.sum(weights))
        tolerance = self.parameters['threshold_tolerance']
        reliable_cl = bool(cl >= self.parameters['cl_threshold'] - tolerance)
        reliable_al = bool(al >= self.parameters['al_threshold'] - tolerance)
        outside = [f['label'] for f, v in zip(self.features, values[0]) if v < f['training_min'] or v > f['training_max']]
        matches = int(np.all(np.isclose(self.x_train, values, rtol=0, atol=1e-12), axis=1).sum())
        neighbors.insert(0, 'distance', distances)
        neighbors.insert(1, 'weight_fraction', weights / weights.sum())
        return Evaluation(
            probability, predicted_class, cl, al, reliable_cl, reliable_al,
            reliable_cl and reliable_al, dict(inputs), outside, matches, neighbors,
        )

    def explain(self, inputs: dict[str, float]) -> tuple[np.ndarray, float]:
        """Tree SHAP exato em margem/log-odds, como no notebook 31.

        XGBoost inclui o cálculo nativo de Tree SHAP. Evita carregar o pacote
        shap no servidor; a paridade com TreeExplainer é verificada nos testes.
        """
        values = self.validate_inputs(inputs)
        matrix = xgb.DMatrix(values, feature_names=self.columns)
        contributions = np.asarray(self.model.predict(matrix, pred_contribs=True, approx_contribs=False))[0]
        margin = float(self.model.predict(matrix, output_margin=True)[0])
        if not np.isclose(contributions.sum(), margin, atol=2e-5, rtol=2e-5):
            raise ValueError('A explicação SHAP não reconstrói a margem do modelo.')
        return contributions[:-1].astype(float), float(contributions[-1])
