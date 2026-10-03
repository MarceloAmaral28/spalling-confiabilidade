"""Regressão numérica contra resultados do código original do notebook 31."""

import json
from pathlib import Path
import unittest

import numpy as np

from spalling import SpallingModel

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SpallingModel(ROOT / 'assets')
        cls.fixtures = json.loads((ROOT / 'tests/fixtures.json').read_text(encoding='utf-8'))

    def test_numeric_parity_with_notebook31(self):
        for case in self.fixtures['cases']:
            with self.subTest(case=case['name']):
                actual = self.model.evaluate(case['inputs'])
                expected = case['expected']
                self.assertAlmostEqual(actual.probability, expected['probability'], delta=1e-7)
                self.assertAlmostEqual(actual.cl, expected['cl'], delta=1e-10)
                self.assertAlmostEqual(actual.al, expected['al'], delta=1e-10)
                self.assertEqual(actual.predicted_class, expected['predicted_class'])
                self.assertEqual(actual.reliable, expected['reliable'])
                self.assertEqual(actual.neighbors.idx_original.tolist(), expected['neighbors'])

    def test_shap_reconstructs_probability(self):
        for case in self.fixtures['cases']:
            with self.subTest(case=case['name']):
                result = self.model.evaluate(case['inputs'])
                contributions, base = self.model.explain(case['inputs'])
                self.assertEqual(len(contributions), 8)
                reconstructed = 1 / (1 + np.exp(-(base + contributions.sum())))
                self.assertAlmostEqual(reconstructed, result.probability, delta=2e-6)
                np.testing.assert_allclose(contributions, case['expected']['shap'], atol=2e-6, rtol=2e-6)

    def test_missing_nonfinite_and_invalid_inputs(self):
        valid = self.fixtures['cases'][0]['inputs'].copy()
        with self.assertRaises(ValueError):
            self.model.evaluate({})
        for key, value in [
            (self.model.columns[0], float('nan')),
            (self.model.columns[1], 0),
            (self.model.columns[2], 1.1),
            (self.model.columns[3], -1),
            (self.model.columns[5], 1.1),
        ]:
            with self.subTest(key=key, value=value):
                invalid = {**valid, key: value}
                with self.assertRaises(ValueError):
                    self.model.evaluate(invalid)

    def test_extrapolation_does_not_change_original_reliability_rule(self):
        valid = self.fixtures['cases'][0]['inputs'].copy()
        valid[self.model.columns[6]] = 100
        result = self.model.evaluate(valid)
        self.assertIn(self.model.features[6]['label'], result.outside_ranges)
        self.assertEqual(result.reliable, result.reliable_cl and result.reliable_al)


if __name__ == '__main__':
    unittest.main()
