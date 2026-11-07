"""Interface pública em português. Execute: streamlit run app.py."""

from pathlib import Path
from io import BytesIO
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'spalling-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter

from spalling import Evaluation, SpallingModel


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Spalling — confiabilidade pontual', page_icon='🏗️', layout='wide')


@st.cache_resource(show_spinner='Carregando o modelo e a base de referência…')
def load_model() -> SpallingModel:
    return SpallingModel(ROOT / 'assets')


def display_value(value: float) -> str:
    return f'{value:.6g}'.replace('.', ',')


FORM_GROUPS = (
    ('Material', ('water_binder', 'silica_binder', 'aggregate_size', 'pp_fiber')),
    ('Exposição, geometria e teor de umidade', ('heating_rate', 'temperature', 'characteristic_length', 'moisture')),
)


def feature_label(feature: dict, label: str | None = None) -> str:
    title = label or feature['label']
    return title if feature['unit'] == 'adimensional' else f"{title} ({feature['unit']})"


def display_indicator(label: str, value: float, passed: bool, key: str) -> None:
    with st.container(key=key if passed else f'{key}_below', border=True):
        st.metric(label, f'{value:.3f}'.replace('.', ','))


def history_row(model: SpallingModel, result: Evaluation, name: str) -> dict:
    return {
        'Caso': name,
        **{feature_label(f): result.inputs[f['column']] * f['ui_multiplier'] for f in model.features},
        'Predição de spalling': 'Ocorrência' if result.predicted_class else 'Não ocorrência',
        'Probabilidade estimada': result.probability,
        'CL': result.cl,
        'AL': result.al,
        'Critério CL atendido': 'Sim' if result.reliable_cl else 'Não',
        'Critério AL atendido': 'Sim' if result.reliable_al else 'Não',
        'Atende aos critérios de confiabilidade': 'Sim' if result.reliable else 'Não',
        'Atributos fora das faixas observadas': '; '.join(result.outside_ranges),
    }


def history_xlsx(rows: list[dict]) -> bytes:
    buffer = BytesIO()
    # Nomes dos casos devem ser texto, inclusive se começarem com '='.
    options = {'in_memory': True, 'strings_to_formulas': False, 'strings_to_urls': False}
    with xlsxwriter.Workbook(buffer, options) as workbook:
        sheet = workbook.add_worksheet('Avaliações')
        header = workbook.add_format({
            'bold': True, 'bg_color': '#E6EEF4', 'text_wrap': True, 'valign': 'vcenter',
        })
        percent = workbook.add_format({'num_format': '0.00%'})
        columns = list(rows[0])
        sheet.set_row(0, 42)
        for col, title in enumerate(columns):
            sheet.write_string(0, col, title, header)
            width = 27 if title == 'Caso' else min(30, max(13, len(title) // 2 + 5))
            sheet.set_column(col, col, width, percent if title == 'Probabilidade estimada' else None)
        for index, row in enumerate(rows, start=1):
            sheet.write_row(index, 0, [row[title] for title in columns])
        sheet.freeze_panes(1, 1)
        sheet.autofilter(0, 0, len(rows), len(columns) - 1)
    return buffer.getvalue()


def display_history(rows: list[dict]) -> None:
    count = len(rows)
    st.write(f'{count} avaliação nesta sessão' if count == 1 else f'{count} avaliações nesta sessão')
    st.download_button(
        'Baixar histórico de avaliações (.xlsx)',
        data=history_xlsx(rows) if rows else b'',
        file_name='historico-spalling.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        on_click='ignore',
        disabled=not rows,
    )
    st.caption('O histórico é temporário. Recarregar a página apaga as avaliações desta sessão. Baixe a planilha antes de sair ou recarregar.')


def draw_explanation(model: SpallingModel, result: Evaluation) -> None:
    contributions, base = model.explain(result.inputs)
    order = np.argsort(np.abs(contributions))[::-1]
    labels = [model.features[i]['label'] for i in order]
    values = contributions[order]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor('#FAFCFD')
    ax.set_facecolor('#FAFCFD')
    colors = ['#BE4B45' if v >= 0 else '#207D91' for v in values]
    ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.axvline(0, color='#607481', linewidth=0.8)
    ax.set_xlabel('Contribuição SHAP à saída do modelo (log-odds)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=10)
    ax.grid(axis='x', alpha=0.12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption('Vermelho: aumenta a pontuação de spalling. Azul: reduz a pontuação. O tamanho da barra indica a contribuição nesta avaliação, não uma relação de causa e efeito.')
    with st.expander('Valores informados e contribuições SHAP'):
        st.caption('As contribuições somadas ao valor de base reconstituem a margem do modelo. A conversão logística dessa margem produz sua probabilidade estimada.')
        rows = []
        for index in order:
            feature = model.features[index]
            rows.append({
                'Variável': feature['label'],
                'Valor informado': result.inputs[feature['column']] * feature['ui_multiplier'],
                'Unidade': feature['unit'],
                'Contribuição SHAP': float(contributions[index]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption(f'Valor de base: {base:.5f}. Margem final: {base + contributions.sum():.5f}.')


def display_result(model: SpallingModel, result: Evaluation) -> None:
    st.divider()
    st.subheader('Resultado da última avaliação enviada')
    st.caption('Ao mudar os campos, clique novamente em Avaliar para atualizar o resultado.')
    st.html('<style>.st-key-cl_below,.st-key-al_below {background-color:rgba(221,69,111,.15);border-radius:8px;}</style>')
    prediction, reliability = st.columns(2)
    with prediction:
        with st.container(border=True):
            st.markdown('**Predição de spalling**')
            st.markdown(f"### {'Ocorrência de spalling' if result.predicted_class else 'Não ocorrência de spalling'}")
            st.metric('Probabilidade estimada pelo modelo', f'{100 * result.probability:.1f}%'.replace('.', ','))
            st.caption('Limiar de classificação: 50%. Esta probabilidade não é uma garantia de ocorrência ou de acerto.')
    with reliability:
        with st.container(border=True):
            st.markdown('**Confiabilidade pontual**')
            if result.reliable:
                st.success('Atende aos critérios de confiabilidade')
            else:
                st.warning('Não atende aos critérios de confiabilidade')
            cl_column, al_column = st.columns(2)
            with cl_column:
                display_indicator('Concordância Local — CL', result.cl, result.reliable_cl, 'cl')
            with al_column:
                display_indicator('Ajuste Local — AL', result.al, result.reliable_al, 'al')
            st.caption('Critérios fixos: CL ≥ 0,80 e AL ≥ 0,75. Ambos precisam ser atendidos. O fundo rosa indica que o critério não foi atendido. Os valores são indicadores empíricos, não probabilidades de acerto.')
    if result.outside_ranges:
        st.warning('Valores fora das faixas observadas na base: ' + ', '.join(result.outside_ranges) + '. Há extrapolação em pelo menos um atributo. CL e AL não são um teste de pertencimento ao domínio experimental.')
    if result.matching_records:
        st.info(f'Esta combinação de oito entradas coincide com {result.matching_records} registro(s) da base de treinamento. O resultado do modelo final nesse caso não constitui validação externa.')
    st.subheader('O que influenciou esta predição')
    try:
        draw_explanation(model, result)
    except (ValueError, RuntimeError) as error:
        st.error('Não foi possível calcular a explicação desta avaliação. Os demais resultados estão apresentados acima.')
        st.caption(str(error))
    with st.expander('Como CL e AL foram calculados'):
        st.write('São usados os 10 vizinhos mais próximos na região de maior densidade da base de referência, após padronizar os oito atributos. Vizinhos próximos recebem maior peso.')
        check = '<span style="color: #16803C; font-weight: 600">✓</span>'
        cross = '<span style="color: #C93636; font-weight: 600">✗</span>'
        st.markdown(
            '**CL — Concordância Local:** indica quanto os resultados experimentais dos vizinhos concordam com a nova predição, considerando seus pesos. Veja a coluna **“Ocorreu spalling? (0/1)”**.',
            unsafe_allow_html=True,
        )
        st.markdown(
            '**AL — Ajuste Local:** indica quanto o modelo acertou nas predições dos vizinhos, considerando seus pesos. Veja a coluna **“O modelo acertou? (0/1)”**.',
            unsafe_allow_html=True,
        )
        neighbors = result.neighbors[['idx_original', 'Reference', 'output', 'correct_oof', 'distance', 'weight_fraction']].rename(columns={
            'idx_original': 'Registro', 'Reference': 'Referência', 'output': 'Ocorreu spalling? (0/1)',
            'correct_oof': 'O modelo acertou? (0/1)', 'distance': 'Distância padronizada', 'weight_fraction': 'Peso relativo',
        })
        observed_column = 'Ocorreu spalling? (0/1)'
        accuracy_column = 'O modelo acertou? (0/1)'
        display_neighbors = neighbors.copy()
        for column in ('Registro', 'Referência'):
            display_neighbors[column] = display_neighbors[column].map(lambda value: str(int(value)))
        display_neighbors[observed_column] = neighbors[observed_column].map(
            lambda value: f'{int(value)} {check if int(value) == result.predicted_class else cross}',
        )
        display_neighbors[accuracy_column] = neighbors[accuracy_column].map(
            lambda value: f'{int(value)} {check if int(value) == 1 else cross}',
        )
        for column in ('Distância padronizada', 'Peso relativo'):
            display_neighbors[column] = neighbors[column].map(lambda value: f'{value:.6g}')
        # HTML permite colorir só o símbolo, mantendo o número na cor do tema.
        table_html = display_neighbors.to_html(index=False, escape=False, border=0, classes='neighbor-status')
        st.markdown(
            '<style>.neighbor-status-wrap {overflow-x:auto; border:1px solid rgba(128,128,128,.25); border-radius:8px;}'
            '.neighbor-status {width:100%; border-collapse:collapse; color:inherit; font-size:0.9rem;}'
            '.neighbor-status th,.neighbor-status td {padding:9px 12px; border:1px solid rgba(128,128,128,.20); text-align:right; white-space:nowrap;}'
            '.neighbor-status th {font-weight:500; white-space:normal; min-width:5ch;}'
            '.neighbor-status td span {margin-left:4px;}</style>'
            f'<div class="neighbor-status-wrap">{table_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('**Legenda**')
        st.markdown(
            f'- **Ocorreu spalling?** 0 = não; 1 = sim, conforme o resultado experimental do vizinho. {check} indica concordância com a nova predição; {cross} indica discordância.\n'
            f'- **O modelo acertou?** 0 = não; 1 = sim. {check} indica acerto; {cross} indica erro. A avaliação utiliza LORO-CV, com a referência bibliográfica inteira do vizinho excluída do treinamento.\n'
            '- **Peso relativo:** participação de cada vizinho no cálculo de CL e AL.',
            unsafe_allow_html=True,
        )
        st.markdown('**Origem dos registros e referências:** [Fire-Induced Concrete Spalling Dataset](https://github.com/MarceloAmaral28/Fire-Induced-Concrete-Spalling-Dataset).')
        st.caption('Registro corresponde ao identificador idx_original da base processada. Referência identifica a fonte bibliográfica do experimento.')
        st.markdown('[Consultar o mapeamento das referências bibliográficas](https://github.com/MarceloAmaral28/Fire-Induced-Concrete-Spalling-Dataset/blob/main/reference_mapping.csv)')
        st.caption('O filtro remove regiões pouco densas do treino. Ele não verifica, isoladamente, se a nova entrada está perto o suficiente da base. Atender aos critérios de confiabilidade não assegura acerto.')
    with st.expander('Características dos vizinhos utilizados'):
        st.write('Compare a amostra informada com os mesmos 10 vizinhos usados no cálculo de CL e AL, apresentados na mesma ordem da tabela anterior.')
        short_labels = {
            'temperature': 'Temp. máx. de exposição',
            'aggregate_size': 'Tam. máx. do agregado',
            'pp_fiber': 'Fibras de PP',
            'silica_binder': 'Sílica ativa/aglomerante',
        }
        labels = {f['column']: feature_label(f, short_labels.get(f['key'])) for f in model.features}
        rows = [{'Amostra': 'Entrada informada', 'Registro': '—', 'Referência': '—', **{
            labels[f['column']]: result.inputs[f['column']] * f['ui_multiplier']
            for f in model.features
        }}]
        for position, (_, neighbor) in enumerate(result.neighbors.iterrows(), start=1):
            rows.append({'Amostra': f'Vizinho {position}', 'Registro': str(int(neighbor['idx_original'])),
                         'Referência': str(int(neighbor['Reference'])), **{
                             labels[f['column']]: float(neighbor[f['column']]) * f['ui_multiplier']
                             for f in model.features
                         }})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=460,
                     column_config={
                         'Amostra': st.column_config.TextColumn(pinned=True),
                         **{labels[f['column']]: st.column_config.NumberColumn(help=feature_label(f))
                            for f in model.features},
                     })
        st.caption('Os atributos são exibidos nas unidades do formulário, antes da padronização usada nas distâncias. As razões água/aglomerante e sílica ativa/aglomerante não têm unidade. Role a tabela horizontalmente para consultar todos os oito atributos.')
        st.markdown('[Origem dos dados e identificação dos registros e referências](https://github.com/MarceloAmaral28/Fire-Induced-Concrete-Spalling-Dataset)')


st.title('Spalling do concreto')
st.markdown('Predição e confiabilidade pontual para concreto exposto a temperaturas elevadas.')
st.caption('XGBoost · 8 variáveis · base de 904 amostras de 47 referências')

try:
    model = load_model()
except Exception:
    st.error('Não foi possível carregar os arquivos do modelo. O responsável pelo aplicativo precisa verificar a instalação e os artefatos da versão publicada.')
    st.stop()

with st.sidebar:
    st.subheader('Sobre a ferramenta')
    st.write('Preencha os oito campos, mantendo as unidades indicadas, e clique em Avaliar.')
    st.markdown('**Domínio experimental**')
    st.write('Corpos de prova de concreto de dimensões reduzidas, aquecidos sem carregamento mecânico externo e sem restrição imposta à deformação.')
    st.write('A base não representa diretamente elementos estruturais carregados ou restringidos. Use a ferramenta como apoio à análise, dentro desses limites.')
    st.caption('Os valores informados não são gravados pelo aplicativo em arquivos ou banco de dados.')
    st.caption(f"Versão {model.metadata['app_version']} · cenário operacional fixo")

st.subheader('Características da amostra')
st.caption('Todos os campos são obrigatórios. As faixas observadas são orientações; estar dentro delas não garante suporte para a combinação de atributos.')
example_button, clear_button, _ = st.columns([1, 1, 3])
if example_button.button('Carregar exemplo', width='stretch'):
    for feature in model.features:
        st.session_state[feature['key']] = feature['example'] * feature['ui_multiplier']
    st.session_state.pop('evaluation', None)
    st.session_state['example_loaded'] = True
if clear_button.button('Limpar campos', width='stretch'):
    for feature in model.features:
        st.session_state[feature['key']] = None
    st.session_state.pop('evaluation', None)
    st.session_state['example_loaded'] = False
if st.session_state.get('example_loaded'):
    st.caption('O exemplo é um registro da base, para demonstrar o preenchimento. Ele pode ser alterado e não é um teste externo do modelo.')

with st.form('sample_form', clear_on_submit=False):
    material, exposure = st.columns(2, gap='large')
    input_values = {}
    feature_by_key = {feature['key']: feature for feature in model.features}
    for column, (title, keys) in zip((material, exposure), FORM_GROUPS):
        with column:
            st.markdown(f'**{title}**')
            for key in keys:
                feature = feature_by_key[key]
                label = feature_label(feature)
                max_value = feature.get('physical_max')
                max_value = None if max_value is None else float(max_value * feature['ui_multiplier'])
                input_values[feature['column']] = st.number_input(
                    label, min_value=0.0, max_value=max_value, value=None,
                    step=feature['ui_step'], format=feature['ui_format'], key=feature['key'],
                    placeholder='Informe o valor', help=feature['help'],
                )
                minimum = display_value(feature['training_min'] * feature['ui_multiplier'])
                maximum = display_value(feature['training_max'] * feature['ui_multiplier'])
                st.caption(f'Faixa observada na base: {minimum} a {maximum} {feature["unit"]}.')
    submitted = st.form_submit_button('Avaliar', type='primary', width='stretch')

history = st.session_state.setdefault('evaluation_history', [])
if submitted:
    st.session_state.pop('evaluation', None)
    missing = [f['label'] for f in model.features if input_values[f['column']] is None]
    if missing:
        st.error('Preencha: ' + ', '.join(missing) + '.')
    else:
        inputs = {f['column']: float(input_values[f['column']]) / f['ui_multiplier'] for f in model.features}
        try:
            with st.spinner('Calculando a predição e a confiabilidade…'):
                evaluation = model.evaluate(inputs)
                history.append(history_row(model, evaluation, f'Caso {len(history) + 1}'))
                st.session_state['evaluation'] = evaluation
        except ValueError as error:
            st.error(str(error))

display_history(history)

result = st.session_state.get('evaluation')
if result is not None:
    display_result(model, result)

st.divider()
st.caption('Pesquisa de Marcelo Mesquita do Amaral e Mauro de Vasconcellos Real · avaliação de spalling e confiabilidade pontual.')
with st.expander('Método e referências'):
    st.write('Esta versão utiliza o cenário XGBoost com oito variáveis. O modelo final foi treinado com todas as 904 amostras; o AL utiliza acertos de predições OOF com exclusão por referência. Os parâmetros de decisão ficam fixos para todos os usuários.')
    st.markdown('[Artigo publicado — Ambiente Construído: Podemos confiar na predição de spalling do concreto sob incêndio por aprendizado de máquina?](https://doi.org/10.1590/s1678-86212026000101027)')
    st.caption('O artigo português descreve a etapa inicial, com 855 amostras e parâmetros diferentes. O aplicativo representa a evolução posterior do projeto; seus resultados não devem ser confundidos com os daquele experimento.')
