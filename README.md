# Spalling do concreto

Aplicativo em português para predição da ocorrência de spalling e avaliação de confiabilidade pontual. Destinado ao uso pelo navegador, sem conhecimento de Python.

O cenário é fixo: **XGBoost com oito variáveis**, modelo previamente treinado com 904 amostras de 47 referências. A ferramenta mostra a predição, CL, AL, atendimento conjunto dos critérios e explicação local por Tree SHAP. Não treina modelos durante o uso, não exige login e não grava as avaliações em arquivos ou banco de dados.

## Publicar e obter um link

Siga [PUBLICAR.md](PUBLICAR.md). É possível fazer o upload pelo navegador do GitHub, sem instalar Git. A hospedagem proposta é o Streamlit Community Cloud.

## Usar a ferramenta

1. Preencha os oito campos nas unidades indicadas.
2. Clique em **Avaliar**.
3. Consulte a predição e a confiabilidade. Na sequência, veja quais variáveis influenciaram a saída do modelo.

O botão **Carregar exemplo** preenche um registro real da base apenas para demonstração. Ele não representa validação externa. **Limpar campos** remove a entrada e o resultado da sessão.

| Entrada | Unidade na tela |
|---|---|
| Relação água/aglomerante | Razão adimensional, por exemplo 0,30 |
| Taxa de aquecimento | °C/min |
| Teor de umidade | %, por exemplo 3 para 3% |
| Comprimento característico | mm |
| Temperatura máxima de exposição | °C |
| Relação sílica ativa/aglomerante | Razão adimensional, por exemplo 0,10 para 10% |
| Tamanho máximo do agregado | mm |
| Quantidade de fibras de polipropileno | kg/m³ |

A interface converte a umidade de porcentagem para fração antes da inferência. Os outros atributos mantêm as unidades da base. Comprimento característico é a menor distância de escape do vapor entre o centroide e a superfície do corpo de prova, conforme definido no projeto.

## Limites de interpretação

A base representa corpos de prova de dimensões reduzidas, aquecidos sem carregamento mecânico externo e sem restrição imposta à deformação. Não representa diretamente elementos estruturais carregados ou restringidos.

Atender a CL ≥ 0,80 e AL ≥ 0,75 fornece evidência empírica de concordância e ajuste local; não é garantia de acerto nem certificação da segurança de um elemento. CL e AL não são probabilidades de acerto. O filtro de densidade atua na base de referência, não constitui sozinho um teste de pertencimento da nova amostra ao domínio experimental. Valores fora das faixas observadas são sinalizados na tela sem mudar a regra original CL + AL.

Os resultados do artigo inicial em português, com 855 amostras e outros parâmetros, não devem ser atribuídos a esta versão do aplicativo.

## Arquivos

```text
app.py                         Interface Streamlit
spalling.py                    Predição, CL, AL e Tree SHAP
requirements.txt               Dependências com versões fixas
assets/model.ubj               Modelo XGBoost original, formato nativo
assets/reference.csv           Entradas, classes e acertos OOF-LORO
assets/metadata.json           Ordem das variáveis, parâmetros e hashes
.streamlit/config.toml         Tema e configuração de execução
tests/                         Paridade numérica e fluxo da interface
.github/workflows/tests.yml    Testes automáticos no GitHub
docs/METODOLOGIA.md             Método, origem e limites
docs/VALIDACAO.md               Evidências dos testes desta versão
```

Os artefatos operacionais foram exportados do cenário `08Var_XGB_todas_refs` do notebook 31. Não é necessário enviar os notebooks, resultados intermediários, PDFs dos artigos ou objetos joblib/pickle para a hospedagem. O modelo é carregado em formato nativo UBJSON; a base de referência é um CSV com precisão preservada.

## Execução local para desenvolvimento

Com Python 3.12, na pasta deste repositório:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes comparam resultados a casos de referência calculados usando as funções originais do projeto e o modelo operacional do notebook 31. A validação de software não amplia a validação científica do método.

O cálculo nativo de Tree SHAP do XGBoost é usado para reduzir as dependências da hospedagem. Sua equivalência com `shap.TreeExplainer` foi verificada na preparação desta versão. As contribuições são em margem/log-odds, não pontos percentuais de probabilidade.

## Autoria e referências

Pesquisa de Marcelo Mesquita do Amaral e Mauro de Vasconcellos Real.

- Amaral, M. M. do; Real, M. de V. **Podemos confiar na predição de spalling do concreto sob incêndio por aprendizado de máquina?** Ambiente Construído, v. 26, e155274, 2026. [DOI](https://doi.org/10.1590/s1678-86212026000101027).
- [Documentação do Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud).
- [Documentação de persistência do XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

## Licença

A licença de distribuição do código, modelo e base deve ser definida pelo autor antes da publicação. Este pacote não acrescenta uma licença aos materiais originais. Consulte [docs/LICENCA.md](docs/LICENCA.md).
