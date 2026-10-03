# Verificação da versão 1.0.0

Verificação local realizada em 2 de outubro de 2026, com Python 3.12.

## Equivalência com o projeto original

Foram comparadas 944 entradas: as 904 amostras da base operacional e 40 combinações sintéticas utilizadas exclusivamente para verificar o software. A referência foi o pipeline serializado original, as funções originais de CL e AL e o `TreeExplainer` original. O modelo não foi retreinado.

- Diferença máxima de probabilidade, CL, AL e contribuições SHAP: zero nas entradas verificadas.
- Nenhuma divergência na classe, nos critérios de confiabilidade ou nos identificadores dos vizinhos.
- Subconjunto denso: 813 amostras; raio de corte: 1,7888611102573966.
- A integridade do artefato original foi conferida por hash antes e depois da exportação.

Os números detalhados estão em `validation_results.json`. Os casos de regressão em `tests/fixtures.json` foram produzidos a partir do código original.

## Interface e testes automatizados

Os cinco testes de `python -m unittest discover -s tests -v` passaram. Foram verificados preenchimento obrigatório, exemplo, conversão da umidade de porcentagem para fração, avaliação, aviso de extrapolação, rejeição de entradas inválidas, limpeza dos campos, equivalência numérica e reconstrução da probabilidade por SHAP.

O aplicativo também foi aberto em navegador local. O exemplo foi carregado e avaliado; predição, CL, AL e gráfico SHAP apareceram na tela sem erros.

## Limites desta verificação

Esta é uma verificação de implementação, não uma nova validação científica ou externa do modelo. As combinações sintéticas não possuem resultados experimentais. O exemplo da interface pertence à base de treinamento. A instalação e execução no Streamlit Community Cloud devem ser conferidas após a publicação; essa hospedagem ainda não foi realizada.
