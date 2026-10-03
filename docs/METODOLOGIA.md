# Método e artefatos operacionais

## Cenário

XGBoost, oito variáveis, todas as 904 amostras de 47 referências incluídas no treino do modelo final. O cenário corresponde a `08Var_XGB_todas_refs` do notebook `31_app_nova_amostra.ipynb`.

O modelo e os acertos OOF foram exportados do objeto operacional já salvo. Não houve retreinamento, tuning ou alteração da base durante a preparação do aplicativo. `assets/metadata.json` registra o hash SHA-256 do objeto de origem, os hashes dos artefatos distribuídos, a ordem dos atributos e a configuração efetiva.

## Predição

O modelo nativo XGBoost retorna a probabilidade estimada de classe 1. A ocorrência de spalling é predita se esse valor for maior ou igual a 0,50. O modelo operacional XGBoost não tem padronização antes da classificação; a padronização é usada exclusivamente nas distâncias para CL e AL.

## Base de confiabilidade

`reference.csv` contém o identificador original, referência, oito atributos, resposta real e indicador de acerto OOF-LORO. Os acertos são alinhados por identificador aos registros do treino operacional. O AL não utiliza acertos do modelo avaliado nos próprios registros de treinamento.

As estatísticas do `StandardScaler` são calculadas uma vez sobre as 904 observações operacionais. Para cada observação de treinamento, obtém-se o raio até o décimo vizinho, excluindo a própria observação da contagem mediante a consulta de 11 pontos. O limiar de densidade é o quantil 0,90 desses raios. Mantêm-se registros com raio menor ou igual a esse limiar.

Para a nova amostra, buscam-se os dez vizinhos mais próximos nesse subconjunto filtrado e calculam-se pesos:

`w_i = 1 / (d_i + 10^-12)`

`CL = soma(w_i × [classe real do vizinho = classe predita]) / soma(w_i)`

`AL = soma(w_i × [predição OOF-LORO do vizinho correta]) / soma(w_i)`

O critério combinado requer simultaneamente `CL >= 0,80 - 10^-8` e `AL >= 0,75 - 10^-8`, preservando a tolerância do código original. Vizinhos coincidentes mantêm a mesma ponderação com epsilon; não se altera esse caso para favorecer resultados.

O filtro remove amostras pouco densas do treino, mas não recusa automaticamente uma amostra nova pela distância absoluta à base. A sinalização de faixas excedidas na interface é informativa e não redefine o critério CL + AL.

## SHAP

O aplicativo usa `Booster.predict(pred_contribs=True, approx_contribs=False)`, cálculo nativo exato de Tree SHAP do XGBoost. A validação compara esse resultado com o `shap.TreeExplainer` usado no notebook. Não é necessário instalar o pacote SHAP na hospedagem.

O vetor contém uma contribuição para cada atributo e um valor de base. A soma corresponde à margem/log-odds; sua conversão logística corresponde à probabilidade estimada. As barras descrevem o comportamento do modelo para a entrada, não efeitos causais ou variações diretas de probabilidade.

## Entradas e domínio

A umidade é exibida em porcentagem e dividida por 100 antes da inferência. As razões água/aglomerante e sílica ativa/aglomerante são adimensionais. As demais unidades estão nos campos e no metadado.

Dados incompletos, não finitos, negativos ou fisicamente incompatíveis com as definições dos campos são rejeitados. Valores fisicamente admissíveis fora das faixas individuais da base são sinalizados, sem alterar as fórmulas originais. O aplicativo não infere valores ausentes nem substitui campos não informados por medianas.

A base representa corpos de prova de dimensões reduzidas, sem carregamento mecânico externo e sem restrição imposta à deformação. Não cobre diretamente elementos estruturais carregados ou restringidos. Valores individuais dentro das faixas observadas não garantem suporte para uma combinação nova.

CL e AL são indicadores empíricos. Um resultado aceito não é garantia de acerto; um resultado não aceito não significa necessariamente erro. As métricas dos subconjuntos do artigo inicial não constituem garantia para novas entradas.

## Referências

- Amaral, M. M. do; Real, M. de V. *Podemos confiar na predição de spalling do concreto sob incêndio por aprendizado de máquina?* Ambiente Construído, v. 26, e155274, 2026. [DOI](https://doi.org/10.1590/s1678-86212026000101027). Etapa inicial: 855 amostras, 17 atributos e configuração anterior.
- [API do XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
- [TreeExplainer e escala da explicação](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
