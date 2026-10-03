# Publicar no GitHub e no Streamlit

Este pacote já contém o modelo e a base operacional. Não execute o notebook 31 para iniciar o aplicativo.

## 1. Preparar o repositório

1. Entre na sua conta em [github.com](https://github.com).
2. Crie um repositório **público**. Sugestão de nome: `spalling-confiabilidade`.
3. Extraia o arquivo ZIP entregue e abra a pasta `spalling-app`.
4. No GitHub, escolha **Add file → Upload files**. Envie o conteúdo dessa pasta, mantendo as subpastas. O arquivo `app.py` deve ficar na raiz do repositório, junto com `requirements.txt`.
5. Inclua as pastas `assets`, `.streamlit`, `tests`, `docs` e `.github`, além dos demais arquivos. Pastas cujo nome começa por ponto também fazem parte do pacote. O GitHub não importa automaticamente o conteúdo de um ZIP: é necessário extraí-lo antes.
6. Registre os arquivos com **Commit changes**. Confira se `assets/model.ubj`, `assets/reference.csv` e `assets/metadata.json` aparecem no repositório.

Defina a licença em conjunto com a autoria do projeto antes de disponibilizar os materiais. A escolha ainda não foi feita neste pacote.

## 2. Criar o aplicativo

1. Entre em [Streamlit Community Cloud](https://share.streamlit.io/) e conecte sua conta do GitHub.
2. Escolha **Create app** e a opção de usar um aplicativo existente do GitHub.
3. Selecione seu repositório e a branch onde os arquivos foram enviados, normalmente `main`.
4. Em **Main file path**, informe `app.py`.
5. Em **Advanced settings**, selecione **Python 3.12**, o ambiente testado para este pacote.
6. Se desejar, escolha um subdomínio disponível para o aplicativo.
7. Clique em **Deploy** e aguarde a instalação das dependências.

Não é necessário cadastrar chaves de API, configurar Secrets, montar Google Drive ou subir todos os arquivos da pasta original.

## 3. Conferir o link

Abra o endereço fornecido pelo Streamlit e verifique:

- O aplicativo abre sem login para o visitante.
- Os oito campos aparecem e o formulário identifica campos ausentes.
- **Carregar exemplo → Avaliar** apresenta predição, CL, AL e o gráfico SHAP.
- **Limpar campos** remove a entrada e o resultado.
- O aplicativo também abre em uma janela anônima, confirmando o acesso público.

Depois desses passos, compartilhe o link fornecido pelo serviço. Não use um endereço ilustrativo ou o endereço local do teste no seu computador.

## Atualizações

Atualizações enviadas à branch conectada do GitHub são refletidas no aplicativo. Execute os testes antes de trocar o modelo, a base ou os parâmetros. `metadata.json` contém hashes do modelo e da base: mudanças nesses arquivos exigem atualizar os hashes e registrar uma nova versão validada.

## Hospedagem gratuita

O Community Cloud tem limites de processamento/memória e pode colocar aplicativos sem uso em repouso. Um visitante pode reativar o aplicativo pela página; nesse primeiro acesso, o carregamento pode demorar mais. A disponibilidade e os limites são determinados pelo serviço.

Fontes oficiais:

- [Publicar um aplicativo](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
- [Gerenciar o aplicativo e seus limites](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).
