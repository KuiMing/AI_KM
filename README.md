## Set `.env` file

```
AZURE_OPENAI_ENDPOINT = <your endpoint of Azure OpenAI>
AZURE_OPENAI_DEPLOYMENT_NAME = <your deployment name of gpt on Azure>
AZURE_OPENAI_KEY = <your Azure OpenAI Key>
AZURE_OPENAI_API_VERSION = <API version of Azure>
AZURE_OPENAI_Embedding_DEPLOYMENT_NAME = <your deployment name of embedding model on Azure>
OPENAI_API_KEY = <your OpenAI API key>
SOURCE = "OpenAI"
QDDRANT_URL = <your qdrant url>
BOT_URL = <your streamlit url>
```

## Poetry

### Installation

- Linux, macOS, Windows(WSL)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
- Powershell
```powershell
Invoke-WebRequest -Uri https://install.python-poetry.org
-UseBasicParsing).Content | py -
```

### Updating dependencies
```bash
poetry lock
poetry install
```

### Spawning shell
```bash
poetry shell
```

## Qdrant
- Run qdrant with docker
```bash
mkdir qdrant_storage
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

## Chatbot
```bash
streamlit run streaming.py
```

## Home page
```bash
uvicorn app:app
```