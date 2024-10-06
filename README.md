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