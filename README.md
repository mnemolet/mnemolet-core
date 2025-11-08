# mnemolet-core

**Mnemolet** is a minimal, end-to-end **Retrieval-Augmented Generation (RAG)**
system that runs entirely **locally**: no API keys, no cloud.

## MVP Goals

### Core objectives

- Ingest plain `.txt` files
- Chunk text with configurable overlap
- Embed chunks using a local sentence-transformer model
- Store embeddings in a local Qdrant vector database
- Track ingested files and hashes in a local SQLite database
- Retrieve top-k most relevant chunks for a query
- Generate answers using a local LLM (via [Ollama](https://ollama.ai) or compatible API)
- Provide a clean command-line interface (`ingest`, `query`)

## Run Qdrant

with docker:

```
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

with podman:

```
podman run -p 6333:6333 \
    -v ./qdrant_storage:/qdrant/storage \
    docker.io/qdrant/qdrant
```

## API

The API is implemented using [FastAPI](https://fastapi.tiangolo.com/).

### Available Endpoint

- **`GET /search`**: Search indexed texts.

**Query Parameters:**
- `query` (str) - search query.
- `top_k` (int, optional) - number of results to return (default: 3).

### Running the API

Start the FastAPI server with:
`uv run uvicorn api.server:app --reload`

### Testing the Endpoint

`curl "http://127.0.0.1:8000/search?query=<query>"`

or 

`curl "http://127.0.0.1:8000/search?query=<query>&top_k=2"`
