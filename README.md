# mnemolet-core

**MnemoLet** is a minimal, end-to-end **Retrieval-Augmented Generation (RAG)**
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

Before running the container, create a storage directory:

`$ mkdir -p ./qdrant_storage`

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

## Installation

MnemoLet requires Python and [uv](https://uv.run/) for managing virtual environments and dependencies.

1. **clone the repo**:

2. **install uv**

3. **create virtual environment and install dependencies**

```
$ uv init
$ uv sync
```

## Configuration

MnemoLet uses a `config.toml` file to configure Qdrant, 
embeddings, Ollama, and local storage.

Below is default configuration:

```
[qdrant]
host = "localhost"
port = 6333
collection = "documents"

[embedding]
model = "all-MiniLM-L6-v2"
batch_size = 100

[ollama]
host = "localhost"
port = 11434

[storage]
db_path = "./data/tracker.sqlite"
```

## CLI

**Note:** Before using the CLI or API, make sure the Qdrant server is running.

### Help

`uv run python -m cli.main --help`

### Ingest Files

`uv run python -m cli.main ingest <directory>`

or

add `--force` to re-ingest files and recreate Qdrant collection

`uv run python -m cli.main ingest <directory> --force`

### Search in Qdrant Collection

`uv run python -m cli.main search "<query>" --top-k <number of results>`

`--top-k`: [optional]

### Generate Answer

`uv run python -m cli.main answer "<query>" --top-k <number of results>`

`--top-k`: [optional]

## API

The API is implemented using [FastAPI](https://fastapi.tiangolo.com/).

### Available Endpoint

- **`GET /search`**: Search indexed texts.

**Query Parameters:**
- `query` (str) - search query.
- `top_k` (int, optional) - number of results to return (default: 3).

- **`GET /answer`**: Search indexed texts.

**Query Parameters:**
- `query` (str) - search query.
- `top_k` (int, optional) - number of results to return (default: 3).

### Running the API

Start the FastAPI server with:

`uv run uvicorn api.server:app --reload`

### Testing the Endpoint

#### Search

`curl "http://127.0.0.1:8000/search?query=<query>"`

or 

`curl "http://127.0.0.1:8000/search?query=<query>&top_k=2"`

#### Answer

`curl "http://127.0.0.1:8000/answer?query=<query>"`

or 

`curl "http://127.0.0.1:8000/answer?query=<query>&top_k=2"`
