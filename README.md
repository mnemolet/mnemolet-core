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
