import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from mnemolet_core.config import (
    EMBED_MODEL,
    MIN_SCORE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SIZE_CHARS,
    TOP_K,
    UPLOAD_DIR,
)
from mnemolet_core.core.utils.qdrant import QdrantManager

logger = logging.getLogger(__name__)

app = FastAPI(title="MnemoLet API", version="0.0.1")
api_router = APIRouter()


@api_router.post("/ingest")
async def ingest_files(
    files: list[UploadFile] = File(...),
    force: bool = Query(
        False, description="Recreate Qdrant collection before ingestion"
    ),
):
    """
    Ingest multiple files into Qdrant.
    """
    saved_files, result = await do_ingestion(files, force)

    return {
        "status": "ok",
        "uploaded": saved_files,
        "force": force,
        "message": "Ingestion complete",
        "ingestion": {
            "files": result["files"],
            "chunks": result["chunks"],
            "time": result["time"],
        },
    }


async def do_ingestion(files, force: bool = False):
    from mnemolet_core.core.ingestion.ingest import ingest

    saved_files = []

    for f in files:
        dest = UPLOAD_DIR / f.filename

        content = await f.read()
        dest.write_bytes(content)

        saved_files.append(str(dest))

    batch_size = 100  # TODO: move to config.toml
    result = ingest(
        UPLOAD_DIR, batch_size, QDRANT_URL, QDRANT_COLLECTION, SIZE_CHARS, force=force
    )

    return saved_files, result


@api_router.get("/search")
def search(
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    top_k: int = TOP_K,
):
    """
    Search documents in Qdrant.
    """
    return do_search(query, qdrant_url, collection_name, embed_model, top_k)


def do_search(
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    top_k: int = TOP_K,
):
    from mnemolet_core.core.query.retrieval.search_documents import search_documents

    try:
        results = search_documents(
            qdrant_url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            embed_model=EMBED_MODEL,
            query=query,
            top_k=top_k,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@api_router.get("/answer")
def answer(
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    top_k: int = TOP_K,
):
    return StreamingResponse(
        get_answer(
            query,
            qdrant_url,
            collection_name,
            embed_model,
            ollama_url,
            ollama_model,
            top_k,
        ),
        media_type="application/json",
    )


def get_answer(
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Generate answer from local LLM.
    """
    from mnemolet_core.core.query.generation.generate_answer import generate_answer

    try:
        for chunk, sources in generate_answer(
            qdrant_url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            embed_model=EMBED_MODEL,
            ollama_url=OLLAMA_URL,
            model=ollama_model,
            query=query,
            top_k=top_k,
            min_score=MIN_SCORE,
        ):
            if chunk:
                # answer_chunks.append(answer)
                logger.info(f"Chunk: {chunk}")
                yield (json.dumps({"type": "chunk", "data": chunk}) + "\n").encode(
                    "utf-8"
                )
            elif sources:
                yield (json.dumps({"type": "sources", "data": sources}) + "\n").encode(
                    "utf-8"
                )

    except Exception as e:
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"


@api_router.get("/stats")
def stats(collection_name: str):
    return get_stats(collection_name)


def get_stats(collection_name: str):
    """
    Output statistics about Qdrant database.
    """
    try:
        qm = QdrantManager(QDRANT_URL)
        stats = qm.get_collection_stats(collection_name)
        return {"status": "success", "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats:{str(e)}")


@api_router.get("/list-collections")
def list_collections():
    return get_collections()


def get_collections():
    """
    List all Qdrant collections.
    """
    try:
        qm = QdrantManager(QDRANT_URL)
        xz = qm.list_collections()
        if not xz:
            return {
                "status": "success",
                "collections": [],
                "message": "No collections found.",
            }
        return {"status": "success", "collections": xz}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections:{str(e)}"
        )
