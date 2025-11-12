from fastapi import FastAPI, HTTPException

from mnemolet_core.config import (
    EMBED_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)
from mnemolet_core.query.generation.generate_answer import generate_answer
from mnemolet_core.query.retrieval.search_documents import search_documents
from mnemolet_core.utils.qdrant import QdrantManager

app = FastAPI(title="MnemoLet API", version="0.0.1")


@app.get("/search")
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
    results = search_documents(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        embed_model=EMBED_MODEL,
        query=query,
        top_k=top_k,
    )
    return {"results": results}


@app.get("/answer")
def answer(
    query: str,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
    embed_model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    top_k: int = TOP_K,
):
    """
    Generate answer from local LLM.
    """
    answer, _ = generate_answer(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        embed_model=EMBED_MODEL,
        ollama_url=OLLAMA_URL,
        model=ollama_model,
        query=query,
        top_k=top_k,
    )
    return {"response": answer}


@app.get("/stats")
def stats(collection_name: str):
    """
    Output statistics about Qdrant database.
    """
    print("test")
    try:
        qm = QdrantManager(QDRANT_URL)
        stats = qm.get_collection_stats(collection_name)
        return {"status": "success", "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats:{str(e)}")


@app.delete("/remove")
def remove(collection_name: str):
    """
    Remove Qdrant collection.
    """
    try:
        qm = QdrantManager(QDRANT_URL)
        if not qm.collection_exists(collection_name):
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_name}' not found."
            )

        qm.remove_collection(collection_name)
        return {
            "status": "success",
            "message": f"Collection '{collection_name}' removed successfully.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove collection '{collection_name}':{str(e)}",
        )


@app.get("/list-collections")
def list_collections_cli():
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
