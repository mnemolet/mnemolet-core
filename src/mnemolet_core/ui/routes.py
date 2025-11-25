from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from mnemolet_core.api.routes import do_search, get_answer, get_collections, get_stats

ui_router = APIRouter()

templates = Jinja2Templates(directory="src/mnemolet_core/ui/templates")

API_BASE = "http://localhost:8000"  # TODO: hardcoded url


@ui_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "error": None}
    )


@ui_router.get("/list-collections", response_class=HTMLResponse)
async def list_collections_ui(request: Request):
    data = get_collections()
    return templates.TemplateResponse(
        "list_collections.html",
        {
            "request": request,
            "collections": data.get("collections", []),
            "status": data.get("status"),
        },
    )


@ui_router.get("/stats", response_class=HTMLResponse)
async def stats_ui(request: Request, collection_name: str = "documents"):
    data = get_stats(collection_name)
    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "stats": data.get("data", []),
            "status": data.get("status"),
            "collection_name": collection_name,
        },
    )


@ui_router.get("/search", response_class=HTMLResponse)
async def search_ui(request: Request):
    return templates.TemplateResponse(
        "search.html", {"request": request, "results": None}
    )


@ui_router.post("/search", response_class=HTMLResponse)
async def search_ui_post(request: Request, query: str = Form(...)):
    data = do_search(query)
    return templates.TemplateResponse(
        "search.html",
        {"request": request, "results": data.get("results", []), "query": query},
    )


@ui_router.get("/answer", response_class=HTMLResponse)
async def answer_ui(request: Request):
    return templates.TemplateResponse(
        "answer.html", {"request": request, "results": None}
    )


@ui_router.post("/answer", response_class=HTMLResponse)
async def answer_ui_post(request: Request, query: str = Form(...)):
    data = get_answer(query)
    return templates.TemplateResponse(
        "answer.html",
        {
            "request": request,
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            "query": query,
        },
    )
