from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ui_router = APIRouter()

templates = Jinja2Templates(directory="src/mnemolet_core/ui/templates")

API_BASE = "http://localhost:8000"  # TODO: hardcoded url


@ui_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "error": None}
    )
