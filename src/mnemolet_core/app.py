from fastapi import FastAPI

from mnemolet_core.api.routes import api_router
from mnemolet_core.ui.routes import ui_router

app = FastAPI()

# API
app.include_router(api_router, prefix="/api")

# UI
app.include_router(ui_router)
