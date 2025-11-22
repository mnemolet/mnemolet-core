from fastapi import FastAPI

from mnemolet_core.ui.routes import router

app = FastAPI(title="MnemoLet UI")

app.include_router(router)
