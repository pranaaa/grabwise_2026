"""FastAPI entrypoint for GrabWise.

Run from project root with the venv active:
    uvicorn backend.main:app --reload

Then open http://localhost:8000 in a browser.
"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.db.database import init_db
from backend.api import chat as chat_api
from backend.api import users as users_api
from backend.api import auth as auth_api
from backend.api import admin as admin_api
from backend.api import driver_dash as driver_dash_api
from backend.api import driver_planner as driver_planner_api
from backend.api import merchant_dash as merchant_dash_api
from backend.api import customer_dash as customer_dash_api
from backend.api import llm as llm_api
from backend.llm.bedrock import llm_provider_name


app = FastAPI(title="GrabWise", version="0.3.0")

# Initialize DB on startup so first request doesn't pay the cost.
init_db()

# API routes
app.include_router(auth_api.router)
app.include_router(chat_api.router)
app.include_router(users_api.router)
app.include_router(admin_api.router)
app.include_router(driver_dash_api.router)
app.include_router(driver_planner_api.router)
app.include_router(merchant_dash_api.router)
app.include_router(customer_dash_api.router)
app.include_router(llm_api.router)

# Static UI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"ok": True, "ui": "not built", "provider": llm_provider_name()}


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "provider": llm_provider_name()}
