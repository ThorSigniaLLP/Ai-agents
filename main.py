"""
main.py
FastAPI application entry point for the Autonomous Research System.

Run with:
  uvicorn main:app --reload --port 8002

Or directly:
  python main.py
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Load .env before anything else
load_dotenv()

from api.routes import router
from core.config import get_settings
from core.database import init_db, close_db

import sys
import redis.asyncio as aioredis
sys.modules["aioredis"] = aioredis

from fastapi.templating import Jinja2Templates
original_template_response = Jinja2Templates.TemplateResponse

def patched_template_response(self, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str):
        name = args[0]
        context = kwargs.get("context", {})
        request = context.get("request")
        if request is not None:
            return original_template_response(self, request=request, name=name, **kwargs)
    return original_template_response(self, *args, **kwargs)

Jinja2Templates.TemplateResponse = patched_template_response

from fastapi_admin.app import app as admin_app
from fastapi_admin.providers.login import UsernamePasswordProvider

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🚀 Starting Autonomous Research System...")
    settings = get_settings()

    # Validate API keys on startup
    missing = []
    if not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.openrouter_api_key:
        missing.append("openrouter_api_key")
    if not settings.cerebras_api_key:
        missing.append("cerebras_api_key")

    if missing:
        logger.warning(f"⚠️  Missing API keys: {', '.join(missing)}")
    else:
        logger.info("✅ All API keys loaded")

    # Pre-build the LangGraph (warm up)
    try:
        from graph.research_graph import get_graph
        get_graph()
        logger.info("✅ LangGraph research graph compiled and ready")
    except Exception as e:
        logger.error(f"❌ Failed to compile research graph: {e}")

    # Init database for FastAPI Admin
    await init_db()
    
    # Init FastAPI Admin
    redis_conn = aioredis.from_url("redis://localhost:6379")
    
    from core.models import Admin
    login_provider = UsernamePasswordProvider(
        admin_model=Admin,
        login_logo_url="https://preview.tabler.io/static/logo.svg"
    )
    
    # We must import resources to register them
    import admin.resources
    
    import os
    import fastapi_admin
    from fastapi_admin.template import templates
    from fastapi_admin.depends import get_current_admin, get_resources
    from starlette.status import HTTP_401_UNAUTHORIZED
    
    @admin_app.exception_handler(HTTP_401_UNAUTHORIZED)
    async def unauthorized_exception_handler(request: Request, exc: HTTPException):
        return RedirectResponse(url="/admin/login")
    
    @admin_app.get("/")
    async def home(
        request: Request,
    ):
        admin = get_current_admin(request)
        resources = get_resources(request)
        return templates.TemplateResponse(
            "dashboard.html",
            context={
                "request": request,
                "resources": resources,
                "resource_label": "Dashboard",
                "page_pre_title": "overview",
                "page_title": "Dashboard",
            },
        )
        
    await admin_app.configure(
        logo_url="https://preview.tabler.io/static/logo.svg",
        template_folders=[
            os.path.join(os.path.dirname(fastapi_admin.__file__), "templates"),
            os.path.join(os.path.dirname(__file__), "templates")
        ],
        providers=[login_provider],
        redis=redis_conn,
    )

    yield

    await close_db()
    logger.info("🛑 Research System shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Autonomous Company Research & Intelligence System",
        description=(
            "Production-grade research system using LangGraph + Browser-Use. "
            "Researches companies via live web browsing and produces source-grounded "
            "structured JSON with confidence scores. No hallucinations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router, prefix="/api/v1", tags=["Research"])
    
    # Mount Admin
    app.mount("/admin", admin_app)

    # Root
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({
            "service": "Autonomous Research System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "endpoints": {
                "start_research": "POST /api/v1/research",
                "get_result": "GET /api/v1/research/{job_id}",
                "stream": "GET /api/v1/research/{job_id}/stream",
            },
        })

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
