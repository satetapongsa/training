import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.db.session import init_db
from app.api.v1 import api_v1_router
from app.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing AI Vision Training Studio services...")
    settings.init_directories()
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down AI Vision Training Studio...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production-grade AI Vision Training Platform",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please check server logs for details.",
            "error_type": type(exc).__name__,
            "path": str(request.url.path),
        },
    )


# Register Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)

# Mount Static Files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount React Frontend Build (if present)
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/favicon.svg")
async def get_favicon():
    if frontend_dist.exists() and (frontend_dist / "favicon.svg").exists():
        return FileResponse(str(frontend_dist / "favicon.svg"))
    fav_static = static_dir / "favicon.svg"
    if fav_static.exists():
        return FileResponse(str(fav_static))
    return JSONResponse(status_code=404, content={"detail": "Favicon not found"})


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root_index():
    """Serves the Production Single Page Application."""
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(str(frontend_dist / "index.html"))
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "status": "online",
        "name": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
    }


@app.get("/{full_path:path}")
async def catch_all_spa(full_path: str):
    """Fallback SPA routing for React client-side navigation."""
    if full_path.startswith(("api", "ws", "docs", "redoc", "static", "assets")):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(str(frontend_dist / "index.html"))
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(status_code=404, content={"detail": "Not found"})

