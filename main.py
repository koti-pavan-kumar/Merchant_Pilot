from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from pathlib import Path

from config import get_settings
from api.health import router as health_router
from api.actions import router as actions_router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered merchant health monitoring and growth automation",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/health", tags=["Health"])
app.include_router(actions_router, prefix="/api/actions", tags=["Actions"])


@app.get("/", response_class=HTMLResponse)
async def root():
    return _serve_dashboard()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return _serve_dashboard()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "app_name": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get("/api/info")
async def api_info():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "dashboard": "/dashboard",
    }


def _serve_dashboard():
    """Read and return the dashboard HTML file on every request."""
    candidates = [
        Path(__file__).resolve().parent / "dashboard" / "index.html",
        Path.cwd() / "dashboard" / "index.html",
    ]
    for path in candidates:
        try:
            if path.is_file():
                html = path.read_text(encoding="utf-8")
                return HTMLResponse(content=html, status_code=200)
        except Exception as e:
            continue

    # Absolute fallback so the route never 500s
    fallback = """
    <html><body>
    <h1>MerchantPilot AI</h1>
    <p>Dashboard file not found. <a href="/docs">Open API Docs</a></p>
    </body></html>
    """
    return HTMLResponse(content=fallback, status_code=200)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
