from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.middleware.guardrails import GuardrailsMiddleware
from api.routes import eval as eval_router
from api.routes import ingest as ingest_router
from api.routes import metrics as metrics_router
from api.routes import query as query_router
from api.routes import query_stream as query_stream_router
from core.config import settings
from observability.tracer import setup_tracing

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("finsight.startup", model=settings.groq_model)
    setup_tracing()
    yield
    log.info("finsight.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinSight",
        description="Regulatory-Grade Financial Intelligence System",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GuardrailsMiddleware)

    app.include_router(ingest_router.router)
    app.include_router(query_router.router)
    app.include_router(query_stream_router.router)
    app.include_router(eval_router.router)
    app.include_router(metrics_router.router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        _index = static_dir / "index.html"

        @app.get("/ui", include_in_schema=False)
        @app.get("/ui/", include_in_schema=False)
        async def ui_index():
            return FileResponse(
                str(_index), headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
            )

        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")

        dashboard_file = static_dir / "dashboard.html"
        if dashboard_file.exists():

            @app.get("/dashboard", include_in_schema=False)
            async def dashboard():
                return FileResponse(str(dashboard_file))

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": settings.groq_model}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
