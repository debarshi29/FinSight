from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.guardrails import guardrails_middleware
from api.routes import eval as eval_router
from api.routes import ingest as ingest_router
from api.routes import query as query_router
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
    app.middleware("http")(guardrails_middleware)

    app.include_router(ingest_router.router)
    app.include_router(query_router.router)
    app.include_router(eval_router.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": settings.groq_model}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
