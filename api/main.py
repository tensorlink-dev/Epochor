"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from .database import init_db
from .routes import miner as miner_routes
from .routes import scoring as scoring_routes
from .routes import validator as validator_routes


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="Epochor Platform API")
    app.include_router(miner_routes.router)
    app.include_router(validator_routes.router)
    app.include_router(scoring_routes.router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
