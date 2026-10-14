"""ASGI entrypoint: ``uvicorn tdeez_ai.main:app``."""

from tdeez_ai.app import create_app

app = create_app()
