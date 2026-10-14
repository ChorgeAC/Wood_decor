"""AWS Lambda entrypoint (API Gateway HTTP API payload v2) via Mangum."""

from mangum import Mangum

from tdeez_ai.main import app

handler = Mangum(app, lifespan="off")
