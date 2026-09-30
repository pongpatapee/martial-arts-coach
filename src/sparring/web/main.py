"""ASGI entry point for `uvicorn sparring.web.main:app` (used by `sparring serve --reload` and
production servers). Configured through environment variables."""

import os
from pathlib import Path

from sparring.web.app import create_app

app = create_app(
    Path(os.environ.get("SPARRING_DATA", "data")),
    max_upload_mb=int(os.environ.get("SPARRING_MAX_UPLOAD_MB", "2048")),
)
