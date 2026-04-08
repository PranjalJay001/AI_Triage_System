"""
app.py — HuggingFace Spaces entry point for Hospital Triage OpenEnv.

HF Spaces looks for `app.py` in the repo root and expects the FastAPI/Gradio
app object to be importable as `app`, or it will execute the file directly.
This module re-exports the FastAPI `app` from server.py and launches uvicorn
when run as __main__, making it work both ways.
"""

import os

# Re-export the FastAPI application so HF Spaces can discover it via import.
from server import app  # noqa: F401  (must be importable as `app`)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
