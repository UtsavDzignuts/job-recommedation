"""Main FastAPI application entry point for the KRA-KPA Job Board Platform.

Registers the AI Intelligence Layer routes and configures middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.routes import ai_router

app = FastAPI(
    title="KRA-KPA Job Board Platform",
    description=(
        "Job Board Platform with AI Intelligence Layer providing RAG-based Q&A, "
        "job recommendations, description improvement, and autonomous AI agent capabilities."
    ),
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register AI routes at root level (no prefix)
app.include_router(ai_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Health check endpoint.

    Returns a simple status object indicating the service is running.
    """
    return {"status": "ok"}
