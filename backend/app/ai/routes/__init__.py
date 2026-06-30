"""API routes for the AI Intelligence Layer.

Combines all AI sub-routers (rag, recommend, improve, agent, sync)
into a single router that can be included in the main FastAPI application.
"""

from fastapi import APIRouter

from app.ai.routes.agent import router as agent_router
from app.ai.routes.improve import router as improve_router
from app.ai.routes.rag import router as rag_router
from app.ai.routes.recommend import router as recommend_router
from app.ai.routes.sync import router as sync_router

ai_router = APIRouter()

ai_router.include_router(rag_router)
ai_router.include_router(recommend_router)
ai_router.include_router(improve_router)
ai_router.include_router(agent_router)
ai_router.include_router(sync_router)
