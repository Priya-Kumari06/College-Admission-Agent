that from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.services.rag.vector_store import vector_store
from app.services.rag.llm_service import watsonx_llm
from app.api.routes import auth, chat, documents, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    print("🚀 Starting College Admission Assistant API...")
    await connect_to_mongo()
    vector_store.initialize()
    watsonx_llm.initialize()
    print("✅ All services initialized successfully")
    yield
    print("🛑 Shutting down...")
    await close_mongo_connection()


app = FastAPI(
    title="AI-Powered College Admission Assistant",
    description=(
        "Intelligent RAG-based admission assistant powered by "
        "IBM Watsonx Granite with enterprise-grade guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "College Admission Assistant API",
        "version": "1.0.0",
        "status": "running",
        "powered_by": "IBM Watsonx Granite + LangChain + ChromaDB"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "vector_store": vector_store.get_collection_stats(),
        "llm": settings.watsonx_model_id,
        "environment": settings.app_env
    }
