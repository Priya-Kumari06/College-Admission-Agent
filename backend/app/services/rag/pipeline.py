"""
RAG Orchestration Engine
Module 4: Query Retrieval Engine + Module 5: LLM Response Generation
Combines vector retrieval, LLM generation, and guardrails into one pipeline.
"""
import time
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from app.services.rag.vector_store import vector_store
from app.services.rag.llm_service import watsonx_llm
from app.services.guardrails.engine import guardrails_engine
from app.models.chat import (
    ChatRequest, ChatResponse, SourceReference,
    GuardrailStatus, QueryCategory
)
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline:
    Query → Input Guardrails → Embedding → Retrieval → LLM → Output Guardrails → Response
    """

    async def process_query(
        self,
        request: ChatRequest,
        user_id: Optional[str] = None
    ) -> Tuple[ChatResponse, Dict]:
        """
        Main RAG pipeline entry point.
        Returns (ChatResponse, audit_log_dict)
        """
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())
        violations = []

        # ── Step 1: Input Guardrails ──────────────────────────────────
        input_check = guardrails_engine.validate_query(request.message)
        if not input_check.passed:
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = ChatResponse(
                session_id=session_id,
                message=input_check.blocked_response,
                sources=[],
                confidence_score=0.0,
                query_category=QueryCategory.RESTRICTED,
                guardrail_status=input_check.status,
                human_handoff_required=False
            )
            audit = self._build_audit(
                session_id, user_id, request.message,
                input_check.blocked_response, QueryCategory.RESTRICTED,
                input_check.status, 0.0, [], False,
                elapsed_ms, input_check.violations
            )
            return response, audit

        # ── Step 2: Query Classification ─────────────────────────────
        query_category = guardrails_engine.classify_query(request.message)
        logger.info(f"Query classified as: {query_category}")

        # ── Step 3: Vector Retrieval ──────────────────────────────────
        retrieved_chunks, max_confidence = vector_store.query(
            query_text=request.message,
            n_results=5
        )
        context_found = len(retrieved_chunks) > 0 and max_confidence > 0.3

        # ── Step 4: Build Sources ─────────────────────────────────────
        sources = [
            SourceReference(
                document_name=chunk["metadata"].get("document_name", "Unknown"),
                page_number=int(chunk["metadata"].get("page_number", 0)),
                chunk_id=chunk["metadata"].get("chunk_id", ""),
                relevance_score=chunk["relevance_score"]
            )
            for chunk in retrieved_chunks
            if chunk["relevance_score"] >= 0.3
        ]

        # ── Step 5: LLM Generation ────────────────────────────────────
        if context_found:
            history = [msg.dict() for msg in request.chat_history]
            raw_response = watsonx_llm.generate_with_context(
                query=request.message,
                context_chunks=retrieved_chunks,
                chat_history=history
            )
        else:
            raw_response = ""

        # ── Step 6: Output Guardrails ─────────────────────────────────
        final_response, guardrail_status, human_handoff, out_violations = \
            guardrails_engine.validate_response(
                response=raw_response,
                confidence_score=max_confidence,
                context_found=context_found,
                query_category=query_category
            )
        violations.extend(out_violations)

        elapsed_ms = int((time.time() - start_time) * 1000)

        response = ChatResponse(
            session_id=session_id,
            message=final_response,
            sources=sources,
            confidence_score=round(max_confidence, 4),
            query_category=query_category,
            guardrail_status=guardrail_status,
            human_handoff_required=human_handoff
        )

        audit = self._build_audit(
            session_id, user_id, request.message, final_response,
            query_category, guardrail_status, max_confidence,
            sources, human_handoff, elapsed_ms, violations
        )

        logger.info(
            f"RAG pipeline complete | session={session_id} | "
            f"confidence={max_confidence:.2f} | status={guardrail_status} | "
            f"time={elapsed_ms}ms"
        )
        return response, audit

    def _build_audit(
        self, session_id, user_id, query, response,
        category, status, confidence, sources,
        human_handoff, elapsed_ms, violations
    ) -> Dict:
        return {
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "query_category": category.value if hasattr(category, "value") else category,
            "guardrail_status": status.value if hasattr(status, "value") else status,
            "confidence_score": confidence,
            "sources": [s.dict() for s in sources] if sources else [],
            "human_handoff": human_handoff,
            "response_time_ms": elapsed_ms,
            "guardrail_violations": violations,
            "timestamp": datetime.utcnow().isoformat()
        }


# Singleton pipeline
rag_pipeline = RAGPipeline()
