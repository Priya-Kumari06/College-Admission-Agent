"""
IBM Watsonx.ai LLM Service using IBM Granite Model
Module 5: LLM Response Generation
"""
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from typing import List, Dict, Optional
from app.core.config import settings


ADMISSION_SYSTEM_PROMPT = """You are an official College Admission Assistant powered by IBM Granite AI.
Your role is to help prospective students with accurate admission-related information.

STRICT RULES:
1. ONLY answer questions based on the provided admission documents and context.
2. If information is not in the context, respond: "I could not find this information in the official admission documents. Please contact the admission office directly."
3. NEVER fabricate course names, fee amounts, deadlines, or eligibility criteria.
4. Always cite the source document and page when answering.
5. Be professional, helpful, and concise.
6. Do NOT share any personal student data, staff information, or internal credentials.
7. If asked about anything outside admission scope, politely redirect the student.

You represent the institution's official admission process. Accuracy is paramount."""


def build_rag_prompt(
    query: str,
    context_chunks: List[Dict],
    chat_history: List[Dict] = None
) -> str:
    """Build the RAG prompt with context, history, and query."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        doc_name = chunk["metadata"].get("document_name", "Unknown Document")
        page_num = chunk["metadata"].get("page_number", "N/A")
        score = chunk.get("relevance_score", 0)
        context_text += (
            f"\n[Source {i}: {doc_name}, Page {page_num} | Relevance: {score:.0%}]\n"
            f"{chunk['content']}\n"
            f"{'─' * 60}\n"
        )

    history_text = ""
    if chat_history:
        history_text = "\n=== CONVERSATION HISTORY ===\n"
        for msg in chat_history[-6:]:  # Last 3 turns
            role = "Student" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "=" * 30 + "\n"

    prompt = f"""{ADMISSION_SYSTEM_PROMPT}

=== RETRIEVED ADMISSION DOCUMENTS (USE ONLY THESE) ===
{context_text}
{history_text}
=== STUDENT QUESTION ===
{query}

=== YOUR ANSWER (cite sources, be accurate) ===
"""
    return prompt


class WatsonxLLMService:
    def __init__(self):
        self._model = None

    def initialize(self):
        """Initialize IBM Watsonx.ai model."""
        credentials = Credentials(
            url=settings.watsonx_url,
            api_key=settings.watsonx_api_key
        )
        client = APIClient(credentials)
        self._model = ModelInference(
            model_id=settings.watsonx_model_id,
            api_client=client,
            project_id=settings.watsonx_project_id,
            params={
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.MIN_NEW_TOKENS: 50,
                GenParams.TEMPERATURE: 0.1,  # Low temperature for factual accuracy
                GenParams.TOP_P: 0.9,
                GenParams.TOP_K: 50,
                GenParams.REPETITION_PENALTY: 1.1,
                GenParams.STOP_SEQUENCES: ["Student:", "=== STUDENT QUESTION ==="]
            }
        )
        print(f"IBM Watsonx initialized with model: {settings.watsonx_model_id}")

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        if self._model is None:
            raise RuntimeError("Watsonx LLM not initialized. Call initialize() first.")
        response = self._model.generate_text(prompt=prompt)
        return response.strip() if response else ""

    def generate_with_context(
        self,
        query: str,
        context_chunks: List[Dict],
        chat_history: List[Dict] = None
    ) -> str:
        """Build prompt and generate response with full RAG context."""
        prompt = build_rag_prompt(query, context_chunks, chat_history)
        return self.generate(prompt)


# Singleton instance
watsonx_llm = WatsonxLLMService()
