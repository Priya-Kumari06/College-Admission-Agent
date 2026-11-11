"""
Vector Store Service using ChromaDB + Sentence Transformers
Module 3: Vector Database Creation
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import numpy as np

from app.core.config import settings


class VectorStoreService:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_model = None

    def initialize(self):
        """Initialize ChromaDB client and embedding model."""
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._embedding_model = SentenceTransformer(settings.embedding_model)
        print(f"VectorStore initialized with collection: {settings.chroma_collection_name}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def add_documents(self, chunks: List[Dict], doc_id: str) -> int:
        """
        Add document chunks to ChromaDB.
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        texts = [chunk["content"] for chunk in chunks]
        ids = [f"{doc_id}_{chunk['chunk_id']}" for chunk in chunks]
        metadatas = [
            {
                "document_id": doc_id,
                "document_name": chunk.get("document_name", ""),
                "page_number": str(chunk.get("page_number", 0)),
                "chunk_id": chunk["chunk_id"],
                "char_count": str(chunk.get("char_count", 0))
            }
            for chunk in chunks
        ]

        embeddings = self.generate_embeddings(texts)

        # Batch upsert (handles duplicates)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        doc_filter: Optional[str] = None
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve top-k relevant chunks for a query.
        Returns (results, max_confidence_score).
        """
        query_embedding = self.generate_embeddings([query_text])[0]

        where_filter = {"document_id": doc_filter} if doc_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted = []
        max_score = 0.0

        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # ChromaDB cosine distance: 0=identical, 2=opposite
                # Convert to similarity score 0-1
                similarity = 1 - (dist / 2)
                max_score = max(max_score, similarity)
                formatted.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(similarity, 4)
                })

        # Sort by relevance descending
        formatted.sort(key=lambda x: x["relevance_score"], reverse=True)
        return formatted, max_score

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        existing = self._collection.get(where={"document_id": doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
            return len(existing["ids"])
        return 0

    def get_collection_stats(self) -> Dict:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "total_chunks": count,
            "collection_name": settings.chroma_collection_name
        }


# Singleton instance
vector_store = VectorStoreService()
