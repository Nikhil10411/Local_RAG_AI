import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import chromadb
from chromadb.api.types import Metadata
from langchain_ollama import OllamaEmbeddings
from app.config import settings

CHROMA_DATA_PATH = Path(settings.DATABASE_PATH).parent / "chroma_db"
CHROMA_DATA_PATH.mkdir(parents=True, exist_ok=True)


class RAGEngine:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model="nomic-embed-text",
        )
        self.collection = self.client.get_or_create_collection(
            name="local_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def compute_hash(self, content: str) -> str:
        """Generates a deterministic 16-character SHA-256 fingerprint."""
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]

    def delete_file_documents(self, filename: str) -> int:
        """Removes all stored vectors associated with a specific filename."""
        try:
            existing = self.collection.get(where={"filename": filename})
            ids_to_del = existing.get("ids", [])
            if ids_to_del:
                self.collection.delete(ids=ids_to_del)
                return len(ids_to_del)
        except Exception:
            pass
        return 0

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Embeds and upserts a single chunk with task prefix."""
        formatted_doc = f"search_document: {text}"
        vector = self.embeddings.embed_query(formatted_doc)
        meta_dict: Metadata = cast(Metadata, metadata or {})
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[meta_dict],
        )

    def add_documents_batch(
        self,
        doc_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch embeds chunks in a single pass without type issues."""
        if not texts:
            return
        formatted_texts = [f"search_document: {t}" for t in texts]
        vectors = self.embeddings.embed_documents(formatted_texts)
        clean_metas = cast(
            List[Metadata],
            metadatas if metadatas is not None else [{} for _ in texts]
        )
        self.collection.upsert(
            ids=doc_ids,
            embeddings=vectors,  # type: ignore[arg-type]
            documents=texts,
            metadatas=clean_metas,
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.40,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Oversamples ChromaDB, removes duplicates, filters low-confidence noise,
        and returns enriched chunks with proper static type casting.
        """
        formatted_query = f"search_query: {query.strip()}"
        query_vector = self.embeddings.embed_query(formatted_query)

        # Fetch extra chunks to guarantee top_k unique results after deduplication
        fetch_k = max(top_k * 2, 8)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
        )

        raw_docs = results.get("documents")
        raw_distances = results.get("distances")
        raw_metas = results.get("metadatas")

        # Explicitly cast to prevent static type checker warnings
        docs = cast(List[str], raw_docs[0] if raw_docs and len(raw_docs) > 0 else [])
        distances = cast(List[float], raw_distances[0] if raw_distances and len(raw_distances) > 0 else [])
        metas = cast(List[Dict[str, Any]], raw_metas[0] if raw_metas and len(raw_metas) > 0 else [])

        safe_metas: List[Dict[str, Any]] = metas if metas else [{} for _ in docs]

        retrieved_items: List[Dict[str, Any]] = []
        similarities: List[float] = []
        seen_hashes = set()

        for doc, dist, meta in zip(docs, distances, safe_metas):
            doc_hash = self.compute_hash(doc)
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            # Cosine distance to similarity: 0.0 (identical) to 2.0 (opposite)
            similarity = max(0.0, 1.0 - float(dist))

            if similarity < min_similarity:
                continue

            similarities.append(similarity)

            # Source provenance header
            source_tag = f"[{meta.get('filename', 'Document')} - Chunk {meta.get('chunk_index', 0)}]"
            enriched_content = f"{source_tag}\n{doc}"

            retrieved_items.append({
                "content": enriched_content,
                "raw_text": doc,
                "metadata": meta,
                "similarity": round(similarity, 4),
            })

            if len(retrieved_items) >= top_k:
                break

        avg_retrieval_accuracy = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )
        return retrieved_items, avg_retrieval_accuracy


rag_engine = RAGEngine()