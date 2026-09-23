import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import chromadb
from chromadb.api.types import Documents, Embeddings, Metadata
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

from app.config import settings

CHROMA_DATA_PATH = Path(settings.DATABASE_PATH).parent / "chroma_db"
CHROMA_DATA_PATH.mkdir(parents=True, exist_ok=True)


def _sigmoid(x: float) -> float:
    """Scales unbounded cross-encoder logits into a normalized 0.0 - 1.0 confidence score."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


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

        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self.reranker = None

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
        formatted_doc = f"search_document: {text}"
        vector = self.embeddings.embed_query(formatted_doc)
        meta_dict: Metadata = cast(Metadata, metadata or {})
        self.collection.upsert(
            ids=[doc_id],
            embeddings=cast(Embeddings, [vector]),
            documents=[text],
            metadatas=[meta_dict],
        )

    def add_documents_batch(
        self,
        doc_ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not texts:
            return
        formatted_texts = [f"search_document: {t}" for t in texts]
        raw_vectors = self.embeddings.embed_documents(formatted_texts)
        typed_vectors = cast(Embeddings, raw_vectors)
        typed_docs = cast(Documents, texts)
        clean_metas = cast(
            List[Metadata],
            metadatas if metadatas is not None else [{} for _ in texts],
        )
        self.collection.upsert(
            ids=doc_ids,
            embeddings=typed_vectors,
            documents=typed_docs,
            metadatas=clean_metas,
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.30,
        filename_filter: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        formatted_query = f"search_query: {query.strip()}"
        query_vector = self.embeddings.embed_query(formatted_query)

        fetch_k = max(top_k * 4, 16)
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": fetch_k,
        }

        if filename_filter:
            query_kwargs["where"] = {"filename": filename_filter}

        try:
            results = self.collection.query(**query_kwargs)
        except Exception:
            if "where" in query_kwargs:
                query_kwargs.pop("where")
                results = self.collection.query(**query_kwargs)
            else:
                return [], 0.0

        raw_docs = results.get("documents")
        raw_distances = results.get("distances")
        raw_metas = results.get("metadatas")

        docs = cast(List[str], raw_docs[0] if raw_docs and len(raw_docs) > 0 else [])
        distances = cast(List[float], raw_distances[0] if raw_distances and len(raw_distances) > 0 else [])
        metas = cast(List[Dict[str, Any]], raw_metas[0] if raw_metas and len(raw_metas) > 0 else [])
        safe_metas: List[Dict[str, Any]] = metas if metas else [{} for _ in docs]

        candidate_items: List[Dict[str, Any]] = []
        seen_hashes = set()

        for doc, dist, meta in zip(docs, distances, safe_metas):
            doc_hash = self.compute_hash(doc)
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            # Cosine distance to similarity conversion
            similarity = max(0.0, 1.0 - float(dist))
            if similarity < min_similarity:
                continue

            candidate_items.append({
                "raw_text": doc,
                "metadata": meta,
                "similarity": round(similarity, 4),
            })

        if not candidate_items:
            return [], 0.0

        # High-precision Cross-Encoder Reranking
        if self.reranker and candidate_items:
            try:
                pairs = [[query, item["raw_text"]] for item in candidate_items]
                raw_scores = self.reranker.predict(pairs)
                for item, raw_s in zip(candidate_items, raw_scores):
                    norm_score = _sigmoid(float(raw_s))
                    item["rerank_score"] = round(norm_score, 4)
                    item["final_score"] = round((item["similarity"] * 0.4) + (norm_score * 0.6), 4)

                candidate_items.sort(key=lambda x: x["final_score"], reverse=True)
            except Exception:
                candidate_items.sort(key=lambda x: x["similarity"], reverse=True)
        else:
            candidate_items.sort(key=lambda x: x["similarity"], reverse=True)

        final_items = candidate_items[:top_k]
        retrieved_items: List[Dict[str, Any]] = []
        accuracies: List[float] = []

        for item in final_items:
            meta = item["metadata"]
            doc = item["raw_text"]
            score = item.get("final_score", item["similarity"])
            accuracies.append(score)

            source_tag = f"[{meta.get('filename', 'Document')} - Chunk {meta.get('chunk_index', 0)}]"
            enriched_content = f"{source_tag}\n{doc}"

            retrieved_items.append({
                "content": enriched_content,
                "raw_text": doc,
                "metadata": meta,
                "similarity": item["similarity"],
                "score": score,
            })

        avg_accuracy = (sum(accuracies) / len(accuracies)) if accuracies else 0.0
        return retrieved_items, avg_accuracy


rag_engine = RAGEngine()