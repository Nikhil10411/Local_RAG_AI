import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


class MemoryManager:
    def __init__(self) -> None:
        self.db_path = settings.DATABASE_PATH
        # Automatically create the parent directory if it does not exist yet
        Path(self.db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_fts()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # Enable Write-Ahead Logging (WAL) for rapid concurrent reading/writing
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Episodic Memory (Raw chat interactions)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    role TEXT,
                    content TEXT,
                    feedback INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 2. Semantic User Identity (Persistent facts, preferences, and learned rules)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_identity (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 3. Telemetry & RAG Performance Metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    retrieval_similarity REAL,
                    grounding_score REAL,
                    user_feedback INTEGER DEFAULT 0,
                    latency_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _init_fts(self) -> None:
        """Initializes SQLite Full-Text Search 5 (FTS5) for sub-millisecond lexical chunk retrieval."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                    doc_id UNINDEXED,
                    filename UNINDEXED,
                    chunk_index UNINDEXED,
                    content,
                    tokenize = 'porter unicode61'
                )
                """
            )
            conn.commit()

    # ----------------- CHAT & FEEDBACK MEMORY -----------------

    def save_message(self, user_id: str, role: str, content: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def record_feedback(self, message_id: int, feedback: int) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_history SET feedback = ? WHERE id = ?",
                (feedback, message_id),
            )
            conn.commit()

    def get_recent_history(self, user_id: str, limit: int = 6) -> List[Dict[str, str]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": str(r[0]), "content": str(r[1])} for r in reversed(rows)]

    # ----------------- USER PROFILE & LEARNED CONSTRAINTS -----------------

    def get_user_identity(self, user_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_data FROM user_identity WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
            return {"preferences": [], "technical_level": "intermediate", "learned_rules": []}

    def update_user_identity(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_identity (user_id, profile_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data = excluded.profile_data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, json.dumps(profile_data)),
            )
            conn.commit()

    def record_rag_metric(
        self,
        user_id: str,
        query: str,
        similarity: float,
        grounding: float,
        latency: float,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rag_metrics (user_id, query, retrieval_similarity, grounding_score, latency_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, query, similarity, grounding, latency),
            )
            conn.commit()
            return cursor.lastrowid or 0

    # ----------------- SQLITE FTS5 LEXICAL SEARCH & STORAGE -----------------

    def index_document_chunks_fts(self, filename: str, chunks: List[str]) -> int:
        """
        Idempotently inserts or updates document chunks in the FTS5 virtual table.
        Purges prior chunks associated with the filename to prevent stale results.
        """
        if not chunks:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Clean out previous chunks of the same file
            cursor.execute("DELETE FROM document_fts WHERE filename = ?", (filename,))

            # 2. Insert new chunks in batch
            records = [
                (f"{filename}_chunk_{idx}", filename, idx, chunk)
                for idx, chunk in enumerate(chunks)
            ]
            cursor.executemany(
                "INSERT INTO document_fts (doc_id, filename, chunk_index, content) VALUES (?, ?, ?, ?)",
                records,
            )
            conn.commit()
            return len(records)

    def delete_document_chunks_fts(self, filename: str) -> int:
        """Removes all FTS5 indexed chunks for a deleted file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_fts WHERE filename = ?", (filename,))
            conn.commit()
            return cursor.rowcount

    def search_fts(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Performs high-speed BM25 ranked lexical retrieval using FTS5.
        Bypasses CPU vector generation completely for exact keyword matches.
        """
        # Strip special characters and filter short tokens (< 3 characters)
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query)

        # Ignore conversational stop words to prevent false lexical hits
        stop_words = {
            "what", "where", "when", "which", "who", "whom", "this", "that",
            "these", "those", "have", "from", "with", "about", "your", "tell",
            "give", "show", "current", "latest", "please", "does", "will"
        }
        search_terms = [t for t in raw_tokens if t.lower() not in stop_words]

        if not search_terms:
            return []

        # Prepare formatted query for FTS5: "token1" OR "token2" OR ...
        fts_match_query = " OR ".join([f'"{term}"' for term in search_terms])

        results: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT doc_id, filename, chunk_index, content, bm25(document_fts) AS rank
                    FROM document_fts
                    WHERE document_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (fts_match_query, limit),
                )
                rows = cursor.fetchall()
                for r in rows:
                    doc_id, filename, chunk_index, content, rank = r
                    results.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "content": f"[{filename} - Chunk {chunk_index}]\n{content}",
                        "raw_text": content,
                        "score": abs(float(rank)),
                    })
        except sqlite3.OperationalError:
            # Safe fallback if syntax parsing encounters irregular characters
            return []

        return results


memory_manager = MemoryManager()