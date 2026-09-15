import os
from pathlib import Path
from typing import Dict, Any
from app.config import settings
from app.memory import memory_manager

def get_directory_size(path: Path) -> float:
    """Returns size in Megabytes (MB)."""
    total_bytes = 0
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 3)
    for entry in path.rglob("*"):
        if entry.is_file():
            total_bytes += entry.stat().st_size
    return round(total_bytes / (1024 * 1024), 3)

def fetch_system_metrics() -> Dict[str, Any]:
    db_file = Path(settings.DATABASE_PATH)
    chroma_dir = db_file.parent / "chroma_db"
    exports_dir = db_file.parent / "exports"

    sqlite_size_mb = get_directory_size(db_file)
    chroma_size_mb = get_directory_size(chroma_dir)
    exports_size_mb = get_directory_size(exports_dir)
    total_db_storage = round(sqlite_size_mb + chroma_size_mb, 3)

    # Calculate average scores from SQLite
    avg_similarity = 0.0
    avg_grounding = 0.0
    avg_latency = 0.0
    total_queries = 0

    try:
        with memory_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*),
                    AVG(retrieval_similarity),
                    AVG(grounding_score),
                    AVG(latency_seconds)
                FROM rag_metrics
            """)
            row = cursor.fetchone()
            if row and row[0] > 0:
                total_queries = row[0]
                avg_similarity = round(float(row[1] or 0.0), 2)
                avg_grounding = round(float(row[2] or 0.0), 2)
                avg_latency = round(float(row[3] or 0.0), 2)
    except Exception:
        pass

    return {
        "storage": {
            "sqlite_mb": sqlite_size_mb,
            "chroma_mb": chroma_size_mb,
            "exports_mb": exports_size_mb,
            "total_occupied_mb": total_db_storage
        },
        "performance": {
            "total_indexed_queries": total_queries,
            "avg_retrieval_similarity": avg_similarity,
            "avg_grounding_score": avg_grounding,
            "avg_latency_seconds": avg_latency
        }
    }