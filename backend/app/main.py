import uuid
import time
import json
from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.agent import AgentState, EngineMode, OutputDepth, rag_app
from app.config import settings
from app.Normalization import get_logical_group_id
from app.converters import (
    EXPORTS_DIR,
    convert_docx_to_pdf_exact,
    convert_pdf_to_docx_exact,
    export_to_csv,
    export_to_docx,
    export_to_excel,
    export_to_pdf,
    export_to_pptx,
    extract_content_from_file,
)
from app.memory import memory_manager
from app.rag import rag_engine
from app.system_stats import fetch_system_metrics

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount exports directory for direct static access
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")


# 2. Forced attachment download endpoint
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"File '{filename}' was not found in exports."
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str
    mode: Literal["chroma_only", "model_only", "hybrid"] = "hybrid"
    detail_level: Literal["short", "detailed", "full"] = "detailed"


class FeedbackRequest(BaseModel):
    message_id: int
    feedback: int


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "model": settings.OLLAMA_MODEL}


@app.get("/api/system/stats")
def system_stats() -> Dict[str, Any]:
    """Provides real-time DB disk occupancy and accuracy metrics."""
    return fetch_system_metrics()


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest) -> Dict[str, str]:
    memory_manager.record_feedback(req.message_id, req.feedback)
    return {"status": "feedback_recorded"}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing or invalid")

    original_filename = file.filename
    logical_group = get_logical_group_id(original_filename)
    raw_content = await file.read()

    extracted_text = extract_content_from_file(original_filename, raw_content)

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text. For images, verify Tesseract OCR is installed.",
        )

    # 1. Version-Aware Purge: Remove stale chunks matching the logical group across both stores
    deleted_chroma = 0
    try:
        logical_existing = rag_engine.collection.get(where={"logical_group": logical_group})
        if logical_existing and logical_existing.get("ids"):
            rag_engine.collection.delete(ids=logical_existing["ids"])
            deleted_chroma = len(logical_existing["ids"])
    except Exception:
        pass

    deleted_fts = memory_manager.delete_logical_group_chunks_fts(logical_group)

    # 2. Chunk text with safe overlap buffer
    chunk_size = 1500
    overlap = 250
    chunks: list[str] = []
    start = 0
    text_length = len(extracted_text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(extracted_text[start:end])
        start += chunk_size - overlap

    # 3. Dual-Store Ingestion with Logical Mapping
    memory_manager.index_document_chunks_fts(original_filename, logical_group, chunks)

    doc_ids = [f"{logical_group}_chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [
        {
            "filename": original_filename,
            "logical_group": logical_group,
            "chunk_index": idx,
            "doc_id": logical_group,
        }
        for idx in range(len(chunks))
    ]

    rag_engine.add_documents_batch(
        doc_ids=doc_ids,
        texts=chunks,
        metadatas=metadatas,
    )

    # 4. Reclaim disk space immediately by vacuuming SQLite storage
    memory_manager.vacuum_database()

    return {
        "status": "success",
        "filename": original_filename,
        "logical_group": logical_group,
        "replaced_previous_chunks": deleted_chroma + deleted_fts,
        "chunks_indexed": len(chunks),
        "fts_indexed": True,
        "preview": extracted_text[:200],
    }


@app.delete("/api/documents/{filename}")
def delete_document_endpoint(filename: str) -> Dict[str, Any]:
    """Purges a file and its chunks completely across ChromaDB and SQLite FTS5."""
    chroma_deleted = rag_engine.delete_file_documents(filename)
    fts_deleted = memory_manager.delete_document_chunks_fts(filename)
    return {
        "status": "deleted",
        "filename": filename,
        "chunks_removed": chroma_deleted,
        "fts_records_removed": fts_deleted,
    }


@app.post("/api/convert")
async def convert_document(
    file: UploadFile = File(...), target_format: str = Form(...)
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    safe_filename: str = file.filename
    raw_content = await file.read()
    base_name = safe_filename.rsplit(".", 1)[0]
    source_ext = safe_filename.lower().rsplit(".", 1)[-1]
    fmt = target_format.lower().strip()

    def _process() -> str:
        if source_ext == "pdf" and fmt in ["docx", "word"]:
            return convert_pdf_to_docx_exact(raw_content, base_name)
        elif source_ext in ["docx", "doc"] and fmt == "pdf":
            return convert_docx_to_pdf_exact(raw_content, base_name)
        else:
            extracted_text = extract_content_from_file(safe_filename, raw_content)
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No readable content found")
            if fmt in ["docx", "word"]:
                return export_to_docx(extracted_text, base_name)
            elif fmt in ["pptx", "powerpoint", "ppt"]:
                return export_to_pptx(extracted_text, base_name)
            elif fmt in ["xlsx", "excel"]:
                return export_to_excel(extracted_text, base_name)
            elif fmt == "csv":
                return export_to_csv(extracted_text, base_name)
            elif fmt == "pdf":
                return export_to_pdf(extracted_text, base_name)
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    generated_filename = await run_in_threadpool(_process)

    return {
        "status": "success",
        "converted_file": generated_filename,
        "download_url": f"http://localhost:8000/api/download/{generated_filename}",
    }


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    try:
        memory_manager.save_message(req.user_id, "user", req.message)

        initial_state: AgentState = {
            "user_id": req.user_id,
            "input": req.message,
            "mode": req.mode,
            "detail_level": req.detail_level,
            "history": [],
            "profile": {},
            "retrieved_docs": [],
            "retrieval_accuracy": 0.0,
            "grounding_score": 0.0,
            "response": "",
            "latency": 0.0,
        }

        async def event_generator():
            full_response = ""
            start_time = time.time()

            result = await run_in_threadpool(rag_app.invoke, initial_state)
            response_text = str(result.get("response", ""))

            # Simulate streaming words/tokens smoothly to the client
            words = response_text.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            elapsed_time = time.time() - start_time
            total_tokens = len(full_response.split())
            tokens_per_sec = round(total_tokens / max(elapsed_time, 0.1), 1)

            # Save final message to history after streaming finishes
            ai_message_id = memory_manager.save_message(
                req.user_id, "assistant", full_response
            )

            # Send final metadata payload including token speed
            metadata = {
                "done": True,
                "message_id": ai_message_id,
                "profile": result.get("profile", {}),
                "retrieval_accuracy": result.get("retrieval_accuracy", 0.0),
                "grounding_score": result.get("grounding_score", 0.0),
                "tokens_per_second": tokens_per_sec,
            }
            yield f"data: {json.dumps(metadata)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))