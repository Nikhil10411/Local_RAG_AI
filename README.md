# Local AI Engineering Workspace ⚡

A fully offline, privacy-focused engineering workspace and document intelligence agent powered by **FastAPI**, **LangGraph**, **Ollama (Llama-3-8B)**, **ChromaDB**, and **SQLite FTS5**, featuring a dynamic **React** dashboard.

The system runs locally on commodity CPU hardware with zero external API calls, zero telemetry, grounded retrieval verification, and a non-blocking document layout conversion pipeline.

---

## Key Capabilities

* **100% Air-Gapped Operation:** Direct document parsing, vector indexing, and token inference occur strictly on the host system.
* **Deterministic CPU Execution:** Runs quantized Llama-3-8B (`Q4_K_M`) using greedy token decoding (`temperature=0.0`) and bounded context (`num_ctx=2048`) to eliminate sampling drift.
* **Hybrid Two-Tier Retrieval:**
  * **Tier 1 (Lexical Match):** SQLite FTS5 index executes queries in under 2ms. If search keywords do not appear in the ingested text, false vector associations are prevented.
  * **Tier 2 (Dense Vector Space):** ChromaDB with `nomic-embed-text` embeddings and an enforced similarity floor ($\ge 0.65$) for conceptual matching.
* **Continuous Negative Learning:** Automatically identifies user feedback signals (`incorrect`, `wrong`, `hallucination`) and writes persistent negative guardrails to SQLite to prevent repeated extraction errors.
* **Layout-Preserving Document Conversion:** Non-blocking format conversions across PDF, DOCX, PPTX, Excel, and CSV that preserve vector coordinates, margins, and inline tabs.

---

## Architecture Overview

```text
[ React 18 + Vite (Frontend) ]
        │  ▲
        │  │ REST APIs / Binary Blob Streaming
        ▼  │
[ FastAPI Application Server ]
  ├── run_in_threadpool (Converters: pdf2docx / docx2pdf / Tesseract OCR)
  └── LangGraph State Machine
        ├── SQLite FTS5 (Lexical BM25 Gate)
        ├── ChromaDB (HNSW Cosine Vector Store)
        ├── Hallucination Guardrail (Regex Date & Claim Verification)
        └── Ollama Runtime (Llama-3-8B Q4_K_M on CPU)
# Local AI Engineering Workspace ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Node: 18+](https://img.shields.io/badge/Node.js-18%2B-green.svg)](https://nodejs.org/)
[![Architecture: 100% Offline](https://img.shields.io/badge/Privacy-100%25_Air--Gapped-red.svg)]()

A high-performance, fully air-gapped, privacy-first AI engineering workspace. Designed to execute local quantized LLMs on commodity CPU hardware with zero external cloud dependencies, zero telemetry, grounded retrieval verification, and a high-fidelity document layout transformation engine.

---

## 📑 Table of Contents
1. [Core Features](#core-features)
2. [System Architecture](#system-architecture)
3. [Prerequisites & System Requirements](#prerequisites--system-requirements)
4. [Step-by-Step Installation on a New Machine](#step-by-step-installation-on-a-new-machine)
   - [1. Clone Repository](#1-clone-repository)
   - [2. Local LLM Runtime Setup (Ollama)](#2-local-llm-runtime-setup-ollama)
   - [3. Backend Setup (FastAPI & LangGraph)](#3-backend-setup-fastapi--langgraph)
   - [4. Frontend Setup (React & Vite)](#4-frontend-setup-react--vite)
   - [5. System Dependencies (OCR & COM Engines)](#5-system-dependencies-ocr--com-engines)
5. [Two-Tier Hybrid Retrieval Pipeline](#two-tier-hybrid-retrieval-pipeline)
6. [Exact Document Conversion Pipeline](#exact-document-conversion-pipeline)
7. [Environment Configuration Reference](#environment-configuration-reference)
8. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 🚀 Core Features

- **100% Offline & Private:** Zero internet connection required after initial installation. No data packets ever leave the local network boundary.
- **Deterministic CPU Inference:** Powered by quantized **Llama-3-8B (Q4_K_M)** using greedy decoding (`temperature=0.0`) and bounded context (`num_ctx=2048`) to eliminate hallucinations.
- **Dual-Tier Retrieval Engine:**
  - **Tier 1 (Sub-millisecond Lexical Gate):** SQLite FTS5 indexes query keywords in under 2ms. If search keywords do not appear in the ingested text, it immediately bypasses vector false positives.
  - **Tier 2 (Dense Semantic Vector Store):** ChromaDB with `nomic-embed-text` embeddings and an enforced similarity floor ($\ge 0.65$) for conceptual extraction.
- **Continuous Negative Learning:** Real-time feedback loops capture corrections (`incorrect`, `wrong`, `hallucination`) and persist them directly to SQLite to prevent repeated extraction errors.
- **Non-Blocking Document Conversions:** Background threadpool workers offload heavy vector transformations (PDF, Word DOCX, PowerPoint PPTX, Excel XLSX, CSV) while maintaining sub-second UI responsiveness.

---

## 🏗️ System Architecture

```text
                     ┌──────────────────────────────────────┐
                     │       React 18 + Vite Frontend       │
                     │  (Live Telemetry, Markdown, Stream)  │
                     └──────────────────┬───────────────────┘
                                        │ REST APIs & Direct Binary Streams
                                        ▼
                     ┌──────────────────────────────────────┐
                     │        FastAPI ASGI Server           │
                     │      (Uvicorn on Port 8000)          │
                     └──────┬────────────────────────┬──────┘
                            │                        │
        run_in_threadpool   │                        │ LangGraph Workflow
     ┌──────────────────────┴────────┐      ┌────────┴─────────────────────────┐
     │ Exact Layout Document Engines │      │ Deterministic State Orchestrator │
     │  - pdf2docx (Vector Bounds)   │      │  - Intent Triage & Fast-Paths    │
     │  - docx2pdf (Native Word COM) │      │  - Negative Rule Injection       │
     │  - Tesseract OCR Extraction   │      │  - Hallucination Guardrail Node  │
     └───────────────────────────────┘      └────────┬─────────────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             │                                               │
                             ▼                                               ▼
              ┌─────────────────────────────┐                 ┌─────────────────────────────┐
              │    SQLite FTS5 + WAL DB     │                 │     ChromaDB Vector Store   │
              │  (Lexical Keyword Matching) │                 │   (HNSW Cosine Vector Space)│
              └─────────────────────────────┘                 └──────────────┬──────────────┘
                                                                             │
                                                                             ▼
                                                              ┌─────────────────────────────┐
                                                              │   Local Ollama Engine       │
                                                              │  (Llama-3-8B Q4_K_M on CPU) │
                                                              └─────────────────────────────┘