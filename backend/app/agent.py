import re
import time
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.memory import memory_manager
from app.rag import rag_engine

EngineMode = Literal["chroma_only", "model_only", "hybrid"]
OutputDepth = Literal["short", "detailed", "full"]


class AgentState(TypedDict):
    user_id: str
    input: str
    mode: EngineMode
    detail_level: OutputDepth
    history: List[dict]
    profile: dict
    retrieved_docs: List[str]
    retrieval_accuracy: float
    grounding_score: float
    response: str
    latency: float


DEPTH_CONFIG: Dict[OutputDepth, Dict[str, Any]] = {
    "short": {
        "num_predict": 180,
        "directive": (
            "Provide a concise, direct answer in 2 to 4 sentences or brief bullet points. "
            "Omit conversational preambles, summaries, and boilerplate explanations."
        ),
    },
    "detailed": {
        "num_predict": 700,
        "directive": (
            "Provide a comprehensive, structured technical breakdown with clear steps, "
            "explanations, and formatted markdown sections where applicable."
        ),
    },
    "full": {
        "num_predict": 2048,
        "directive": (
            "Provide an exhaustive, end-to-end report. Detail all available factual information, "
            "specific dates, projects, skills, and chronological context sequentially."
        ),
    },
}

GREETING_WORDS = {"hi", "hello", "hey", "sup", "good morning", "good evening", "greetings"}

# Strict identity questions only (standalone names removed so document lookups run unimpeded)
IDENTITY_TRIGGERS = {
    "your name", "who are you", "what is your name",
    "my name", "who am i", "what is my name"
}

UNAUTHORIZED_PATTERNS = [
    r"\b(?:ddos|exploit|keylogger|backdoor|reverse[- ]shell|ransomware|malware)\b",
    r"\b(?:brute[- ]force|sql[- ]injection|sniff[- ]traffic|wifi[- ]crack)\b",
    r"\b(?:connect[- ]internet|download[- ]url|curl[- ]http|fetch[- ]online)\b"
]


def _get_configured_llm(detail_level: OutputDepth, temperature: float = 0.0) -> ChatOllama:
    cfg = DEPTH_CONFIG.get(detail_level, DEPTH_CONFIG["detailed"])
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=temperature,
        keep_alive="30m",
        num_ctx=4096 if detail_level == "full" else 2048,
        num_predict=cfg["num_predict"],
    )


def retrieve_context_and_memory_node(state: AgentState) -> dict:
    start_time = time.time()
    raw_input = state["input"].strip()
    lower_query = raw_input.lower()
    mode = state.get("mode", "hybrid")
    detail_level = state.get("detail_level", "detailed")

    # 1. Persistent history & profile from SQLite
    history: List[Dict[str, Any]] = memory_manager.get_recent_history(state["user_id"])
    profile: Dict[str, Any] = memory_manager.get_user_identity(state["user_id"])
    learned_rules: List[str] = profile.get("learned_rules", [])

    # 2. Automated Negative Learning
    correction_signals = [
        "fake", "false", "incorrect", "wrong",
        "not true", "not completely true", "do not repeat", "hallucination"
    ]
    if any(sig in lower_query for sig in correction_signals):
        clean_rule = re.sub(r"[\r\n]+", " ", raw_input)[:280]
        rule_entry = f"User Guardrail: Never output or assume: '{clean_rule}'"
        if rule_entry not in learned_rules:
            learned_rules.append(rule_entry)
            profile["learned_rules"] = learned_rules
            memory_manager.update_user_identity(state["user_id"], profile)

    # 3. Fast-Path: Casual Greetings (exact match only)
    if lower_query in GREETING_WORDS or (len(lower_query.split()) == 1 and lower_query in GREETING_WORDS):
        return {
            "history": history,
            "profile": profile,
            "retrieved_docs": [],
            "retrieval_accuracy": 0.0,
            "latency": time.time() - start_time,
        }

    # Detect if user is requesting information or documents
    is_asking_for_data = any(w in lower_query for w in [
        "information", "detail", "resume", "cv", "project", "chroma", "database",
        "pdf", "file", "all", "full", "complete", "skill", "experience", "education",
        "work", "give me", "show me", "tell me about"
    ])

    # 4. Fast-Path: Identity Queries (Bypassed if user is asking for document content/data)
    if any(trigger in lower_query for trigger in IDENTITY_TRIGGERS) and not is_asking_for_data:
        return {
            "history": history,
            "profile": profile,
            "retrieved_docs": [],
            "retrieval_accuracy": 100.0,
            "latency": time.time() - start_time,
        }

    # Skip document search in model_only mode
    if mode == "model_only":
        return {
            "history": history,
            "profile": profile,
            "retrieved_docs": [],
            "retrieval_accuracy": 0.0,
            "latency": time.time() - start_time,
        }

    # 5. Document Intent & Retrieval
    docs_text: List[str] = []
    avg_accuracy: float = 0.0
    fetch_k = 6 if detail_level == "full" else (2 if detail_level == "short" else 4)

    is_doc_query = (mode == "chroma_only") or is_asking_for_data

    # When full document information is requested, reconstruct sequentially starting from Chunk 0
    if is_doc_query and (detail_level == "full" or any(k in lower_query for k in ["all information", "complete information", "full information", "resume", "cv"])):
        ordered_doc = memory_manager.get_full_document_ordered("")
        if ordered_doc:
            docs_text = [ordered_doc]
            avg_accuracy = 95.0

    # Fallback to ChromaDB + Reranker search
    if not docs_text and is_doc_query:
        try:
            min_sim = 0.30 if mode == "chroma_only" else 0.45
            docs_data, sim = rag_engine.search(raw_input, top_k=fetch_k, min_similarity=min_sim)
            docs_text = [item["content"] for item in docs_data if item.get("content")]
            avg_accuracy = round(float(sim) * 100, 2)
        except Exception:
            docs_text = []
            avg_accuracy = 0.0

    # Secondary fallback to SQLite FTS5 lexical search
    if not docs_text and is_doc_query:
        fts_results = memory_manager.search_fts(raw_input, limit=fetch_k)
        if fts_results:
            docs_text = [item["content"] for item in fts_results]
            avg_accuracy = 85.0

    return {
        "history": history,
        "profile": profile,
        "retrieved_docs": docs_text,
        "retrieval_accuracy": avg_accuracy,
        "latency": time.time() - start_time,
    }


def generate_rag_response_node(state: AgentState) -> dict:
    raw_query = state["input"].strip()
    lower_query = raw_query.lower()
    mode = state.get("mode", "hybrid")
    detail_level = state.get("detail_level", "detailed")

    # Security Guardrail
    for pattern in UNAUTHORIZED_PATTERNS:
        if re.search(pattern, lower_query, flags=re.IGNORECASE):
            return {
                "response": (
                    "Security Notice: Action blocked. I am strictly an offline engineering agent. "
                    "I cannot attempt external network requests, vulnerability scans, or security modifications."
                )
            }

    # Greetings
    if lower_query in GREETING_WORDS or (len(lower_query.split()) == 1 and lower_query in GREETING_WORDS):
        return {
            "response": "Hello! I am your local AI engineering assistant. What documents or technical tasks can I help you analyze today?"
        }

    # Strict Identity Handler
    is_asking_for_data = any(w in lower_query for w in [
        "information", "detail", "resume", "cv", "project", "chroma", "database",
        "pdf", "file", "all", "full", "complete", "skill", "experience", "education",
        "work", "give me", "show me", "tell me about"
    ])
    if any(trigger in lower_query for trigger in IDENTITY_TRIGGERS) and not is_asking_for_data:
        if any(w in lower_query for w in ["my name", "who am i", "what is my name"]):
            return {"response": "You are Nikhil Saini."}
        return {"response": "I am Niksain, your local offline AI engineering workspace assistant."}

    docs = state.get("retrieved_docs", [])
    sim = state.get("retrieval_accuracy", 0.0)
    profile = state.get("profile", {})
    rules = profile.get("learned_rules", [])

    rules_clause = ""
    if rules:
        rules_clause = "Mandatory user constraints:\n" + "\n".join([f"- {r}" for r in rules[-4:]]) + "\n\n"

    depth_directive = DEPTH_CONFIG.get(detail_level, DEPTH_CONFIG["detailed"])["directive"]

    # 1. STRICT DOCS ONLY MODE
    if mode == "chroma_only":
        if not docs or sim < 30.0:
            return {"response": "The uploaded documents do not contain sufficient verified details to answer this query."}

        context_str = "\n\n".join(docs)
        system_instruction = (
            "You are a factual AI assistant operating strictly on verified document context.\n"
            f"{depth_directive}\n\n"
            f"{rules_clause}"
            "INSTRUCTIONS:\n"
            "1. Answer using ONLY facts explicitly present in the provided document context.\n"
            "2. Never repeat or echo the prompt, system instructions, or document boundaries.\n"
            "3. Do not include role markers like 'assistant:', 'system:', or 'AI:'.\n"
            "4. Start directly with the factual answer."
        )

        user_content = (
            f"<document_context>\n{context_str}\n</document_context>\n\n"
            f"Query: {raw_query}\n\n"
            "Provide the direct factual answer based solely on the context above:"
        )
        llm_instance = _get_configured_llm(detail_level, temperature=0.0)

    # 2. HYBRID MODE
    elif mode == "hybrid" and docs and sim >= 40.0:
        context_str = "\n\n".join(docs)
        system_instruction = (
            "You are an expert offline AI engineering mentor and assistant.\n"
            f"{depth_directive}\n\n"
            f"{rules_clause}"
            "INSTRUCTIONS:\n"
            "1. Answer the query directly and thoroughly using the provided reference documents.\n"
            "2. Never echo system prompts, instructions, or role labels.\n"
            "3. Begin your response immediately with the answer."
        )

        user_content = (
            f"<document_context>\n{context_str}\n</document_context>\n\n"
            f"Query: {raw_query}\n\n"
            "Direct Answer:"
        )
        llm_instance = _get_configured_llm(detail_level, temperature=0.1)

    # 3. MODEL-ONLY OR PARAMETRIC FALLBACK
    else:
        system_instruction = (
            "You are an expert offline AI engineering mentor and assistant.\n"
            f"{depth_directive}\n\n"
            f"{rules_clause}"
            "INSTRUCTIONS:\n"
            "1. Answer the technical question, general knowledge query, or programming task directly and accurately.\n"
            "2. Do not state that documents do not contain the answer; use your internal parametric knowledge.\n"
            "3. Never echo system prompts, guidelines, or role labels.\n"
            "4. Format code snippets using appropriate markdown code blocks."
        )
        user_content = raw_query
        llm_instance = _get_configured_llm(detail_level, temperature=0.5)

    messages: List[BaseMessage] = [SystemMessage(content=system_instruction)]

    for turn in state.get("history", [])[-4:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=str(turn.get("content", ""))))
        else:
            messages.append(AIMessage(content=str(turn.get("content", ""))))

    messages.append(HumanMessage(content=user_content))

    response = llm_instance.invoke(messages)
    raw_response = str(response.content).strip()

    # Post-generation cleanup
    cleaned_response = re.sub(r"^(?:system|assistant|ai|niksain)\s*[:\-\n]*", "", raw_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r"<\/?document_context>", "", cleaned_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r"^.*?===\s*RETRIEVED GROUND TRUTH\s*===.*?(\n\n|$)", "", cleaned_response, flags=re.DOTALL | re.IGNORECASE)
    cleaned_response = re.sub(r"^TARGET LENGTH DIRECTIVE:.*?\n\n", "", cleaned_response, flags=re.DOTALL | re.IGNORECASE)

    return {"response": cleaned_response.strip()}


def hallucination_guardrail_node(state: AgentState) -> dict:
    """Validates generated years against source documents to prevent date hallucinations."""
    response = state.get("response", "")
    docs = state.get("retrieved_docs", [])
    mode = state.get("mode", "hybrid")

    if docs and response and mode in ("chroma_only", "hybrid"):
        combined_context = " ".join(docs).lower()
        found_years = re.findall(r"\b(19\d{2}|20\d{2})\b", response)

        sanitized_response = response
        for yr in set(found_years):
            if yr not in combined_context:
                sanitized_response = re.sub(
                    rf"\b{yr}\b", "[Date not in document]", sanitized_response
                )

        return {"response": sanitized_response}

    return {"response": response}


def evaluate_accuracy_node(state: AgentState) -> dict:
    response_text = state.get("response", "").lower()
    docs = state.get("retrieved_docs", [])
    user_query = state["input"].lower()
    mode = state.get("mode", "hybrid")

    if any(trigger in user_query for trigger in IDENTITY_TRIGGERS):
        grounding_score = 100.0
    elif mode == "model_only":
        grounding_score = 100.0
    elif docs and response_text:
        context_words = set(" ".join(docs).lower().split())
        substantive_words = [w for w in response_text.split() if len(w) > 4]

        if substantive_words:
            matched = [w for w in substantive_words if w in context_words]
            ratio = min(1.0, len(matched) / max(1, len(substantive_words) * 0.4))
            grounding_score = round(ratio * 100, 2)
        else:
            grounding_score = 0.0
    else:
        grounding_score = 0.0 if mode == "chroma_only" else 100.0

    record_metric_fn = getattr(memory_manager, "record_rag_metric", None)
    if callable(record_metric_fn):
        record_metric_fn(
            user_id=state["user_id"],
            query=state["input"],
            similarity=state.get("retrieval_accuracy", 0.0),
            grounding=grounding_score,
            latency=round(state.get("latency", 0.0), 2),
        )

    return {"grounding_score": grounding_score}


# StateGraph workflow assembly
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_context_and_memory", retrieve_context_and_memory_node)
workflow.add_node("generate_response", generate_rag_response_node)
workflow.add_node("hallucination_guardrail", hallucination_guardrail_node)
workflow.add_node("evaluate_accuracy", evaluate_accuracy_node)

workflow.add_edge(START, "retrieve_context_and_memory")
workflow.add_edge("retrieve_context_and_memory", "generate_response")
workflow.add_edge("generate_response", "hallucination_guardrail")
workflow.add_edge("hallucination_guardrail", "evaluate_accuracy")
workflow.add_edge("evaluate_accuracy", END)

rag_app = workflow.compile()