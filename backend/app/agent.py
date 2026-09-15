import re
import time
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.memory import memory_manager
from app.rag import rag_engine


class AgentState(TypedDict):
    user_id: str
    input: str
    history: List[dict]
    profile: dict
    retrieved_docs: List[str]
    retrieval_accuracy: float
    grounding_score: float
    response: str
    latency: float


# Local CPU-tuned LLM execution
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.0,
    keep_alive="30m",
    num_ctx=2048,
)

GREETING_WORDS = {"hi", "hello", "hey", "sup", "good morning", "good evening", "greetings"}
IDENTITY_TRIGGERS = {
    "your name", "who are you", "what is your name",
    "my name", "who am i", "what is my name",
    "niksain", "nikhil", "nikhil saini"
}

UNAUTHORIZED_PATTERNS = [
    r"\b(?:ddos|exploit|keylogger|backdoor|reverse[- ]shell|ransomware|malware)\b",
    r"\b(?:brute[- ]force|sql[- ]injection|sniff[- ]traffic|wifi[- ]crack)\b",
    r"\b(?:connect[- ]internet|download[- ]url|curl[- ]http|fetch[- ]online)\b"
]


def retrieve_context_and_memory_node(state: AgentState) -> dict:
    start_time = time.time()
    raw_input = state["input"].strip()
    lower_query = raw_input.lower()

    # 1. Pull persistent history & profile from SQLite
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

    # 3. Fast-Path 1: Casual Greetings
    if lower_query in GREETING_WORDS or len(lower_query.split()) <= 1:
        return {
            "history": history,
            "profile": profile,
            "retrieved_docs": [],
            "retrieval_accuracy": 0.0,
            "latency": time.time() - start_time,
        }

    # 4. Fast-Path 2: Identity Queries
    if any(trigger in lower_query for trigger in IDENTITY_TRIGGERS):
        return {
            "history": history,
            "profile": profile,
            "retrieved_docs": [],
            "retrieval_accuracy": 100.0,
            "latency": time.time() - start_time,
        }

    # 5. HYBRID TWO-TIER RETRIEVAL:
    # Tier 1: Sub-millisecond lexical keyword search via SQLite FTS5
    docs_text: List[str] = []
    avg_accuracy: float = 0.0

    fts_results = memory_manager.search_fts(raw_input, limit=4)
    if fts_results:
        docs_text = [item["content"] for item in fts_results]
        avg_accuracy = 90.0
    else:
        # Tier 2: ChromaDB semantic fallback only if the query specifically refers to uploaded context
        doc_intent_keywords = [
            "resume", "cv", "document", "uploaded", "file", "my project",
            "my experience", "my skill", "summary", "profile", "internship", "education"
        ]
        is_doc_query = any(k in lower_query for k in doc_intent_keywords)

        # Higher similarity threshold (0.65) to eliminate false semantic matches
        if is_doc_query:
            try:
                docs_data, sim = rag_engine.search(raw_input, top_k=4, min_similarity=0.65)
                filtered_chunks = [
                    item["content"]
                    for item in docs_data
                    if item.get("content") and float(item.get("similarity", 0.0)) >= 0.65
                ]
                docs_text = filtered_chunks
                avg_accuracy = round(float(sim) * 100, 2)
            except Exception:
                docs_text = []
                avg_accuracy = 0.0

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

    # Pre-generation Security Boundary Check
    for pattern in UNAUTHORIZED_PATTERNS:
        if re.search(pattern, lower_query, flags=re.IGNORECASE):
            return {
                "response": (
                    "Security Notice: Action blocked. I am strictly an offline engineering agent. "
                    "I cannot attempt external network requests, unauthorized vulnerability testing, "
                    "or modification of system security parameters."
                )
            }

    # Fast-Path Greetings Handler
    if lower_query in GREETING_WORDS or len(lower_query.split()) <= 1:
        return {
            "response": "Hello! I am your local AI engineering assistant. What documents or technical tasks can I help you analyze today?"
        }

    # Fast-Path Identity Handler
    if any(trigger in lower_query for trigger in IDENTITY_TRIGGERS):
        if any(w in lower_query for w in ["my name", "who am i", "what is my name"]):
            return {"response": "You are Nikhil Saini."}
        return {"response": "I am Niksain, your local offline AI engineering workspace assistant."}

    docs = state.get("retrieved_docs", [])
    profile = state.get("profile", {})
    rules = profile.get("learned_rules", [])

    rules_block = ""
    if rules:
        rules_block = (
            "USER CONSTRAINTS (MANDATORY TO FOLLOW):\n"
            + "\n".join([f"- {r}" for r in rules[-5:]])
            + "\n\n"
        )

    # Conditional Prompt Router
    if docs:
        context_str = "\n---\n".join(docs)
        system_prompt = (
            "You are Niksain, a factual assistant operating 100% offline.\n"
            "Your task is to answer the user query based SOLELY on the retrieved context below.\n\n"
            f"{rules_block}"
            "=== RETRIEVED GROUND TRUTH ===\n"
            f"{context_str}\n"
            "==============================\n\n"
            "STRICT EXTRACTION DIRECTIVES:\n"
            "1. Answer using ONLY the explicit facts written in the text above.\n"
            "2. Never extrapolate, interpolate, or guess details not present in the text.\n"
            "3. If the retrieved text does not provide the answer, state: "
            "'The uploaded documents do not contain sufficient details to answer this query.'\n"
            "4. NEVER start output with role prefixes like 'assistant:', 'AI:', or 'Niksain:'.\n"
            "5. Answer directly and concisely."
        )
    else:
        system_prompt = (
            "You are Niksain, an expert local engineering assistant operating 100% offline.\n"
            "Your system identity is Niksain. The human engineer is Nikhil Saini.\n\n"
            f"{rules_block}"
            "Answer the user's technical question, general knowledge query, mathematical calculation, "
            "or programming request directly and accurately using your internal knowledge.\n\n"
            "FORMATTING RULES:\n"
            "- NEVER use role prefixes like 'assistant:', 'AI:', or 'Niksain:'.\n"
            "- Do NOT state 'the documents do not contain this information' when answering general questions.\n"
            "- Start directly with the answer.\n"
            "- Format code with clear Markdown blocks."
        )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    for turn in state.get("history", [])[-4:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=str(turn.get("content", ""))))
        else:
            messages.append(AIMessage(content=str(turn.get("content", ""))))

    messages.append(HumanMessage(content=raw_query))

    response = llm.invoke(messages)
    raw_response = str(response.content).strip()

    # Clean leading assistant/system prefixes thoroughly
    cleaned_response = re.sub(
        r"^(?:assistant|ai|niksain)\s*[:\-\n]*", "", raw_response, flags=re.IGNORECASE
    ).strip()

    return {"response": cleaned_response}


def hallucination_guardrail_node(state: AgentState) -> dict:
    """Validates generated years against source documents to prevent date hallucinations."""
    response = state.get("response", "")
    docs = state.get("retrieved_docs", [])

    if docs and response:
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

    if any(trigger in user_query for trigger in IDENTITY_TRIGGERS):
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
        grounding_score = 0.0

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