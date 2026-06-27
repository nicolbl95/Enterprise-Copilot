import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END


# ============================================================
# 0. CHARGEMENT DU .ENV + ALIGNEMENT DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print("ANTHROPIC_API_KEY exists in core.graph:", bool(os.getenv("ANTHROPIC_API_KEY")))


# Import après chargement du .env
from agents.orchestrator import run_agentic_rag


# ============================================================
# 1. ÉTAT PARTAGÉ DU COPILOT
# ============================================================

class CopilotState(TypedDict, total=False):
    """
    État partagé entre les nœuds du graphe.

    C'est le "plateau" qui circule dans LangGraph :
    question -> agentic_rag -> format_response -> END
    """

    question: str
    session_id: str
    chat_history: List[Dict[str, Any]]
    response_language: str

    context: str
    sources: List[str]
    steps: List[str]

    raw_answer: str
    final_answer: str
    evaluation: Optional[Dict[str, Any]]

    trace_id: Optional[str]


# ============================================================
# 2. HELPERS
# ============================================================

def normalize_response_language(language: str | None) -> str:
    """
    Normalise la langue de réponse.

    Valeurs acceptées :
    - fr
    - en
    """

    language = (language or "fr").lower().strip()

    if language not in ["fr", "en"]:
        return "fr"

    return language


# ============================================================
# 3. NŒUD PRINCIPAL : ORCHESTRATEUR AGENTIC RAG
# ============================================================

def agentic_rag_node(state: CopilotState) -> CopilotState:
    """
    Nœud principal qui appelle l'orchestrateur déjà validé.

    agents/orchestrator.py gère :
    - retrieval Qdrant
    - routage / grading
    - génération Claude
    - mémoire conversationnelle
    - évaluation LLM-as-judge
    - langue de réponse FR / EN
    """

    question = state.get("question", "")
    session_id = state.get("session_id", "default_session")
    chat_history = state.get("chat_history", [])
    response_language = normalize_response_language(state.get("response_language", "fr"))

    result = run_agentic_rag(
        question=question,
        session_id=session_id,
        chat_history=chat_history,
        response_language=response_language,
    )

    return {
        **state,
        "response_language": response_language,
        "context": result.get("context", ""),
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
        "raw_answer": result.get("answer", ""),
        "final_answer": result.get("answer", ""),
        "evaluation": result.get("evaluation"),
    }


# ============================================================
# 4. NŒUD DE FORMATAGE FINAL
# ============================================================

def format_response_node(state: CopilotState) -> CopilotState:
    """
    Nœud de sortie.

    Pour l'instant, il ne modifie pas la réponse.
    Il prépare surtout l'étape API REST FastAPI :
    on pourra y standardiser le JSON retourné à l'API.
    """

    final_answer = state.get("final_answer") or state.get("raw_answer") or ""

    return {
        **state,
        "final_answer": final_answer,
    }


# ============================================================
# 5. CONSTRUCTION DU GRAPHE
# ============================================================

def build_copilot_graph():
    """
    Construit le graphe principal du Copilot.

    Structure :
    agentic_rag -> format_response -> END
    """

    graph = StateGraph(CopilotState)

    graph.add_node("agentic_rag", agentic_rag_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("agentic_rag")

    graph.add_edge("agentic_rag", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# Instance globale utilisée par FastAPI ou Streamlit
copilot = build_copilot_graph()


# ============================================================
# 6. FONCTION D'APPEL SIMPLE POUR FASTAPI / TESTS
# ============================================================

def run_copilot(
    question: str,
    session_id: str = "default_session",
    chat_history: List[Dict[str, Any]] | None = None,
    trace_id: str | None = None,
    response_language: str = "fr",
) -> Dict[str, Any]:
    """
    Fonction simple à appeler depuis FastAPI.

    Elle renvoie une réponse structurée :
    - answer
    - sources
    - steps
    - context
    - evaluation
    - trace_id
    - response_language
    """

    clean_response_language = normalize_response_language(response_language)

    state: CopilotState = {
        "question": question,
        "session_id": session_id,
        "chat_history": chat_history or [],
        "trace_id": trace_id,
        "response_language": clean_response_language,
    }

    output = copilot.invoke(state)

    return {
        "answer": output.get("final_answer", ""),
        "sources": output.get("sources", []),
        "steps": output.get("steps", []),
        "context": output.get("context", ""),
        "evaluation": output.get("evaluation"),
        "trace_id": output.get("trace_id"),
        "response_language": output.get("response_language", clean_response_language),
    }