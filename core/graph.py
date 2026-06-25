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

    context: str
    sources: List[str]
    steps: List[str]

    raw_answer: str
    final_answer: str
    evaluation: Optional[Dict[str, Any]]

    trace_id: Optional[str]


# ============================================================
# 2. NŒUD PRINCIPAL : ORCHESTRATEUR AGENTIC RAG
# ============================================================

def agentic_rag_node(state: CopilotState) -> CopilotState:
    """
    Nœud principal qui appelle l'orchestrateur déjà validé.

    agents/orchestrator.py gère déjà :
    - retrieval Qdrant
    - routage / grading
    - génération Claude
    - mémoire conversationnelle
    - évaluation RAGAS-style interne
    """

    question = state.get("question", "")
    session_id = state.get("session_id", "default_session")
    chat_history = state.get("chat_history", [])

    result = run_agentic_rag(
        question=question,
        session_id=session_id,
        chat_history=chat_history,
    )

    return {
        **state,
        "context": result.get("context", ""),
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
        "raw_answer": result.get("answer", ""),
        "final_answer": result.get("answer", ""),
        "evaluation": result.get("evaluation"),
    }


# ============================================================
# 3. NŒUD DE FORMATAGE FINAL
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
# 4. CONSTRUCTION DU GRAPHE
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
# 5. FONCTION D'APPEL SIMPLE POUR FASTAPI / TESTS
# ============================================================

def run_copilot(
    question: str,
    session_id: str = "default_session",
    chat_history: List[Dict[str, Any]] | None = None,
    trace_id: str | None = None,
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
    """

    state: CopilotState = {
        "question": question,
        "session_id": session_id,
        "chat_history": chat_history or [],
        "trace_id": trace_id,
    }

    output = copilot.invoke(state)

    return {
        "answer": output.get("final_answer", ""),
        "sources": output.get("sources", []),
        "steps": output.get("steps", []),
        "context": output.get("context", ""),
        "evaluation": output.get("evaluation"),
        "trace_id": output.get("trace_id"),
    }