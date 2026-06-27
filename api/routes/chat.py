import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. CHARGEMENT DU .ENV + ALIGNEMENT DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT DU GRAPHE PRINCIPAL
# ============================================================

from core.graph import run_copilot


# ============================================================
# 3. CRÉATION DE L'APP FASTAPI
# ============================================================

app = FastAPI(
    title="Enterprise Knowledge Copilot API",
    description="API REST pour un Copilot RAG multi-agents avec LangGraph, Qdrant, Anthropic et Langfuse.",
    version="1.0.0"
)


# ============================================================
# 4. SCHÉMAS PYDANTIC
# ============================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Rôle du message : user ou assistant")
    content: str = Field(..., description="Contenu du message")


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question posée par l'utilisateur")
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session. Si absent, un UUID est généré."
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Historique de conversation envoyé par Gradio"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    steps: List[str]
    session_id: str
    evaluation: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    folder_path: str = Field(..., description="Chemin du dossier contenant les documents à ingérer")


# ============================================================
# 5. ROUTE CHAT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal du Copilot.

    Il reçoit une question, l'historique de conversation,
    appelle le graphe LangGraph, puis retourne la réponse,
    les sources, les étapes et l'évaluation.
    """

    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Conversion Pydantic -> dict simple pour LangGraph / LangChain
        chat_history = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in request.chat_history
            if msg.content
        ]

        result = run_copilot(
            question=request.question,
            session_id=session_id,
            chat_history=chat_history,
            trace_id=None
        )

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            steps=result.get("steps", []),
            session_id=session_id,
            evaluation=result.get("evaluation")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 6. ROUTE INGESTION
# ============================================================

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Endpoint pour lancer l'ingestion de documents.

    Cette route suppose que tu as une fonction ingest_documents
    dans agents/ingestion.py.
    """

    try:
        from agents.ingestion import ingest_documents

        ingest_documents(request.folder_path)

        return {
            "status": "success",
            "message": "Documents ingérés avec succès",
            "folder_path": request.folder_path
        }

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="La fonction agents.ingestion.ingest_documents n'existe pas encore."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 7. ROUTE HEALTHCHECK
# ============================================================

@app.get("/health")
async def health():
    """
    Route simple pour vérifier que l'API tourne.
    """

    return {
        "status": "ok",
        "service": "Enterprise Knowledge Copilot",
        "version": "1.0.0"
    }