import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from anthropic import Anthropic
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

ORCHESTRATOR_VERSION = "memory-fix-chat-history-v3"

print("✅ ORCHESTRATOR LOADED:", __file__)
print("✅ ORCHESTRATOR VERSION:", ORCHESTRATOR_VERSION)

# ============================================================
# ALIGNEMENT DES CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print("ANTHROPIC_API_KEY exists in orchestrator:", bool(os.getenv("ANTHROPIC_API_KEY")))


# ============================================================
# 1. DÉFINITION DE L'ÉTAT DU GRAPHE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    question: str
    context: str
    sources: List[str]
    steps: List[str]
    answer: str
    evaluation: Dict[str, Any] | None


# ============================================================
# 2. HELPERS
# ============================================================

def extract_text_safely(message: Any) -> str:
    """Extrait proprement le texte d'une réponse Anthropic."""
    if not hasattr(message, "content") or not message.content:
        return ""

    first_block = message.content[0]

    if hasattr(first_block, "text"):
        return str(first_block.text)

    if isinstance(first_block, dict) and "text" in first_block:
        return str(first_block["text"])

    return str(first_block)


def format_message_for_history(message: BaseMessage) -> str:
    """Convertit un message LangChain en texte lisible pour Claude."""
    role = "Utilisateur"

    if isinstance(message, AIMessage):
        role = "Assistant"
    elif isinstance(message, HumanMessage):
        role = "Utilisateur"
    else:
        role = getattr(message, "type", "Message")

    return f"{role}: {message.content}"


def format_history(messages: List[BaseMessage], max_messages: int = 12) -> str:
    """Formate les derniers messages de la conversation."""
    if not messages:
        return "Aucun historique conversationnel disponible."

    recent_messages = messages[-max_messages:]
    return "\n".join(format_message_for_history(msg) for msg in recent_messages)


# ============================================================
# 3. NŒUDS LANGGRAPH
# ============================================================

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Étape 1 : recherche sémantique dans Qdrant, compatible local + Qdrant Cloud."""
    print("🔍 [Node: Retriever] Recherche dans Qdrant...")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

    if qdrant_url:
        print("🌐 Qdrant mode: cloud")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
    else:
        print("💻 Qdrant mode: local")
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(state["question"])

    context = "\n\n".join([n.text for n in nodes])
    sources = [n.metadata.get("file_name", "inconnu") for n in nodes]

    return {
        "context": context,
        "sources": list(set(sources)),
        "steps": state.get("steps", []) + ["Retrieved from Qdrant"]
    }

def grade_documents_route(state: AgentState) -> Literal["generate", "web_search"]:
    """
    Étape 2 : Claude évalue si le contexte local ou l'historique suffit.
    """
    print("🧠 [Router: Grader] Claude évalue la pertinence du contexte et de l'historique...")

    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    history_text = format_history(state.get("messages", []))

    prompt = f"""Analyse la question, l'historique conversationnel et le contexte documentaire.

Tu dois répondre UNIQUEMENT par 'OUI' ou 'NON'.

Réponds OUI si :
- le contexte documentaire contient l'information nécessaire ;
- OU l'historique conversationnel contient l'information nécessaire ;
- OU la question est une question de suivi qui dépend de la conversation précédente.

Réponds NON seulement si ni le contexte documentaire ni l'historique ne permettent de répondre.

Historique conversationnel :
{history_text}

Question actuelle :
{state['question']}

Contexte documentaire :
{state['context']}
"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    decision = extract_text_safely(message).strip().upper()

    if "OUI" in decision:
        print("✅ Contexte ou historique suffisant. Passage à la génération.")
        return "generate"

    print("⚠️ Contexte insuffisant. Déroutage vers la recherche web.")
    return "web_search"


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Étape 3 : recherche Web via DuckDuckGo."""
    print("🌐 [Node: Web Search] Recherche d'informations fraîches sur le Web...")

    try:
        query_encoded = urllib.parse.quote(state["question"])
        url = f"https://api.duckduckgo.com/?q={query_encoded}&format=json"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

        web_context = data.get("AbstractText", "")

        if not web_context and data.get("RelatedTopics"):
            first_topic = data["RelatedTopics"][0]
            if isinstance(first_topic, dict):
                web_context = first_topic.get("Text", "")

        if not web_context:
            web_context = "Pas de résultats publics directs trouvés."

    except Exception as e:
        web_context = f"Échec de la recherche web : {str(e)}"

    combined_context = f"{state['context']}\n\n[Données Web complémentaires] :\n{web_context}"

    return {
        "context": combined_context,
        "sources": state.get("sources", []) + ["Web (DuckDuckGo)"],
        "steps": state.get("steps", []) + ["Web Search Fallback"]
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Étape 4 : génération finale.
    Claude reçoit maintenant l'historique conversationnel + le contexte.
    """
    print("✍️ [Node: Generator] Claude synthétise la réponse finale avec mémoire...")

    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    history_text = format_history(state.get("messages", []))

    system_prompt = """Tu es un copilote d'entreprise expert.

Règles importantes :
1. Réponds de façon claire, structurée et professionnelle.
2. Utilise l'historique conversationnel pour répondre aux questions de suivi.
3. Si l'utilisateur t'a donné son prénom, son rôle ou une préférence dans la conversation, tu peux t'en souvenir dans cette session.
4. Utilise le contexte documentaire quand la question concerne les documents de l'entreprise.
5. Si la réponse n'est ni dans l'historique ni dans le contexte, dis-le simplement.
6. Ne prétends pas que l'information manque si elle apparaît dans l'historique conversationnel.
"""

    user_prompt = f"""Historique conversationnel :
{history_text}

Contexte documentaire / web :
{state['context']}

Question actuelle :
{state['question']}

Réponds maintenant à la question de l'utilisateur.
"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    final_answer = extract_text_safely(message)

    return {
        "answer": final_answer,
        # On ajoute la réponse assistant à la mémoire.
        "messages": [AIMessage(content=final_answer)]
    }


def evaluate_node(state: AgentState) -> Dict[str, Any]:
    """Étape 5 : évaluation LLM-as-a-judge."""
    print("⚖️ [Node: Evaluator] Claude évalue la qualité de sa réponse...")

    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    history_text = format_history(state.get("messages", []), max_messages=6)

    eval_prompt = f"""Évalue cette réponse selon 3 critères.
Réponds UNIQUEMENT en JSON brut, sans markdown, sans fioritures.

Question :
{state['question']}

Historique conversationnel :
{history_text}

Contexte :
{state['context'][:1000]}

Réponse :
{state['answer']}

Format attendu :
{{
  "faithfulness": 0.9,
  "relevance": 0.8,
  "completeness": 0.7
}}"""

    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": eval_prompt}]
        )

        raw_json = extract_text_safely(message).strip()
        scores = json.loads(raw_json)
        print(f"📊 Scores obtenus : {scores}")

        return {
            "evaluation": scores
        }

    except Exception as e:
        print(f"⚠️ Échec de l'auto-évaluation : {str(e)}")

        return {
            "evaluation": None
        }


# ============================================================
# 4. COMPOSITION DU GRAPH LANGGRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

workflow.set_entry_point("retrieve")

workflow.add_conditional_edges(
    "retrieve",
    grade_documents_route,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("evaluate", END)


# ============================================================
# 5. EXÉCUTION DU GRAPHE AVEC MÉMOIRE REDIS + HISTORIQUE STREAMLIT
# ============================================================

def run_agentic_rag(
    question: str,
    session_id: str = "default_session",
    chat_history: list | None = None
) -> dict:
    """Exécute le graphe Agentic RAG avec mémoire Redis + historique Streamlit."""

    print("✅ run_agentic_rag called with chat_history:", chat_history is not None)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print("🧠 Redis URL utilisée:", redis_url)

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()

        app_graph = workflow.compile(checkpointer=saver)

        langgraph_messages = []

        if chat_history:
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content", "")

                if not content:
                    continue

                if role == "user":
                    langgraph_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langgraph_messages.append(AIMessage(content=content))

        if not langgraph_messages or langgraph_messages[-1].content != question:
            langgraph_messages.append(HumanMessage(content=question))

        inputs = {
            "question": question,
            "messages": langgraph_messages,
            "context": "",
            "sources": [],
            "steps": [],
            "answer": "",
            "evaluation": None
        }

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        output = app_graph.invoke(inputs, config=config)  # type: ignore

        return {
            "answer": output.get("answer", ""),
            "sources": output.get("sources", []),
            "steps": output.get("steps", []),
            "context": output.get("context", ""),
            "evaluation": output.get("evaluation")
        }