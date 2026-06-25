import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

from anthropic import Anthropic
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.runnables import RunnableConfig  # ← Ajouté pour corriger l'erreur Pylance

# Alignement des chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ============================================================
# 1. DÉFINITION DE L'ÉTAT DU GRAPHE (State)
# ============================================================
class AgentState(TypedDict):
    question: str
    context: str
    sources: List[str]
    steps: List[str]
    answer: str

# ============================================================
# HELPER POUR SUPPRIMER LES ERREURS PYLANCE
# ============================================================
def extract_text_safely(message: Any) -> str:
    """Extrait le texte de la réponse d'Anthropic sans lever d'alerte Pylance."""
    if not hasattr(message, "content") or not message.content:
        return ""
    first_block = message.content[0]
    if hasattr(first_block, "text"):
        return str(first_block.text)
    if isinstance(first_block, dict) and "text" in first_block:
        return str(first_block["text"])
    return str(first_block)

# ============================================================
# 2. LES NŒUDS DE L'ORCHESTRATEUR (Nodes)
# ============================================================

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Étape 1 : Recherche sémantique dans Qdrant."""
    print("🔍 [Node: Retriever] Recherche dans Qdrant...")
    
    client = QdrantClient(host="localhost", port=6333)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    vector_store = QdrantVectorStore(client=client, collection_name="enterprise_docs")
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
    """Étape 2 (Conditionnelle) : Claude évalue la pertinence du contexte local."""
    print("🧠 [Router: Grader] Claude évalue la pertinence du contexte...")
    
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = f"""Analyse la question et le contexte fourni. Détermine si le contexte contient l'information nécessaire pour répondre précisément à la question.
Réponds UNIQUEMENT par le mot 'OUI' ou le mot 'NON'.

Question: {state['question']}
Contexte: {state['context']}"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    
    decision = extract_text_safely(message).strip().upper()
    
    if "OUI" in decision:
        print("✅ Contexte suffisant. Passage à la génération.")
        return "generate"
    else:
        print("⚠️ Contexte insuffisant ! Déroutage vers la Recherche Web.")
        return "web_search"


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Étape 3 : Recherche Web via DuckDuckGo (sans dépendance externe instable)."""
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
            web_context = data["RelatedTopics"][0].get("Text", "")
            
        if not web_context:
            web_context = "Pas de résultats publics directs trouvés."
            
    except Exception as e:
        web_context = f"Échec de la recherche web : {str(e)}"

    combined_context = f"{state['context']}\n\n[Données Web complémentaires] :\n{web_context}"
    
    return {
        "context": combined_context,
        "sources": state["sources"] + ["Web (DuckDuckGo)"],
        "steps": state["steps"] + ["Web Search Fallback"]
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """Étape 4 : Génération de la réponse finale structurée."""
    print("✍️ [Node: Generator] Claude synthétise la réponse finale...")
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="Tu es un copilote d'entreprise expert. Réponds de façon claire et structurée à l'aide du contexte fourni.",
        messages=[{"role": "user", "content": f"Contexte:\n{state['context']}\n\nQuestion: {state['question']}"}]
    )
    
    final_answer = extract_text_safely(message)
    return {"answer": final_answer}

def evaluate_node(state: AgentState) -> Dict[str, Any]:
    """Étape 5 : Le LLM-as-a-judge évalue la qualité de la réponse générée."""
    print("⚖️ [Node: Evaluator] Claude évalue la qualité de sa réponse...")
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    eval_prompt = f"""Évalue cette réponse selon 3 critères. Réponds UNIQUEMENT en JSON brut, sans markdown, sans fioritures.
Question : {state['question']}
Contexte : {state['context'][:1000]}
Réponse : {state['answer']}

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
        
    except Exception as e:
        print(f"⚠️ Échec de l'auto-évaluation : {str(e)}")
        
    return {}

# ============================================================
# 3. COMPOSITION DU GRAPH (LangGraph Workflow)
# ============================================================
workflow = StateGraph(AgentState)

# Enregistrement des nœuds
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

# Point d'entrée
workflow.set_entry_point("retrieve")

# Logique conditionnelle de routage
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_route,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

# Liens de terminaison
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("evaluate", END)


def run_agentic_rag(question: str, session_id: str = "default_session") -> dict:
    """Fonction d'appel pour exécuter le graphe de RAG avec mémoire Redis."""
    inputs: AgentState = {
        "question": question, 
        "context": "", 
        "sources": [], 
        "steps": [], 
        "answer": ""
    }
    
    # 🔑 Typage explicite avec RunnableConfig pour corriger l'erreur Pylance
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    
    # 💾 Connexion au Checkpointer Redis ouverte dynamiquement à chaque appel de l'application
    with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
        app_graph = workflow.compile(checkpointer=saver)
        output = app_graph.invoke(inputs, config=config)
    
    # ✨ Le return obligatoire bien aligné qui résout l'erreur Pylance
    return {
        "answer": output["answer"],
        "sources": output["sources"],
        "steps": output["steps"]
    }