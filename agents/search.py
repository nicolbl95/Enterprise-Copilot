import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient


# ---------------------------------------------------------
# 1. Chargement du fichier .env depuis la racine du projet
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

groq_key = os.getenv("GROQ_API_KEY")

print("ENV chargé depuis:", ENV_PATH)
print("GROQ_API_KEY exists:", bool(groq_key))
print("GROQ_API_KEY length:", len(groq_key) if groq_key else 0)
print("GROQ_API_KEY starts with:", groq_key[:6] if groq_key else None)

if not groq_key:
    raise ValueError("GROQ_API_KEY est manquante ou vide dans le fichier .env")


# ---------------------------------------------------------
# 2. Configuration explicite des modèles
# ---------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = Groq(
    model=GROQ_MODEL,
    api_key=groq_key,
)

Settings.embed_model = embed_model
Settings.llm = llm

print("GROQ_MODEL utilisé:", GROQ_MODEL)
print("LLM configuré:", Settings.llm)


# ---------------------------------------------------------
# 3. Connexion à Qdrant
# ---------------------------------------------------------

def setup_search_engine(collection_name: str = "enterprise_docs"):
    """
    Se connecte à la base de données Qdrant et reconstruit l'index.
    """
    client = QdrantClient(host="127.0.0.1", port=6333)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index


# ---------------------------------------------------------
# 4. Question au moteur RAG
# ---------------------------------------------------------

def ask_enterprise_copilot(query: str):
    """
    Pose une question au moteur RAG avec Groq comme LLM.
    """
    print(f"🤖 Le Copilot réfléchit à ta question : '{query}'...\n")

    index = setup_search_engine()

    retriever = index.as_retriever(
        similarity_top_k=2
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    response = query_engine.query(query)

    print("✨ Réponse rédigée du Copilot :")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    ask_enterprise_copilot(
        "Quels sont les outils d'embeddings et de bases vectorielles à maîtriser ?"
    )