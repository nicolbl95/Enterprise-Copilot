import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient

# 1. Chargement de la clé API présente dans ton fichier .env
load_dotenv()

def setup_search_engine(collection_name: str = "enterprise_docs"):
    """
    Configure proprement l'embedding et le LLM Groq de manière globale et locale.
    """
    # Connexion au conteneur Docker Qdrant
    client = QdrantClient(host="127.0.0.1", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    
    # Configuration stricte des Settings globaux
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    
    # Reconstruction de l'index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

def ask_enterprise_copilot(query: str):
    """
    Pose une question en forçant chaque composant à utiliser Groq au lieu d'OpenAI.
    """
    print(f"🤖 Le Copilot réfléchit à ta question : '{query}'...\n")
    
    index = setup_search_engine()
    
    # On configure explicitement le retriever (recherche Qdrant)
    retriever = index.as_retriever(similarity_top_k=2)
    
    # PIÈGE OPENAI ÉVITÉ : On force le synthétiseur de réponse à utiliser Groq
    response_synthesizer = get_response_synthesizer(llm=Settings.llm)
    
    # On assemble le moteur de requêtes final à la main
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    
    # Exécution du RAG
    response = query_engine.query(query)
    
    print("✨ Réponse rédigée du Copilot :")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    ask_enterprise_copilot("Quels sont les outils d'embeddings et de bases vectorielles à maîtriser ?")