import os
from dotenv import load_dotenv

# Blocage de sécurité pour couper court aux instanciations par défaut
os.environ["OPENAI_API_KEY"] = "not-needed"

from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient

# 1. Chargement de ta clé Groq présente dans ton fichier .env
load_dotenv()

# 2. CONFIGURATION GLOBALE STRICTE
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

def setup_search_engine(collection_name: str = "enterprise_docs"):
    """
    Se connecte à la base de données Qdrant et reconstruit l'index.
    """
    client = QdrantClient(host="127.0.0.1", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

def ask_enterprise_copilot(query: str):
    """
    Pose une question au moteur RAG en liant manuellement tous les blocs à Groq.
    """
    print(f"🤖 Le Copilot réfléchit à ta question : '{query}'...\n")
    
    index = setup_search_engine()
    
    # Extraction manuelle des passages pertinents depuis Qdrant
    retriever = index.as_retriever(similarity_top_k=2)
    
    # FORCE L'USAGE DE GROQ : On crée le synthétiseur en lui transmettant explicitement notre LLM
    response_synthesizer = get_response_synthesizer(llm=Settings.llm)
    
    # Assemblage final du moteur de requêtes sans passer par les automatismes par défaut
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    
    # Exécution de la génération de la réponse
    response = query_engine.query(query)
    
    print("✨ Réponse rédigée du Copilot :")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    ask_enterprise_copilot("Quels sont les outils d'embeddings et de bases vectorielles à maîtriser ?")