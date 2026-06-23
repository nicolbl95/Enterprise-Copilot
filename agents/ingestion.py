import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from qdrant_client import QdrantClient 

def setup_embedding_model(): 
    """
    Charge un modèle d'embedding gratuit depuis HuggingFace
    (un modèle qui transforme du texte en vecteurs mathématiques)
    """
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 

def ingest_documents(folder_path: str, collection_name: str = "enterprise_docs"): 
    # Connexion à Qdrant (qui tourne dans Docker) 
    client = QdrantClient(host="localhost", port=6333) 

    # Lecture de tous les PDFs dans le dossier 
    reader = SimpleDirectoryReader(input_dir=folder_path, required_exts=[".pdf"]) 
    documents = reader.load_data() 
    
    # Configuration du modèle d'embedding 
    Settings.embed_model = setup_embedding_model() 
    Settings.chunk_size = 512 # ← taille de chaque morceau de texte 
    Settings.chunk_overlap = 50 # ← chevauchement entre morceaux 
    
    # Stockage dans Qdrant 
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name) 
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store) 
    
    print(f"{len(documents)} documents ingérés dans Qdrant ✓") 
    return index 

if __name__ == "__main__": 
    # Avant de lancer, on s'assure que le dossier data existe
    if not os.path.exists("./data"):
        os.makedirs("./data")
    ingest_documents("./data")