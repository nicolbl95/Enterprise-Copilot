import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams

def setup_embedding_model(): 
    """
    Charge un modèle d'embedding gratuit depuis HuggingFace
    (un modèle qui transforme du texte en vecteurs mathématiques)
    """
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 

def ingest_documents(folder_path: str, collection_name: str = "enterprise_docs"): 
    # Connexion explicite à Qdrant (qui tourne dans Docker) 
    client = QdrantClient(host="127.0.0.1", port=6333) 

    # Piège Windows/Docker : On s'assure manuellement que la collection existe sur le serveur Docker
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            print(f"La collection {collection_name} n'existe pas dans Docker. Création en cours...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    except Exception as e:
        print(f"Erreur lors de la vérification de la collection Docker : {e}")

    # Lecture de tous les PDFs dans le dossier 
    reader = SimpleDirectoryReader(input_dir=folder_path, required_exts=[".pdf"]) 
    documents = reader.load_data() 
    
    # Configuration du modèle d'embedding 
    Settings.embed_model = setup_embedding_model() 
    Settings.chunk_size = 512 
    Settings.chunk_overlap = 50 
    
    # Injection forcée dans le stockage vectoriel Qdrant
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name) 
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store) 
    
    print(f"{len(documents)} documents ingérés dans Qdrant ✓") 
    return index 

if __name__ == "__main__": 
    if not os.path.exists("./data"):
        os.makedirs("./data")
    ingest_documents("./data")