import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ============================================================
# 0. CHARGEMENT DU .ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 1. MODÈLE D'EMBEDDING
# ============================================================

def setup_embedding_model():
    """
    Charge un modèle d'embedding gratuit depuis HuggingFace.
    BAAI/bge-small-en-v1.5 produit des vecteurs de taille 384.
    """
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ============================================================
# 2. CLIENT QDRANT LOCAL OU CLOUD
# ============================================================

def create_qdrant_client() -> QdrantClient:
    """
    Crée un client Qdrant compatible :
    - Qdrant Cloud si QDRANT_URL existe dans .env
    - Qdrant local sinon
    """

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        print("🌐 Qdrant ingestion mode: cloud")
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    print("💻 Qdrant ingestion mode: local")
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "127.0.0.1"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )


# ============================================================
# 3. CRÉATION / VÉRIFICATION COLLECTION
# ============================================================

def ensure_collection_exists(client: QdrantClient, collection_name: str):
    """
    Vérifie que la collection existe.
    Si elle n'existe pas, elle est créée avec la bonne dimension.
    """

    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            print(f"📦 Collection '{collection_name}' absente. Création en cours...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

            print(f"✅ Collection '{collection_name}' créée.")
        else:
            print(f"✅ Collection '{collection_name}' existe déjà.")

    except Exception as e:
        print(f"❌ Erreur lors de la vérification/création de la collection : {e}")
        raise


# ============================================================
# 4. INGESTION DOCUMENTS
# ============================================================

def ingest_documents(folder_path: str, collection_name: str | None = None):
    """
    Lit les PDFs dans folder_path, génère les embeddings,
    puis écrit les vecteurs dans Qdrant local ou cloud.
    """

    collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "enterprise_docs")

    print("🚀 Démarrage ingestion documents")
    print("📁 Dossier source:", folder_path)
    print("📦 Collection cible:", collection_name)

    client = create_qdrant_client()

    ensure_collection_exists(
        client=client,
        collection_name=collection_name,
    )

    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".pdf"],
    )

    documents = reader.load_data()

    if not documents:
        raise ValueError(f"Aucun document PDF trouvé dans le dossier : {folder_path}")

    print(f"📄 Documents chargés : {len(documents)}")

    Settings.embed_model = setup_embedding_model()
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    print(f"✅ {len(documents)} documents ingérés avec succès dans Qdrant.")
    print(f"📦 Collection utilisée : {collection_name}")

    return index


# ============================================================
# 5. EXÉCUTION DIRECTE
# ============================================================

if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    ingest_documents(str(data_dir))