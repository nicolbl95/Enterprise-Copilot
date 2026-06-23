import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Récupérer les configurations de Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

def get_qdrant_client() -> QdrantClient:
    """
    Initialise et retourne un client pour communiquer avec la base de données vectorielle Qdrant.
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return client
    except Exception as e:
        print(f"Erreur lors de la connexion à Qdrant: {e}")
        raise e

# Instance partagée du client
qdrant_client = get_qdrant_client()