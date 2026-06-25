import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import TextBlock
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient


def retrieve_and_answer(question: str, collection_name: str = "enterprise_docs") -> dict:
    # 1. Connexion à Qdrant et chargement de l'index existant
    client = QdrantClient(host="localhost", port=6333)

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    # 2. Recherche des 5 passages les plus pertinents
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(question)

    context = "\n\n".join([n.text for n in nodes])
    sources = [n.metadata.get("file_name", "inconnu") for n in nodes]

    # 3. Vérification de la clé Anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY est manquante dans le fichier .env")

    # 4. Appel à l'API Anthropic
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "Tu es un assistant expert en documentation d'entreprise.\n"
            "Réponds uniquement en te basant sur le contexte fourni.\n"
            "Si l'information n'est pas dans le contexte, dis-le clairement.\n"
            "Cite toujours la source de ta réponse."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Contexte :\n{context}\n\nQuestion : {question}"
            }
        ]
    )

    # 5. Extraction propre du texte
    first_block = message.content[0]

    if isinstance(first_block, TextBlock):
        answer = first_block.text
    else:
        answer = getattr(first_block, "text", str(first_block))

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "context": context
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env")

    if os.getenv("ANTHROPIC_API_KEY"):
        print("🚀 Lancement du test de l'Agent 2 (Retriever) avec Claude Sonnet 4.6...")

        test_question = "Quels sont les outils d'embeddings à maîtriser ?"

        try:
            result = retrieve_and_answer(test_question)

            print("\n🤖 [Réponse de Claude] :")
            print(result["answer"])

            print("\n📁 [Sources consultées] :")
            print(result["sources"])

        except Exception as e:
            print(f"❌ Une erreur est survenue lors du test : {e}")

    else:
        print("⚠️ La variable ANTHROPIC_API_KEY est introuvable.")