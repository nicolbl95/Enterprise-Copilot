import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# 1. CHEMIN RACINE DU PROJET + CHARGEMENT DU .ENV
# ============================================================

# Si app.py est dans un dossier comme src/ ou frontend/,
# parents[1] pointe vers la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Charger explicitement le fichier .env depuis la racine
load_dotenv(PROJECT_ROOT / ".env")

# Ajouter la racine du projet au path Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# 2. VARIABLES LANGFUSE
# ============================================================

os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY") or ""
os.environ["LANGFUSE_BASE_URL"] = (
    os.getenv("LANGFUSE_BASE_URL")
    or os.getenv("LANGFUSE_HOST")
    or "https://cloud.langfuse.com"
)

# Vérification visible dans le terminal Streamlit
print("LANGFUSE_PUBLIC_KEY exists:", bool(os.environ["LANGFUSE_PUBLIC_KEY"]))
print("LANGFUSE_SECRET_KEY exists:", bool(os.environ["LANGFUSE_SECRET_KEY"]))
print("LANGFUSE_BASE_URL:", os.environ["LANGFUSE_BASE_URL"])
print("GROQ_API_KEY exists:", bool(os.getenv("GROQ_API_KEY")))


# ============================================================
# 3. INITIALISATION LANGFUSE AVANT LES IMPORTS LLAMAINDEX/AGENTS
# ============================================================

# ============================================================
# 3. INITIALISATION LANGFUSE AVANT LES IMPORTS LLAMAINDEX/AGENTS
# ============================================================

from langfuse import Langfuse
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_BASE_URL"],
)

try:
    if langfuse.auth_check():
        print("✅ Langfuse connecté")
    else:
        print("❌ Langfuse non connecté : vérifie les clés Langfuse et LANGFUSE_BASE_URL")
except Exception as e:
    print("❌ Erreur auth_check Langfuse:", e)

# Active le tracing LlamaIndex
LlamaIndexInstrumentor().instrument()


# ============================================================
# 4. CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Enterprise Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Copilot")
st.subheader("Posez vos questions sur les documents de l'entreprise")


# ============================================================
# 5. IMPORTS PROJET APRÈS LANGFUSE
# ============================================================

from agents.search import setup_search_engine, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine


# ============================================================
# 6. MÉMOIRE ET HISTORIQUE STREAMLIT
# ============================================================

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre Copilot. Que souhaitez-vous savoir aujourd'hui ?"
        }
    ]


# ============================================================
# 7. AFFICHAGE DE L'HISTORIQUE
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ============================================================
# 8. INPUT UTILISATEUR + TRAÇAGE LANGFUSE
# ============================================================

if user_query := st.chat_input("Votre question ici..."):

    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Le Copilot consulte les manuels et l'historique..."):
            try:
                # Span manuel pour garantir qu'une trace apparaît dans Langfuse
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="enterprise-copilot-question",
                    input={"question": user_query},
                ) as trace_span:

                    # Récupération de l'index vectoriel
                    index = setup_search_engine()
                    retriever = index.as_retriever(similarity_top_k=2)

                    # Configuration du Chat Engine
                    chat_engine = CondensePlusContextChatEngine.from_defaults(
                        retriever=retriever,
                        llm=Settings.llm,
                        memory=st.session_state.chat_memory,
                        context_prompt=(
                            "Vous êtes un assistant IA de confiance pour l'entreprise.\n"
                            "Répondez à la question de l'utilisateur en vous basant sur le contexte fourni ci-dessous.\n"
                            "Si la réponse n'est pas dans le contexte, dites-le simplement sans inventer d'informations.\n"
                            "Voici le contexte utile :\n"
                            "{context_str}\n"
                        )
                    )

                    # Appel LLM/RAG
                    response = chat_engine.chat(user_query)
                    answer_text = str(response)

                    # Ajouter la réponse dans Langfuse
                    trace_span.update(output={"answer": answer_text})

                # Important avec Streamlit : forcer l'envoi des traces
                langfuse.flush()

                # Afficher la réponse
                st.write(answer_text)

                # Sauvegarder dans l'historique UI
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text}
                )

            except Exception as e:
                error_msg = f"Mince, une erreur est survenue : {e}"
                st.error(error_msg)
                print(error_msg)