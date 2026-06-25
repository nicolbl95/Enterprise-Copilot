import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# 1. CHEMIN RACINE DU PROJET + CHARGEMENT DU .ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

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

print("LANGFUSE_PUBLIC_KEY exists:", bool(os.environ["LANGFUSE_PUBLIC_KEY"]))
print("LANGFUSE_SECRET_KEY exists:", bool(os.environ["LANGFUSE_SECRET_KEY"]))
print("LANGFUSE_BASE_URL:", os.environ["LANGFUSE_BASE_URL"])


# ============================================================
# 3. INITIALISATION LANGFUSE
# ============================================================

from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

langfuse_client = get_client()

try:
    if langfuse_client.auth_check():
        print("✅ Langfuse connecté")
    else:
        print("❌ Langfuse non connecté : vérifie les clés Langfuse")
except Exception as e:
    print("❌ Erreur auth_check Langfuse:", e)

# Initialisation unique de l'instrumentation LlamaIndex
if "instrumented" not in st.session_state:
    LlamaIndexInstrumentor().instrument()
    st.session_state.instrumented = True


# ============================================================
# 4. CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Enterprise Copilot — Agentic",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Copilot — Agentic RAG")
st.subheader("Orchestration LangGraph avec routage autonome")


# ============================================================
# 5. IMPORTS AGENT PROJET
# ============================================================

from agents.orchestrator import run_agentic_rag
import inspect
import agents.orchestrator as orch

print("✅ ORCHESTRATOR FILE USED:", orch.__file__)
print("✅ RUN_AGENTIC_RAG SIGNATURE:", inspect.signature(run_agentic_rag))

if "chat_history" not in str(inspect.signature(run_agentic_rag)):
    raise RuntimeError(
        f"Mauvaise version de run_agentic_rag chargée : {inspect.signature(run_agentic_rag)} depuis {orch.__file__}"
    )
# ============================================================
# 6. MÉMOIRE ET HISTORIQUE STREAMLIT
# ============================================================

# Session stable pour Redis pendant le développement.
# Tu peux remettre uuid.uuid4() plus tard si tu veux une session neuve par utilisateur.
if "session_id" not in st.session_state:
    st.session_state.session_id = "nicolas-dev-session-v3"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre Copilot augmenté par LangGraph. Posez-moi une question sur les documents ou l'actualité !",
            "sources": [],
            "steps": []
        }
    ]


# Bouton pratique pour nettoyer l'historique local Streamlit.
if st.button("🧹 Vider la conversation"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre Copilot augmenté par LangGraph. Posez-moi une question sur les documents ou l'actualité !",
            "sources": [],
            "steps": []
        }
    ]
    st.rerun()


# ============================================================
# 7. AFFICHAGE DE L'HISTORIQUE
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("steps"):
            st.info(f"👣 **Parcours des agents :** {' ➔ '.join(msg['steps'])}")

        if msg.get("sources"):
            st.caption(f"📁 **Sources consultées :** {', '.join(msg['sources'])}")


# ============================================================
# 8. INPUT UTILISATEUR + EXÉCUTION GRAPHE + TRACE LANGFUSE
# ============================================================

if user_query := st.chat_input("Votre question ici..."):

    # Sauvegarde immédiate de la question utilisateur dans l'historique Streamlit.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": [],
            "steps": []
        }
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("L'orchestrateur LangGraph évalue et exécute les nœuds..."):

            try:
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="enterprise-copilot-agentic-rag",
                    input={
                        "question": user_query,
                        "session_id": st.session_state.session_id
                    },
                    metadata={
                        "app": "enterprise-copilot",
                        "architecture": "langgraph-crag",
                        "session_id": st.session_state.session_id
                    }
                ) as trace_span:

                    # IMPORTANT :
                    # On passe maintenant l'historique Streamlit à LangGraph.
                    result = run_agentic_rag(
                        user_query,
                        session_id=st.session_state.session_id,
                        chat_history=st.session_state.messages
                    )

                    answer_text = result["answer"]
                    sources = result["sources"]
                    steps = result["steps"]

                    trace_span.update(
                        output={
                            "answer": answer_text,
                            "sources": sources,
                            "steps": steps
                        }
                    )

                langfuse_client.flush()
                print("✅ Trace envoyée à Langfuse")

                st.write(answer_text)

                if steps:
                    st.info(f"👣 **Parcours des agents :** {' ➔ '.join(steps)}")

                if sources:
                    st.caption(f"📁 **Sources consultées :** {', '.join(sources)}")

                # Sauvegarde de la réponse assistant dans l'historique Streamlit.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "steps": steps
                    }
                )

            except Exception as e:
                error_msg = f"Mince, une erreur est survenue : {e}"
                st.error(error_msg)
                print(error_msg)