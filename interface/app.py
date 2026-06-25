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
# 3. INITIALISATION LANGFUSE VERSION NATIVE (SDK v3)
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

# Initialisation de l'instrumentation pour le suivi global
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
# 5. IMPORTS AGENT PROJET (LangGraph Orchestrator)
# ============================================================

from agents.orchestrator import run_agentic_rag


# ============================================================
# 6. MÉMOIRE ET HISTORIQUE STREAMLIT
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre Copilot augmenté par LangGraph. Posez-moi une question sur les documents ou l'actualité !",
            "sources": [],
            "steps": []
        }
    ]


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
# 8. INPUT UTILISATEUR + ÉXÉCUTION GRAPHE + TRACE LANGFUSE
# ============================================================

if user_query := st.chat_input("Votre question ici..."):

    st.session_state.messages.append({"role": "user", "content": user_query, "sources": [], "steps": []})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("L'orchestrateur LangGraph évalue et exécute les nœuds..."):

            try:
                # Context manager Langfuse SDK v3
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="enterprise-copilot-agentic-rag",
                    input={"question": user_query},
                    metadata={
                        "app": "enterprise-copilot",
                        "architecture": "langgraph-crag"
                    }
                ) as trace_span:

                    # Appel de notre orchestrateur multi-agents
                    result = run_agentic_rag(user_query)
                    
                    answer_text = result["answer"]
                    sources = result["sources"]
                    steps = result["steps"]

                    # Mise à jour de la trace Langfuse
                    trace_span.update(
                        output={"answer": answer_text, "sources": sources, "steps": steps}
                    )

                # Forcer l'envoi des traces
                langfuse_client.flush()
                print("✅ Trace envoyée à Langfuse")

                # Affichage des résultats dans l'UI
                st.write(answer_text)
                st.info(f"👣 **Parcours des agents :** {' ➔ '.join(steps)}")
                if sources:
                    st.caption(f"📁 **Sources consultées :** {', '.join(sources)}")

                # Sauvegarde dans l'historique de session
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