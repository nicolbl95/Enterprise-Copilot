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
    page_title="Enterprise Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Copilot")
st.subheader("Posez vos questions sur les documents de l'entreprise")


# ============================================================
# 5. IMPORTS AGENT PROJET
# ============================================================

from agents.retriever import retrieve_and_answer


# ============================================================
# 6. MÉMOIRE ET HISTORIQUE STREAMLIT
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre Copilot. Que souhaitez-vous savoir aujourd'hui ?",
            "sources": []
        }
    ]


# ============================================================
# 7. AFFICHAGE DE L'HISTORIQUE
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"📁 **Sources consultées :** {', '.join(msg['sources'])}")


# ============================================================
# 8. INPUT UTILISATEUR + ÉXÉCUTION AGENT + TRACE LANGFUSE
# ============================================================

if user_query := st.chat_input("Votre question ici..."):

    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Le Copilot consulte les manuels et génère la réponse..."):

            try:
                # Context manager Langfuse SDK v3 pour tracer proprement l'appel Streamlit
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="enterprise-copilot-question",
                    input={"question": user_query},
                    metadata={
                        "app": "enterprise-copilot",
                        "source": "streamlit"
                    }
                ) as trace_span:

                    # Appel direct de l'Agent 2 validé
                    result = retrieve_and_answer(user_query)
                    
                    answer_text = result["answer"]
                    sources = result["sources"]

                    # Met à jour la fin de la trace avec la réponse finale
                    trace_span.update(
                        output={"answer": answer_text, "sources": sources}
                    )

                # Forcer l'envoi des traces en arrière-plan
                langfuse_client.flush()
                print("✅ Trace envoyée à Langfuse")

                # Affichage dans l'interface UI
                st.write(answer_text)
                if sources:
                    st.caption(f"📁 **Sources consultées :** {', '.join(sources)}")

                # Sauvegarde dans l'historique de session
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text, "sources": sources}
                )

            except Exception as e:
                error_msg = f"Mince, une erreur est survenue : {e}"
                st.error(error_msg)
                print(error_msg)