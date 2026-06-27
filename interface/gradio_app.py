# pyright: reportAttributeAccessIssue=false

import uuid
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any

import gradio as gr
import requests


# ============================================================
# 1. CONFIGURATION API
# ============================================================

API_URL = "http://localhost:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "interface" / "assets" / "NovaTech.pdf"


def build_pdf_download_link() -> str:
    """
    Crée un lien HTML de téléchargement intégré pour le PDF.
    Cette méthode évite les problèmes de chemins Windows avec /file=...
    """

    if not PDF_PATH.exists():
        return "<span style='color:#ff6b6b'>(PDF indisponible)</span>"

    pdf_bytes = PDF_PATH.read_bytes()
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    return (
        f'<a href="data:application/pdf;base64,{encoded_pdf}" '
        f'download="NovaTech.pdf">PDF</a>'
    )


PDF_LINK = build_pdf_download_link()


EXAMPLE_QUESTIONS = [
    "Quel est le délai de première réponse pour le support Business Critical ?",
    "Quelles certifications NovaTech a-t-elle obtenues, et lesquelles sont en cours ?",
    "Quelle remise obtient-on avec un engagement de 36 mois ?",
    "Quel est le budget formation annuel par collaborateur et combien de jours sont garantis ?",
    "Qu'est-ce que le RPO et le RTO en plan Premium, et que se passe-t-il si le SLA n'est pas atteint ?",
]


# ============================================================
# 1.BIS STYLE CSS
# ============================================================

CUSTOM_CSS = """
/* ============================================================
   Layout général
   ============================================================ */

* {
    box-sizing: border-box;
}

body,
.gradio-container {
    overflow-x: hidden !important;
}

/* ============================================================
   Chatbot : éviter texte coupé + scroll horizontal
   ============================================================ */

.custom-chatbot {
    width: 100% !important;
    overflow-x: hidden !important;
}

/* Le viewport interne ne doit pas créer de scroll horizontal */
.custom-chatbot,
.custom-chatbot > div,
.custom-chatbot [data-testid],
.custom-chatbot .wrap {
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* Messages et contenu markdown */
.custom-chatbot .message,
.custom-chatbot .message-row,
.custom-chatbot .message-wrap,
.custom-chatbot .message-container,
.custom-chatbot .prose,
.custom-chatbot .markdown-body {
    max-width: 100% !important;
    width: 100% !important;
    overflow-x: hidden !important;
}

/* Empêcher le texte long de déborder */
.custom-chatbot p,
.custom-chatbot li,
.custom-chatbot code,
.custom-chatbot pre {
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    white-space: normal !important;
}

/* Tables markdown : scroll local si nécessaire, pas scroll global */
.custom-chatbot table {
    display: block;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Padding pour éviter que la première lettre soit coupée */
.custom-chatbot .message,
.custom-chatbot .prose,
.custom-chatbot .markdown-body {
    padding-left: 18px !important;
    padding-right: 18px !important;
}

/* Réduire l'effet de sous-rectangle autour des réponses */
.custom-chatbot .message {
    border: none !important;
    box-shadow: none !important;
}

/* ============================================================
   Masquer les icônes d'action du chatbot
   Share / Delete / Copy
   ============================================================ */

.custom-chatbot .message-buttons,
.custom-chatbot .message-actions,
.custom-chatbot .copy_code_button,
.custom-chatbot button[aria-label*="Share"],
.custom-chatbot button[aria-label*="share"],
.custom-chatbot button[aria-label*="Delete"],
.custom-chatbot button[aria-label*="delete"],
.custom-chatbot button[aria-label*="Copy"],
.custom-chatbot button[aria-label*="copy"],
.custom-chatbot button[title*="Share"],
.custom-chatbot button[title*="share"],
.custom-chatbot button[title*="Delete"],
.custom-chatbot button[title*="delete"],
.custom-chatbot button[title*="Copy"],
.custom-chatbot button[title*="copy"] {
    display: none !important;
}

/* Certains boutons Gradio apparaissent dans le header du Chatbot */
.custom-chatbot .icon-button {
    display: none !important;
}

/* Garder le label Conversation visible */
.custom-chatbot .block-label,
.custom-chatbot label {
    visibility: visible !important;
}
"""


# ============================================================
# 2. FONCTION D'APPEL API
# ============================================================

def chat_with_copilot(
    message: str,
    history: List[Dict[str, Any]],
    session_id: Optional[str],
):
    """
    Envoie la question à l'API FastAPI /chat,
    puis formate la réponse avec sources, parcours et évaluation.
    """

    if not session_id:
        session_id = str(uuid.uuid4())

    if history is None:
        history = []

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "question": message,
                "session_id": session_id,
                "chat_history": history,
            },
            timeout=180,
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.ConnectionError:
        return (
            "❌ Impossible de contacter l'API FastAPI.\n\n"
            "Vérifie que l'API tourne bien sur : http://localhost:8000\n\n"
            "Commande à lancer dans un autre terminal :\n"
            "`python -m uvicorn api.routes.chat:app --reload --port 8000`",
            session_id,
        )

    except requests.exceptions.Timeout:
        return (
            "⏱️ L'API a mis trop longtemps à répondre. "
            "Réessaie ou vérifie les logs FastAPI.",
            session_id,
        )

    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        return (
            f"❌ Erreur HTTP pendant l'appel API : {str(e)}\n\n"
            f"Détail : {error_detail}",
            session_id,
        )

    except Exception as e:
        return (
            f"❌ Erreur pendant l'appel API : {str(e)}",
            session_id,
        )

    answer = result.get("answer", "")

    sources = result.get("sources", [])
    if sources:
        answer += "\n\n---\n📁 **Sources :** " + ", ".join(sources)

    steps = result.get("steps", [])
    if steps:
        answer += "\n\n👣 **Parcours des agents :** " + " ➔ ".join(steps)

    evaluation = result.get("evaluation")
    if evaluation:
        faithfulness = evaluation.get("faithfulness", "N/A")
        relevance = evaluation.get("relevance", "N/A")
        completeness = evaluation.get("completeness", "N/A")

        answer += (
            "\n\n📊 **Qualité** — "
            f"Fidélité: {faithfulness} | "
            f"Pertinence: {relevance} | "
            f"Complétude: {completeness}"
        )

    return answer, session_id


# ============================================================
# 3. CALLBACKS GRADIO
# ============================================================

def respond(message, history, session_id):
    """
    Callback appelé quand l'utilisateur envoie un message.
    Compatible avec le format messages de Gradio récent.
    """

    if not message or not message.strip():
        return "", history, session_id

    if history is None:
        history = []

    answer, new_session_id = chat_with_copilot(
        message=message,
        history=history,
        session_id=session_id,
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, new_session_id


def clear_conversation():
    """
    Réinitialise l'historique Gradio et la session.
    """

    return [], None


# ============================================================
# 4. INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="NovaTech Copilot",
    css=CUSTOM_CSS
) as demo:

    gr.Markdown("# 🤖 NovaTech Copilot")

    gr.HTML(
        f"""
        <p>
        Ce Copilot IA simule un assistant de connaissance d’entreprise pour
        <strong>NovaTech Industries</strong> ({PDF_LINK}), une entreprise fictive.
        Il interroge une base documentaire interne avec recherche sémantique,
        orchestration multi-agents, réponses sourcées et scoring qualité.
        Posez votre propre question ou testez l’une des questions d’exemple ci-dessous.
        </p>
        """
    )

    session_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=250,
                label="Conversation",
                elem_classes=["custom-chatbot"]
            )

            msg = gr.Textbox(
                placeholder="Posez votre question...",
                label="Votre question"
            )

            gr.Markdown("")

            with gr.Row():
                submit_btn = gr.Button("Envoyer", variant="primary")
                clear_btn = gr.Button("Effacer toute la conversation")

            gr.Markdown("### 💡 Questions d’exemple")

            example_buttons = []
            for question in EXAMPLE_QUESTIONS:
                btn = gr.Button(
                    question,
                    variant="secondary"
                )
                example_buttons.append(btn)

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Architecture")
            gr.Markdown(
                """
**Ce que démontre cette démo :**

- **RAG d’entreprise** : recherche sémantique dans une base documentaire interne.
- **Qdrant Cloud** : stockage vectoriel cloud, prêt pour un usage production.
- **LangGraph** : orchestration multi-agents avec étapes de retrieval, routage et évaluation.
- **FastAPI** : backend REST séparé de l’interface.
- **Gradio** : interface web déployable en un clic sur HuggingFace Spaces.
- **Langfuse** : observabilité des appels LLM et suivi des traces.
- **LLM-as-judge** : scoring automatique des réponses selon fidélité, pertinence et complétude.

---

### Agents

**Agent 1** : Retrieval Qdrant  
**Agent 2** : Routage / grading LangGraph  
**Agent 3** : Mémoire conversationnelle  
**Agent 4** : Évaluation qualité LLM-as-judge  
"""
            )

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, session_state],
        outputs=[msg, chatbot, session_state],
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot, session_state],
        outputs=[msg, chatbot, session_state],
    )

    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, session_state],
    )

    for btn, question in zip(example_buttons, EXAMPLE_QUESTIONS):
        btn.click(
            fn=lambda q=question: q,
            inputs=[],
            outputs=msg,
        )


# ============================================================
# 5. LANCEMENT LOCAL
# ============================================================

if __name__ == "__main__":
    demo.launch(server_port=7860)