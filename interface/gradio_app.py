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


# ============================================================
# 1.BIS TEXTES FR / EN
# ============================================================

TEXTS = {
    "fr": {
        "title": "# 🤖 NovaTech Copilot",
        "intro": f"""
        <p>
        Ce Copilot IA simule un assistant de connaissance d’entreprise pour
        <strong>NovaTech Industries</strong> ({PDF_LINK}), une entreprise fictive.
        Il interroge une base documentaire interne avec recherche sémantique,
        orchestration multi-agents, réponses sourcées et scoring qualité.
        Posez votre propre question ou testez l’une des questions d’exemple ci-dessous.
        </p>
        """,
        "conversation_label": "Conversation",
        "textbox_label": "Votre question",
        "textbox_placeholder": "Posez votre question...",
        "send_button": "Envoyer",
        "clear_button": "Effacer toute la conversation",
        "examples_title": "### 💡 Questions d’exemple",
        "sources_label": "📁 **Sources :** ",
        "steps_label": "👣 **Parcours des agents :** ",
        "quality_label": "📊 **Qualité** — ",
        "faithfulness": "Fidélité",
        "relevance": "Pertinence",
        "completeness": "Complétude",
        "connection_error": (
            "❌ Impossible de contacter l'API FastAPI.\n\n"
            "Vérifie que l'API tourne bien sur : http://localhost:8000\n\n"
            "Commande à lancer dans un autre terminal :\n"
            "`python -m uvicorn api.routes.chat:app --reload --port 8000`"
        ),
        "timeout_error": (
            "⏱️ L'API a mis trop longtemps à répondre. "
            "Réessaie ou vérifie les logs FastAPI."
        ),
        "http_error": "❌ Erreur HTTP pendant l'appel API",
        "generic_error": "❌ Erreur pendant l'appel API",
        "sidebar": """
### 🧠 Architecture

**Ce que démontre cette démo :**

* **RAG d’entreprise** : recherche sémantique dans une base documentaire interne.
* **Qdrant Cloud** : stockage vectoriel cloud, prêt pour un usage production.
* **Anthropic API** : génération des réponses avec Claude via l’API Anthropic.
* **LangGraph** : orchestration multi-agents avec étapes de retrieval, routage et évaluation.
* **FastAPI** : backend REST séparé de l’interface.
* **Gradio** : interface web déployable en un clic sur HuggingFace Spaces.
* **Langfuse** : observabilité des appels LLM et suivi des traces.
* **LLM-as-judge** : scoring automatique des réponses selon fidélité, pertinence et complétude.

---

### Agents

**Agent 1** : Retrieval Qdrant  
**Agent 2** : Routage / grading LangGraph  
**Agent 3** : Mémoire conversationnelle  
**Agent 4** : Évaluation qualité LLM-as-judge  
""",
        "examples": [
            "Quel est le délai de première réponse pour le support Business Critical ?",
            "Quelles certifications NovaTech a-t-elle obtenues, et lesquelles sont en cours ?",
            "Quelle remise obtient-on avec un engagement de 36 mois ?",
            "Quel est le budget formation annuel par collaborateur et combien de jours sont garantis ?",
            "Qu'est-ce que le RPO et le RTO en plan Premium, et que se passe-t-il si le SLA n'est pas atteint ?",
        ],
    },
    "en": {
        "title": "# 🤖 NovaTech Copilot",
        "intro": f"""
        <p>
        This AI Copilot simulates an enterprise knowledge assistant for
        <strong>NovaTech Industries</strong> ({PDF_LINK}), a fictional company.
        It queries an internal knowledge base using semantic search,
        multi-agent orchestration, cited answers, and automated quality scoring.
        Ask your own question or try one of the example questions below.
        </p>
        """,
        "conversation_label": "Conversation",
        "textbox_label": "Your question",
        "textbox_placeholder": "Ask your question...",
        "send_button": "Send",
        "clear_button": "Clear entire conversation",
        "examples_title": "### 💡 Example questions",
        "sources_label": "📁 **Sources:** ",
        "steps_label": "👣 **Agent path:** ",
        "quality_label": "📊 **Quality** — ",
        "faithfulness": "Faithfulness",
        "relevance": "Relevance",
        "completeness": "Completeness",
        "connection_error": (
            "❌ Unable to contact the FastAPI backend.\n\n"
            "Make sure the API is running at: http://localhost:8000\n\n"
            "Command to run in another terminal:\n"
            "`python -m uvicorn api.routes.chat:app --reload --port 8000`"
        ),
        "timeout_error": (
            "⏱️ The API took too long to respond. "
            "Try again or check the FastAPI logs."
        ),
        "http_error": "❌ HTTP error during API call",
        "generic_error": "❌ Error during API call",
        "sidebar": """
### 🧠 Architecture

**What this demo demonstrates:**

- **Enterprise RAG**: semantic search over internal company documentation.
- **Qdrant Cloud**: cloud vector storage suitable for production-style retrieval.
- **LangGraph**: multi-agent orchestration with retrieval, routing, and evaluation steps.
- **FastAPI**: REST backend separated from the user interface.
- **Gradio**: web interface deployable on HuggingFace Spaces.
- **Langfuse**: observability for LLM calls and traces.
- **LLM-as-judge**: automated response scoring for faithfulness, relevance, and completeness.

---

### Agents

**Agent 1**: Qdrant retrieval  
**Agent 2**: LangGraph routing / grading  
**Agent 3**: Conversational memory  
**Agent 4**: LLM-as-judge quality evaluation  
""",
        "examples": [
            "What is the first response time for Business Critical support?",
            "Which certifications has NovaTech obtained, and which ones are in progress?",
            "What discount is granted for a 36-month commitment?",
            "What is the annual training budget per employee and how many training days are guaranteed?",
            "What are the RPO and RTO in the Premium plan, and what happens if the SLA is not met?",
        ],
    },
}


# ============================================================
# 1.TER STYLE CSS
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
   Boutons langue FR / EN
   ============================================================ */

.language-row {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 10px !important;
}

.language-button {
    min-width: 48px !important;
    max-width: 60px !important;
    height: 42px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* ============================================================
   Chatbot : éviter texte coupé + scroll horizontal
   ============================================================ */

.custom-chatbot {
    width: 100% !important;
    overflow-x: hidden !important;
}

.custom-chatbot,
.custom-chatbot > div,
.custom-chatbot [data-testid],
.custom-chatbot .wrap {
    max-width: 100% !important;
    overflow-x: hidden !important;
}

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

.custom-chatbot p,
.custom-chatbot li,
.custom-chatbot code,
.custom-chatbot pre {
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    white-space: normal !important;
}

.custom-chatbot table {
    display: block;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

.custom-chatbot .message,
.custom-chatbot .prose,
.custom-chatbot .markdown-body {
    padding-left: 18px !important;
    padding-right: 18px !important;
}

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

.custom-chatbot .icon-button {
    display: none !important;
}

.custom-chatbot .block-label,
.custom-chatbot label {
    visibility: visible !important;
}
"""


# ============================================================
# 2. FONCTIONS API + NORMALISATION HISTORIQUE
# ============================================================

def normalize_message_content(content: Any) -> str:
    """
    Convertit le contenu Gradio en texte simple.

    Gradio peut parfois stocker le contenu sous forme :
    - str
    - list[dict] avec {"text": "...", "type": "text"}
    - dict
    - autre objet
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")

    return str(content)


def normalize_chat_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Nettoie l'historique Gradio avant envoi à FastAPI.

    FastAPI doit recevoir uniquement :
    {"role": "...", "content": "..."}
    """

    clean_history: List[Dict[str, str]] = []

    if not history:
        return clean_history

    for msg in history:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = normalize_message_content(msg.get("content"))

        if role in ["user", "assistant"] and content.strip():
            clean_history.append(
                {
                    "role": role,
                    "content": content
                }
            )

    return clean_history


def chat_with_copilot(
    message: str,
    history: List[Dict[str, Any]],
    session_id: Optional[str],
    language: str,
):
    """
    Envoie la question à l'API FastAPI /chat,
    puis formate la réponse avec sources, parcours et évaluation.
    """

    lang = language if language in TEXTS else "fr"
    ui = TEXTS[lang]

    if not session_id:
        session_id = str(uuid.uuid4())

    if history is None:
        history = []

    clean_history = normalize_chat_history(history)

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "question": message,
                "session_id": session_id,
                "chat_history": clean_history,
                "response_language": lang,
            },
            timeout=180,
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.ConnectionError:
        return (
            ui["connection_error"],
            session_id,
        )

    except requests.exceptions.Timeout:
        return (
            ui["timeout_error"],
            session_id,
        )

    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        return (
            f"{ui['http_error']} : {str(e)}\n\n"
            f"Détail : {error_detail}",
            session_id,
        )

    except Exception as e:
        return (
            f"{ui['generic_error']} : {str(e)}",
            session_id,
        )

    answer = result.get("answer", "")

    sources = result.get("sources", [])
    if sources:
        answer += "\n\n---\n" + ui["sources_label"] + ", ".join(sources)

    steps = result.get("steps", [])
    if steps:
        answer += "\n\n" + ui["steps_label"] + " ➔ ".join(steps)

    evaluation = result.get("evaluation")
    if evaluation:
        faithfulness = evaluation.get("faithfulness", "N/A")
        relevance = evaluation.get("relevance", "N/A")
        completeness = evaluation.get("completeness", "N/A")

        answer += (
            "\n\n"
            + ui["quality_label"]
            + f"{ui['faithfulness']}: {faithfulness} | "
            + f"{ui['relevance']}: {relevance} | "
            + f"{ui['completeness']}: {completeness}"
        )

    return answer, session_id


# ============================================================
# 3. CALLBACKS GRADIO
# ============================================================

def respond(message, history, session_id, language):
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
        language=language,
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, new_session_id


def clear_conversation():
    """
    Réinitialise l'historique Gradio et la session.
    """

    return [], None


def select_language(language: str):
    """
    Change la langue de l'interface.
    On vide aussi la conversation pour éviter un mélange FR / EN.
    """

    lang = language if language in TEXTS else "fr"
    ui = TEXTS[lang]

    if lang == "fr":
        fr_variant = "primary"
        en_variant = "secondary"
    else:
        fr_variant = "secondary"
        en_variant = "primary"

    return (
        lang,
        ui["title"],
        ui["intro"],
        gr.update(label=ui["conversation_label"]),
        gr.update(label=ui["textbox_label"], placeholder=ui["textbox_placeholder"], value=""),
        gr.update(value=ui["send_button"]),
        gr.update(value=ui["clear_button"]),
        ui["examples_title"],
        *[gr.update(value=q) for q in ui["examples"]],
        ui["sidebar"],
        gr.update(variant=fr_variant),
        gr.update(variant=en_variant),
        [],
        None,
    )


# ============================================================
# 4. INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="NovaTech Copilot",
    css=CUSTOM_CSS
) as demo:

    language_state = gr.State(value="fr")
    session_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=5):
            title_md = gr.Markdown(TEXTS["fr"]["title"])

        with gr.Column(scale=1, elem_classes=["language-row"]):
            with gr.Row():
                fr_btn = gr.Button(
                    "FR",
                    variant="primary",
                    elem_classes=["language-button"],
                )
                en_btn = gr.Button(
                    "EN",
                    variant="secondary",
                    elem_classes=["language-button"],
                )

    intro_html = gr.HTML(TEXTS["fr"]["intro"])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=250,
                label=TEXTS["fr"]["conversation_label"],
                elem_classes=["custom-chatbot"]
            )

            msg = gr.Textbox(
                placeholder=TEXTS["fr"]["textbox_placeholder"],
                label=TEXTS["fr"]["textbox_label"]
            )

            gr.Markdown("")

            with gr.Row():
                submit_btn = gr.Button(TEXTS["fr"]["send_button"], variant="primary")
                clear_btn = gr.Button(TEXTS["fr"]["clear_button"])

            examples_title_md = gr.Markdown(TEXTS["fr"]["examples_title"])

            example_buttons = []
            for question in TEXTS["fr"]["examples"]:
                btn = gr.Button(
                    question,
                    variant="secondary"
                )
                example_buttons.append(btn)

        with gr.Column(scale=1):
            sidebar_md = gr.Markdown(TEXTS["fr"]["sidebar"])

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, session_state, language_state],
        outputs=[msg, chatbot, session_state],
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot, session_state, language_state],
        outputs=[msg, chatbot, session_state],
    )

    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, session_state],
    )

    for index, btn in enumerate(example_buttons):
        btn.click(
            fn=lambda lang, i=index: TEXTS[lang if lang in TEXTS else "fr"]["examples"][i],
            inputs=[language_state],
            outputs=msg,
        )

    language_outputs = [
        language_state,
        title_md,
        intro_html,
        chatbot,
        msg,
        submit_btn,
        clear_btn,
        examples_title_md,
        *example_buttons,
        sidebar_md,
        fr_btn,
        en_btn,
        chatbot,
        session_state,
    ]

    fr_btn.click(
        fn=lambda: select_language("fr"),
        inputs=[],
        outputs=language_outputs,
    )

    en_btn.click(
        fn=lambda: select_language("en"),
        inputs=[],
        outputs=language_outputs,
    )


# ============================================================
# 5. LANCEMENT LOCAL
# ============================================================

if __name__ == "__main__":
    demo.launch(server_port=7860)