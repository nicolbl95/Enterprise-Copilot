import os
import json
from typing import Dict, Any

from anthropic import Anthropic
from langfuse import get_client


anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
langfuse = get_client()


def extract_text_safely(message: Any) -> str:
    """Extrait proprement le texte d'une réponse Anthropic."""
    if not hasattr(message, "content") or not message.content:
        return ""

    first_block = message.content[0]

    if hasattr(first_block, "text"):
        return str(first_block.text)

    if isinstance(first_block, dict) and "text" in first_block:
        return str(first_block["text"])

    return str(first_block)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur en float entre 0 et 1."""
    try:
        number = float(value)
        return max(0.0, min(1.0, number))
    except Exception:
        return default


def evaluate_response(
    question: str,
    answer: str,
    context: str,
    trace_id: str | None = None,
) -> Dict[str, Any]:
    """
    Évalue une réponse RAG selon 3 critères :
    - faithfulness : fidélité au contexte
    - relevance : pertinence par rapport à la question
    - completeness : complétude de la réponse

    Si trace_id est fourni, les scores sont envoyés à Langfuse.
    """

    eval_prompt = f"""Tu es un évaluateur qualité pour un système RAG.

Évalue cette réponse selon 3 critères, chacun entre 0 et 1.

Définitions :
- faithfulness : la réponse est-elle fidèle au contexte fourni, sans hallucination ?
- relevance : la réponse répond-elle directement à la question ?
- completeness : la réponse est-elle suffisamment complète et utile ?

Réponds UNIQUEMENT en JSON brut, sans markdown, sans commentaire avant ou après.

Question :
{question}

Contexte :
{context[:1500]}

Réponse :
{answer}

JSON attendu :
{{
  "faithfulness": 0.9,
  "relevance": 0.8,
  "completeness": 0.7,
  "feedback": "commentaire court"
}}"""

    try:
        result = anthropic.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": eval_prompt}],
        )

        raw_text = extract_text_safely(result).strip()

        # Sécurité simple si Claude ajoute accidentellement du texte autour du JSON
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError(f"Aucun JSON trouvé dans la réponse : {raw_text}")

        scores = json.loads(raw_text[json_start:json_end])

    except Exception as e:
        print("⚠️ Échec de l'évaluation RAGAS-style:", e)

        scores = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "feedback": f"Évaluation échouée : {str(e)}",
        }

    # Normalisation des scores
    faithfulness = safe_float(scores.get("faithfulness"))
    relevance = safe_float(scores.get("relevance"))
    completeness = safe_float(scores.get("completeness"))
    feedback = str(scores.get("feedback", ""))

    normalized_scores = {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "completeness": completeness,
        "feedback": feedback,
    }

    # Envoi vers Langfuse si on a un trace_id
    if trace_id:
        try:
            langfuse.create_score(
                trace_id=trace_id,
                name="faithfulness",
                value=faithfulness,
                comment=feedback,
            )

            langfuse.create_score(
                trace_id=trace_id,
                name="relevance",
                value=relevance,
                comment=feedback,
            )

            langfuse.create_score(
                trace_id=trace_id,
                name="completeness",
                value=completeness,
                comment=feedback,
            )

            langfuse.flush()
            print("✅ Scores envoyés à Langfuse:", normalized_scores)

        except Exception as e:
            print("⚠️ Impossible d'envoyer les scores à Langfuse:", e)

    return normalized_scores