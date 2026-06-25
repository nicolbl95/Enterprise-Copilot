import os
import json
from typing import Dict, Any

from anthropic import Anthropic
from langfuse import get_client


anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
langfuse = get_client()


def extract_text_safely(message: Any) -> str:
    """Extrait proprement tout le texte d'une réponse Anthropic."""
    if not hasattr(message, "content") or not message.content:
        return ""

    texts = []

    for block in message.content:
        if hasattr(block, "text"):
            texts.append(str(block.text))
        elif isinstance(block, dict) and "text" in block:
            texts.append(str(block["text"]))
        else:
            texts.append(str(block))

    return "\n".join(texts)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur en float entre 0 et 1."""
    try:
        number = float(value)
        return max(0.0, min(1.0, number))
    except Exception:
        return default


def extract_json(raw_text: str) -> Dict[str, Any]:
    """Extrait un objet JSON même si Claude ajoute du texte autour."""
    raw_text = raw_text.strip()

    print("🧪 RAW EVALUATOR OUTPUT:")
    print(raw_text)

    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}") + 1

    if json_start == -1 or json_end <= json_start:
        raise ValueError(f"Aucun JSON valide trouvé dans : {raw_text}")

    json_text = raw_text[json_start:json_end]

    return json.loads(json_text)


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

    print("🧪 EVALUATOR CONTEXT LENGTH:", len(context))
    print("🧪 EVALUATOR TRACE ID:", trace_id)

    eval_prompt = f"""Tu es un évaluateur qualité pour un système RAG.

Évalue la réponse selon 3 critères entre 0 et 1.

Critères :
- faithfulness : la réponse est-elle fidèle au contexte fourni, sans hallucination ?
- relevance : la réponse répond-elle directement à la question ?
- completeness : la réponse est-elle suffisamment complète et utile ?

Réponds UNIQUEMENT avec un JSON valide.
N'ajoute aucun markdown.
N'ajoute aucun commentaire hors JSON.

Question :
{question}

Contexte :
{context[:2500]}

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
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": eval_prompt
                }
            ],
        )

        raw_text = extract_text_safely(result)
        scores = extract_json(raw_text)

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

    except Exception as e:
        print("⚠️ Échec de l'évaluation RAGAS-style:", repr(e))

        normalized_scores = {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "feedback": f"Évaluation échouée : {str(e)}",
        }

        return normalized_scores

    # Envoi vers Langfuse si trace_id disponible
    if trace_id:
        try:
            langfuse.create_score(
                trace_id=trace_id,
                name="faithfulness",
                value=normalized_scores["faithfulness"],
                comment=normalized_scores["feedback"],
            )

            langfuse.create_score(
                trace_id=trace_id,
                name="relevance",
                value=normalized_scores["relevance"],
                comment=normalized_scores["feedback"],
            )

            langfuse.create_score(
                trace_id=trace_id,
                name="completeness",
                value=normalized_scores["completeness"],
                comment=normalized_scores["feedback"],
            )

            langfuse.flush()
            print("✅ Scores envoyés à Langfuse:", normalized_scores)

        except Exception as e:
            print("⚠️ Impossible d'envoyer les scores à Langfuse:", repr(e))

    else:
        print("⚠️ Aucun trace_id disponible, scores non attachés à Langfuse.")

    return normalized_scores