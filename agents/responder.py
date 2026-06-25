import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Charger les variables d'environnement (.env)
load_dotenv()

def get_responder_agent():
    """
    Initialise le modèle Claude d'Anthropic avec un prompt système 
    dédié à la formulation de réponses professionnelles.
    """
    # Vérification de sécurité pour s'assurer que la clé est présente
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("La variable ANTHROPIC_API_KEY est manquante dans le fichier .env")

    # On ajoute # type: ignore à la fin de la ligne pour supprimer l'alerte Pylance

    llm = ChatAnthropic(  # type: ignore
    model_name="claude-sonnet-4-6",
    temperature=0.5,
    )

    # Création du prompt de comportement pour notre agent d'entreprise
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es l'agent répondeur d'un Copilot d'entreprise de pointe. "
            "Ton rôle est de formuler des réponses claires, précises et professionnelles "
            "en te basant uniquement sur le contexte fourni. Si tu ne connais pas la réponse, "
            "dis-le poliment sans inventer d'informations."
        ),
        ("human", "Contexte disponible :\n{context}\n\nQuestion de l'utilisateur : {question}")
    ])

    # Association du prompt et du modèle (Chaîne de traitement)
    return prompt | llm

# Création de l'instance de l'agent
responder_agent = get_responder_agent()