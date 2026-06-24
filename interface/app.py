import streamlit as st
import os
import sys
from pathlib import Path

# Ajout de la racine du projet au chemin de recherche Python pour pouvoir importer l'agent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import de la logique RAG développée dans agents/search.py
from agents.search import setup_search_engine, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# Configuration de la page Streamlit
st.set_page_config(page_title="Enterprise Copilot", page_icon="🤖", layout="centered")
st.title("🤖 Enterprise Copilot")
st.subheader("Posez vos questions sur les documents de l'entreprise")

# --- INITIALISATION DE LA MÉMOIRE LLAMAINDEX ---
if "chat_memory" not in st.session_state:
    # Initialise un buffer de mémoire pour retenir le contexte de l'historique
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Initialisation de l'historique d'affichage des messages dans Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre Copilot. Que souhaitez-vous savoir aujourd'hui ?"}
    ]

# Affichage des messages historiques à l'écran
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Zone de saisie de l'utilisateur
if user_query := st.chat_input("Votre question ici..."):
    
    # 1. Afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # 2. Générer la réponse avec le Chat Engine (Moteur avec Mémoire)
    with st.chat_message("assistant"):
        with st.spinner("Le Copilot consulte les manuels et l'historique..."):
            try:
                # Récupération de l'index vectoriel Qdrant
                index = setup_search_engine()
                retriever = index.as_retriever(similarity_top_k=2)
                
                # Configuration du Chat Engine qui combine Contexte RAG + Mémoire
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
                
                # Exécution de la requête (on utilise .chat() au lieu de .query())
                response = chat_engine.chat(user_query)
                answer_text = str(response)
                
                # Affichage du résultat dans l'UI
                st.write(answer_text)
                
                # Sauvegarde dans l'historique d'affichage
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                error_msg = f"Mince, une erreur est survenue : {e}"
                st.error(error_msg)