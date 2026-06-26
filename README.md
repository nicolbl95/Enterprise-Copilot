---
title: Enterprise Knowledge Copilot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🤖 Enterprise Knowledge Copilot

Enterprise Knowledge Copilot est une application RAG multi-agents permettant d’interroger une documentation d’entreprise.

L’application combine :

- **Gradio** pour l’interface web
- **FastAPI** pour l’API REST
- **LangGraph** pour l’orchestration des agents
- **Qdrant Cloud** pour la recherche vectorielle
- **Anthropic Claude** pour la génération et l’évaluation des réponses
- **Langfuse** pour l’observabilité et le tracing

## ✨ Fonctionnalités

- Poser des questions sur des documents d’entreprise
- Récupérer le contexte pertinent depuis Qdrant
- Générer des réponses structurées avec Claude
- Afficher les sources consultées
- Afficher le parcours des agents
- Évaluer automatiquement la qualité des réponses avec un pattern LLM-as-judge
- Afficher des scores de qualité :
  - Fidélité
  - Pertinence
  - Complétude

## 🧠 Architecture

L’application suit un workflow multi-agents :

1. **Agent de retrieval**  
   Recherche les passages pertinents dans la base vectorielle Qdrant.

2. **Agent de routage / grading**  
   Vérifie si le contexte récupéré est suffisant pour répondre à la question.

3. **Agent conversationnel**  
   Génère la réponse finale à partir de la question, du contexte documentaire et de l’historique de conversation.

4. **Agent évaluateur**  
   Évalue automatiquement la réponse avec une approche RAGAS-style / LLM-as-judge.

## 🚀 Déploiement

Ce Space lance l’interface Gradio et démarre l’API FastAPI en arrière-plan.

Les variables sensibles sont configurées dans les secrets HuggingFace Spaces :

- `ANTHROPIC_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `USE_REDIS_CHECKPOINT`

Pour le déploiement HuggingFace Spaces, le checkpoint Redis est désactivé :

```env
USE_REDIS_CHECKPOINT=false
```

La mémoire conversationnelle reste active pendant la session utilisateur grâce à l’historique transmis entre Gradio, FastAPI et LangGraph.

## 📊 Observabilité

Langfuse est utilisé pour tracer les appels LLM, suivre le pipeline RAG et enregistrer les scores de qualité des réponses générées.

## 🧪 Exemple de question

```text
Quels sont les outils d'embeddings à maîtriser ?
```

Le Copilot retourne :

- Une réponse structurée
- Le document source utilisé
- Le parcours des agents
- Les scores d’évaluation qualité

## 🛠️ Stack technique

- Python
- Gradio
- FastAPI
- LangGraph
- LangChain
- LlamaIndex
- Qdrant Cloud
- Anthropic Claude
- Langfuse