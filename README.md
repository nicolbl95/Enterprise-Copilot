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

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-121212?style=for-the-badge&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-6B46C1?style=for-the-badge&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-7B3FE4?style=for-the-badge&logoColor=white)
![Qdrant Cloud](https://img.shields.io/badge/Qdrant_Cloud-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![Anthropic API](https://img.shields.io/badge/Anthropic_API-Claude-191919?style=for-the-badge&logo=anthropic&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-000000?style=for-the-badge&logoColor=white)
![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

[![🚀 Démo Live](https://img.shields.io/badge/%F0%9F%9A%80_D%C3%A9mo_Live-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/nicolbl95/enterprise-knowledge-copilot)

---

## 📖 Description

**Enterprise Knowledge Copilot** est une application web IA permettant d’interroger une documentation d’entreprise à l’aide d’une architecture **RAG multi-agents**.

L’application simule un assistant de connaissance interne pour **NovaTech Industries**, une entreprise fictive. L’utilisateur peut poser des questions sur un document PDF d’entreprise, puis obtenir une réponse structurée basée sur le contexte documentaire récupéré depuis une base vectorielle.

Le projet combine une interface **Gradio**, une API backend **FastAPI**, une orchestration multi-agents avec **LangGraph**, une recherche vectorielle avec **Qdrant Cloud**, des réponses générées avec **Claude via l’API Anthropic**, et une observabilité avec **Langfuse**.

L’objectif du projet est de démontrer une architecture IA proche d’un cas d’usage professionnel : recherche documentaire, routage intelligent, mémoire conversationnelle, génération de réponse sourcée, évaluation qualité et déploiement cloud.

ℹ️ **Démo en ligne :**
Vous pouvez tester directement l’application sans installation, sans VS Code et sans environnement local ici :
https://huggingface.co/spaces/nicolbl95/enterprise-knowledge-copilot

---

## ✨ Fonctionnalités clés

* 🤖 **Architecture RAG multi-agents avec LangGraph**
  Le traitement est séparé en plusieurs étapes spécialisées : retrieval, routage, génération et évaluation.

* 📄 **Question-réponse sur documentation d’entreprise**
  L’utilisateur interroge un document PDF interne représentant la documentation de NovaTech Industries.

* 🔎 **Recherche sémantique avec Qdrant Cloud**
  Les passages pertinents sont récupérés depuis une base vectorielle hébergée dans le cloud.

* 🧠 **Génération avec Claude via Anthropic API**
  Les réponses sont générées par Claude à partir de la question, du contexte documentaire et de l’historique de conversation.

* 🌍 **Interface bilingue français / anglais**
  L’utilisateur peut basculer entre français et anglais avec deux boutons **FR / EN**.
  Quand l’anglais est sélectionné, l’interface et les réponses du Copilot sont en anglais, même si le PDF ou la question sont en français.

* 💬 **Mémoire conversationnelle pendant la session**
  Le Copilot conserve l’historique de la conversation pendant la session utilisateur grâce au passage de l’historique entre Gradio, FastAPI et LangGraph.

* 📁 **Affichage des sources**
  La réponse indique les documents consultés pour produire la réponse.

* 👣 **Affichage du parcours des agents**
  L’interface affiche les étapes suivies par le système, par exemple retrieval Qdrant, routage, fallback web ou évaluation.

* 📊 **Évaluation automatique LLM-as-judge**
  Une étape d’évaluation attribue des scores de qualité à la réponse :

  * fiabilité / fidélité ;
  * pertinence ;
  * complétude.

* 🚀 **Déploiement cloud avec HuggingFace Spaces**
  L’application est accessible en ligne via une démo indépendante de l’environnement local du développeur.

---

## 🧠 Architecture du système

L’application repose sur une architecture RAG multi-agents.

### 1. Agent de retrieval

L’agent de retrieval interroge la base vectorielle **Qdrant Cloud** afin de récupérer les passages les plus pertinents du document NovaTech.

Technologies principales :

* `Qdrant Cloud`
* `LlamaIndex`
* `HuggingFaceEmbedding`
* `BAAI/bge-small-en-v1.5`

### 2. Agent de routage / grading

L’agent de routage utilise Claude pour vérifier si le contexte documentaire ou l’historique conversationnel permet de répondre correctement à la question.

Il décide ensuite si le système peut générer une réponse directement ou s’il doit utiliser un fallback de recherche complémentaire.

### 3. Agent conversationnel

L’agent conversationnel génère la réponse finale à partir de :

* la question utilisateur ;
* les passages récupérés depuis Qdrant ;
* l’historique de conversation ;
* la langue sélectionnée dans l’interface.

Cet agent utilise **Claude via l’API Anthropic**.

### 4. Agent évaluateur

L’agent évaluateur relit la question, le contexte et la réponse générée afin de produire une évaluation qualité au format JSON.

Les scores affichés permettent d’estimer :

* la fiabilité de la réponse par rapport au contexte ;
* la pertinence par rapport à la question ;
* la complétude de la réponse.

---

## 🔁 Flux de traitement

```text
Question utilisateur
        ↓
Interface Gradio
        ↓
API FastAPI
        ↓
LangGraph
        ↓
Agent de retrieval
        ↓
Recherche sémantique dans Qdrant Cloud
        ↓
Agent de routage / grading
        ↓
Génération avec Claude via Anthropic API
        ↓
Évaluation LLM-as-judge
        ↓
Réponse + sources + parcours agents + scores qualité
        ↓
Affichage dans l’interface Gradio
```

---

## 🛠️ Technologies utilisées

| Technologie                | Rôle                                                |
| -------------------------- | --------------------------------------------------- |
| **Python 3.11**            | Langage principal du projet                         |
| **Gradio**                 | Interface web interactive                           |
| **FastAPI**                | API REST backend                                    |
| **LangGraph**              | Orchestration du workflow multi-agents              |
| **LangChain**              | Messages, mémoire et composants LLM                 |
| **LlamaIndex**             | Construction de l’index documentaire et retrieval   |
| **Qdrant Cloud**           | Base vectorielle cloud pour la recherche sémantique |
| **HuggingFace Embeddings** | Modèle d’embeddings pour vectoriser les documents   |
| **Anthropic API**          | Génération et évaluation des réponses avec Claude   |
| **Langfuse**               | Observabilité, tracing et suivi des appels LLM      |
| **HuggingFace Spaces**     | Déploiement cloud de la démo                        |
| **VS Code**                | Environnement de développement                      |

---

## 📂 Structure du projet

```text
├── agents/                         # Agents IA et orchestration RAG
│   ├── __init__.py
│   ├── ingestion.py                # Ingestion des documents dans Qdrant
│   └── orchestrator.py             # Workflow RAG multi-agents avec LangGraph
│
├── api/                            # Backend FastAPI
│   └── routes/
│       └── chat.py                 # Routes API /chat, /ingest et /health
│
├── core/                           # Graphe principal du Copilot
│   └── graph.py                    # Wrapper LangGraph appelé par FastAPI
│
├── data/                           # Documents source locaux
│   └── NovaTech.pdf                # Document d’entreprise fictif
│
├── interface/                      # Interface utilisateur
│   ├── app.py                      # Ancienne ou alternative interface Streamlit
│   ├── gradio_app.py               # Interface principale Gradio
│   └── assets/
│       └── NovaTech.pdf            # PDF accessible depuis l’interface
│
├── app.py                          # Point d’entrée HuggingFace Spaces
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation principale du projet
└── .gitignore                      # Fichiers exclus du suivi Git
```

> Les fichiers sensibles ou locaux comme `.env`, `venv/`, `.vscode/`, `__pycache__/` et les clés API ne doivent jamais être envoyés sur GitHub.

---

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/nicolbl95/enterprise-knowledge-copilot.git
cd enterprise-knowledge-copilot
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l’environnement virtuel

Sur Windows avec Git Bash :

```bash
source venv/Scripts/activate
```

Sur Windows avec PowerShell :

```powershell
.\venv\Scripts\Activate.ps1
```

Sur macOS ou Linux :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🔐 Configuration des variables d’environnement

Créez un fichier `.env` à la racine du projet.

```env
ANTHROPIC_API_KEY=votre_cle_anthropic
LANGFUSE_PUBLIC_KEY=votre_cle_publique_langfuse
LANGFUSE_SECRET_KEY=votre_cle_secrete_langfuse
LANGFUSE_BASE_URL=https://cloud.langfuse.com

QDRANT_URL=votre_url_qdrant_cloud
QDRANT_API_KEY=votre_cle_qdrant
QDRANT_COLLECTION=novatech_docs

USE_REDIS_CHECKPOINT=false
```

⚠️ Le fichier `.env` ne doit jamais être envoyé sur GitHub.

Il doit être présent dans `.gitignore` :

```text
.env
venv/
__pycache__/
*.pyc
.vscode/
```

---

## ▶️ Lancement local

L’application locale utilise deux processus :

1. l’API FastAPI ;
2. l’interface Gradio.

### 1. Lancer FastAPI

Dans un premier terminal :

```bash
source venv/Scripts/activate
python -m uvicorn api.routes.chat:app --reload --port 8000
```

L’API sera disponible à l’adresse :

```text
http://localhost:8000
```

### 2. Lancer Gradio

Dans un second terminal :

```bash
source venv/Scripts/activate
python interface/gradio_app.py
```

L’interface sera disponible à l’adresse :

```text
http://localhost:7860
```

---

## 🌐 Démo en ligne

L’application est déployée avec **HuggingFace Spaces** :

https://huggingface.co/spaces/nicolbl95/enterprise-knowledge-copilot

Cette démo permet de tester le projet directement depuis un navigateur, sans installation locale, sans VS Code et sans configuration manuelle.

---

## 🚀 Déploiement HuggingFace Spaces

Le fichier `app.py` à la racine sert de point d’entrée pour HuggingFace Spaces.

Il démarre :

* le backend **FastAPI** en arrière-plan ;
* l’interface **Gradio** exposée publiquement.

Les variables sensibles sont configurées dans les secrets HuggingFace Spaces :

```text
ANTHROPIC_API_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_BASE_URL
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
USE_REDIS_CHECKPOINT
```

Pour HuggingFace Spaces, le checkpoint Redis est désactivé :

```env
USE_REDIS_CHECKPOINT=false
```

La mémoire conversationnelle reste active pendant la session grâce à l’historique transmis entre Gradio, FastAPI et LangGraph.

---

## 📊 Observabilité

**Langfuse** est utilisé pour suivre les appels LLM et observer le pipeline RAG.

L’observabilité permet notamment de vérifier :

* les appels à Claude via Anthropic API ;
* les étapes du pipeline ;
* les réponses générées ;
* les éventuelles erreurs ;
* les scores qualité produits par l’évaluateur.

---

## 🧪 Exemples de questions

L’interface propose plusieurs questions d’exemple :

```text
Quel est le délai de première réponse pour le support Business Critical ?
```

```text
Quelles certifications NovaTech a-t-elle obtenues, et lesquelles sont en cours ?
```

```text
Quelle remise obtient-on avec un engagement de 36 mois ?
```

```text
Quel est le budget formation annuel par collaborateur et combien de jours sont garantis ?
```

```text
Qu'est-ce que le RPO et le RTO en plan Premium, et que se passe-t-il si le SLA n'est pas atteint ?
```

Le Copilot retourne :

* une réponse structurée ;
* les sources utilisées ;
* le parcours des agents ;
* les scores d’évaluation qualité.

---

## 📊 Scores qualité

Après chaque réponse, l’application affiche une auto-évaluation de la qualité.

Exemple :

```text
📊 Évaluation IA — Fiabilité: 95% | Pertinence: 85% | Complétude: 75%
```

Ces scores sont estimés par un agent évaluateur LLM-as-judge :

| Score                    | Signification                                                               |
| ------------------------ | --------------------------------------------------------------------------- |
| **Fiabilité / fidélité** | La réponse reste fidèle au contexte disponible et limite les hallucinations |
| **Pertinence**           | La réponse répond correctement à la question posée                          |
| **Complétude**           | La réponse couvre les informations importantes attendues                    |

Ces scores ne sont pas des mesures mathématiques absolues. Ils servent à démontrer une couche supplémentaire de contrôle qualité dans le pipeline RAG.

---

## ⚠️ Limites actuelles

* Les scores qualité sont estimés par un LLM et ne garantissent pas une vérité absolue.
* La qualité des réponses dépend de la qualité du document source et du retrieval Qdrant.
* La mémoire conversationnelle est active pendant la session, mais n’est pas persistée durablement sur HuggingFace Spaces.
* Le projet est basé sur une entreprise fictive et un document de démonstration.
* Les réponses générées par IA peuvent contenir des erreurs et doivent être vérifiées dans un contexte professionnel réel.
* Le fallback web est limité et sert surtout à démontrer un routage alternatif.

---

## 🔮 Améliorations possibles

* Ajouter un mode d’administration pour téléverser plusieurs documents.
* Ajouter un dashboard d’observabilité directement dans l’interface.
* Ajouter une exportation PDF ou Word des réponses.
* Ajouter une gestion multi-utilisateurs avec authentification.
* Ajouter une mémoire persistante compatible cloud.
* Ajouter des tests unitaires et tests d’intégration.
* Améliorer l’évaluation qualité avec des métriques RAGAS plus complètes.
* Ajouter un mode streaming pour afficher les réponses progressivement.
* Ajouter une meilleure visualisation des passages sources utilisés.
* Ajouter un système de feedback utilisateur sur la qualité des réponses.

---

## 👨‍💻 Auteur

Projet développé par **nicolbl95**.

GitHub : https://github.com/nicolbl95

Démo live : https://huggingface.co/spaces/nicolbl95/enterprise-knowledge-copilot

---

## 📄 Licence

Ce projet est publié à des fins d’apprentissage, de démonstration et de portfolio.

---

## ✅ Résumé

Ce projet démontre la mise en place d’une application IA complète combinant :

* interface web ;
* API backend ;
* traitement de documents PDF ;
* architecture RAG multi-agents ;
* recherche vectorielle cloud ;
* orchestration LangGraph ;
* génération avec Claude via Anthropic API ;
* mémoire conversationnelle ;
* interface bilingue français / anglais ;
* observabilité Langfuse ;
* évaluation automatique LLM-as-judge ;
* déploiement cloud avec HuggingFace Spaces.

Il illustre une approche pratique de l’ingénierie LLM appliquée à la recherche documentaire d’entreprise.
