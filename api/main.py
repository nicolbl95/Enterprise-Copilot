from fastapi import FastAPI
from api.models.schemas import ChatRequest, ChatResponse
import uuid

app = FastAPI(
    title="Enterprise Copilot API",
    description="L'API backend de notre Copilot Multi-Agents",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de ton Enterprise Copilot ! 🚀"}

# Notre nouvelle route POST pour recevoir les questions de l'utilisateur
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Si l'utilisateur n'envoie pas d'id de conversation, on en génère un nouveau
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    # Simulation temporaire de la réponse de l'IA (en attendant de connecter nos agents)
    fake_ai_response = f"Echo de l'IA : J'ai bien reçu ton message : '{request.message}'"
    
    return ChatResponse(
        response=fake_ai_response,
        conversation_id=conv_id
    )