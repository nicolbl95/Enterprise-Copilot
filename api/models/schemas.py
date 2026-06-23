from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="La question ou la commande de l'utilisateur destinée au Copilot")
    conversation_id: Optional[str] = Field(None, description="L'identifiant unique de la conversation pour maintenir le contexte")

class ChatResponse(BaseModel):
    response: str = Field(..., description="La réponse générée par le système multi-agents")
    conversation_id: str = Field(..., description="L'identifiant de la conversation associé à cette réponse")