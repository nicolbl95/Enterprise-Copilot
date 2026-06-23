from fastapi import FastAPI

# On initialise notre application FastAPI
app = FastAPI(
    title="Enterprise Copilot API",
    description="L'API backend de notre Copilot Multi-Agents",
    version="0.1.0"
)

# On crée notre première route (un point d'accès)
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de ton Enterprise Copilot ! 🚀"}