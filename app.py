import os
import sys
import subprocess
import threading
import time
from pathlib import Path


# ============================================================
# 1. CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "interface" / "assets" / "NovaTech.pdf"


# ============================================================
# 2. LANCEMENT FASTAPI EN ARRIÈRE-PLAN
# ============================================================

def start_api():
    """
    Lance FastAPI en arrière-plan sur le port 8000.
    Gradio appelle ensuite http://localhost:8000/chat.
    """
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.routes.chat:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()

# Laisse quelques secondes à FastAPI pour démarrer
time.sleep(5)


# ============================================================
# 3. LANCEMENT GRADIO
# ============================================================

from interface.gradio_app import demo

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    allowed_paths=[str(PDF_PATH)],
)