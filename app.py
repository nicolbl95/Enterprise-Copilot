import os
import sys
import subprocess
import threading
import time


def start_api():
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

time.sleep(5)

from interface.gradio_app import demo

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)