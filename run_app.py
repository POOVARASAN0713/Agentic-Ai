"""Launcher script to run both the FastAPI REST Backend Server and Streamlit Frontend App concurrently."""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_UVICORN = PROJECT_ROOT / ".venv" / "Scripts" / "uvicorn.exe"
VENV_STREAMLIT = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable


def run_servers():
    print("=" * 65)
    print(" === LAUNCHING AI CUSTOMER SUPPORT SYSTEM (BACKEND + FRONTEND) ===")
    print("=" * 65)
    print(" 🚀 Backend API Server  : http://localhost:8000")
    print(" 🎨 Streamlit Frontend  : http://localhost:8501")
    print("=" * 65 + "\n")

    # 1. Start FastAPI Backend Server
    backend_cmd = [str(VENV_PYTHON), "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
    print(f"[STARTING BACKEND] {' '.join(backend_cmd)}")
    backend_process = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))

    time.sleep(2)

    # 2. Start Streamlit Frontend App
    frontend_cmd = [str(VENV_PYTHON), "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"]
    print(f"[STARTING FRONTEND] {' '.join(frontend_cmd)}")
    frontend_process = subprocess.Popen(frontend_cmd, cwd=str(PROJECT_ROOT))

    try:
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n[SHUTTING DOWN] Stopping Backend and Frontend servers...")
        backend_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    run_servers()
