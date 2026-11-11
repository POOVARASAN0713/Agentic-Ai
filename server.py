"""FastAPI REST API Backend Server for AI Customer Support Multi-Agent System."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import process_support_query
from utils.realtime_voice import process_realtime_voice_query

app = FastAPI(
    title="AI Customer Support Multi-Agent API",
    description="FastAPI REST Backend exprosing LangGraph Multi-Agent Orchestration and Realtime Voice pipeline.",
    version="1.0.0",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., description="User customer support text query", example="How do I reset my password?")


class VoiceRequest(BaseModel):
    transcript: str = Field(..., description="User voice transcript", example="What is the refund window?")
    language: Optional[str] = Field(default=None, description="Optional target language code (e.g. en, ta, hi)")


@app.get("/")
@app.get("/health")
def health_check():
    """Health diagnostic check endpoint."""
    return {
        "status": "online",
        "system": "AI Customer Support Multi-Agent API",
        "agents": ["FAQ", "RAG", "Escalation", "Guardrails"],
        "orchestrator": "LangGraph StateGraph",
    }


@app.post("/api/chat")
def handle_chat_query(payload: ChatRequest):
    """Process a text customer support query through the LangGraph Multi-Agent Orchestrator."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        response = process_support_query(query)
        return {"status": "success", "data": response}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Orchestration Error: {str(error)}")


@app.post("/api/voice")
def handle_voice_query(payload: VoiceRequest):
    """Process a real-time voice query transcript through the LangGraph Orchestrator and Speech Engine."""
    transcript = payload.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript string cannot be empty.")
    try:
        result = process_realtime_voice_query(transcript=transcript, language=payload.language)
        return {"status": "success", "data": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Voice Pipeline Error: {str(error)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
