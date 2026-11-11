"""Realtime Voice Assistant adapter connecting live audio/speech stream inputs directly to the existing LangGraph multi-agent orchestrator."""

from typing import Any, Callable, Dict, Optional

from agents.orchestrator import process_support_query
from utils.text_to_speech import detect_speech_language, synthesize_speech


def process_realtime_voice_query(
    transcript: str,
    language: Optional[str] = None,
    workflow: Callable[[str], Dict[str, Any]] = process_support_query,
    text_to_speech: Callable[[str, Optional[str]], bytes] = synthesize_speech,
) -> Dict[str, Any]:
    """
    Processes a real-time speech transcript through the existing multi-agent LangGraph workflow
    and synthesizes an audio response for automatic voice playback.
    """
    clean_transcript = (transcript or "").strip()
    if not clean_transcript:
        return {
            "transcript": "",
            "response": None,
            "audio": b"",
            "error": "Empty speech transcript provided.",
        }

    # Execute exact LangGraph multi-agent workflow
    try:
        response = workflow(clean_transcript)
    except Exception as error:
        return {
            "transcript": clean_transcript,
            "response": None,
            "audio": b"",
            "error": f"AI Workflow Execution Error: {str(error)}",
        }

    answer = str(response.get("answer", "")).strip()
    if not answer:
        return {
            "transcript": clean_transcript,
            "response": response,
            "audio": b"",
            "error": "The AI workflow returned an empty answer.",
        }

    # Detect speech language if not explicitly provided
    target_lang = language or detect_speech_language(answer)

    # Synthesize audio bytes for playback
    audio_bytes = b""
    tts_error = None
    try:
        audio_bytes = text_to_speech(answer, target_lang)
    except Exception as error:
        tts_error = f"Speech Synthesis Warning: {str(error)}"

    return {
        "transcript": clean_transcript,
        "response": response,
        "audio": audio_bytes,
        "language": target_lang,
        "error": tts_error,
    }
