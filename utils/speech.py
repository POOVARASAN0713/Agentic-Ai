"""Speech helpers for the Streamlit customer support interface."""

from typing import Any

from utils.config import config


def transcribe_audio(audio_file: Any) -> str:
    """Transcribe a Streamlit audio upload with Groq Whisper."""
    if not config.validate_api_keys()["groq_configured"]:
        raise ValueError("Voice questions require a configured GROQ_API_KEY.")

    from groq import Groq

    filename = getattr(audio_file, "name", None) or "question.wav"
    transcription = Groq(api_key=config.GROQ_API_KEY).audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=(filename, audio_file.getvalue()),
        response_format="text",
    )
    text = transcription if isinstance(transcription, str) else transcription.text
    return text.strip()
