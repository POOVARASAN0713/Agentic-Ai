"""In-memory multilingual text-to-speech adapter for voice support responses."""

import asyncio
from io import BytesIO
from typing import Optional


def detect_speech_language(text: str) -> str:
    """Select a gTTS language code from common supported scripts."""
    if any("\u0b80" <= character <= "\u0bff" for character in text):
        return "ta"
    if any("\u0900" <= character <= "\u097f" for character in text):
        return "hi"
    if any("\u0600" <= character <= "\u06ff" for character in text):
        return "ar"
    if any("\u4e00" <= character <= "\u9fff" for character in text):
        return "zh-CN"
    return "en"


def synthesize_speech(text: str, language: Optional[str] = None) -> bytes:
    """Return MP3 audio bytes without writing user content to disk."""
    clean_text = (text or "").strip()
    if not clean_text:
        return b""

    import edge_tts

    voices = {
        "en": "en-US-AriaNeural",
        "ta": "ta-IN-PallaviNeural",
        "hi": "hi-IN-SwaraNeural",
        "ar": "ar-SA-ZariyahNeural",
        "zh-CN": "zh-CN-XiaoxiaoNeural",
    }
    voice = voices.get(language or detect_speech_language(clean_text), voices["en"])

    async def collect_audio() -> bytes:
        stream = edge_tts.Communicate(clean_text, voice)
        audio = BytesIO()
        async for chunk in stream.stream():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
        return audio.getvalue()

    return asyncio.run(collect_audio())