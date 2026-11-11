"""Voice input/output adapter around the existing support workflow."""

from typing import Any, Callable, Dict

from agents.orchestrator import process_support_query
from utils.realtime_voice import process_realtime_voice_query
from utils.speech import transcribe_audio
from utils.text_to_speech import synthesize_speech


def process_voice_message(
    audio_file: Any,
    workflow: Callable[[str], Dict[str, Any]] = process_support_query,
    speech_to_text: Callable[[Any], str] = transcribe_audio,
    text_to_speech: Callable[[str], bytes] = synthesize_speech,
) -> Dict[str, Any]:
    """Transcribe audio, reuse the support workflow, and synthesize its answer."""
    result: Dict[str, Any] = {
        "transcript": "",
        "response": None,
        "audio": None,
        "stt_error": None,
        "workflow_error": None,
        "tts_error": None,
    }

    try:
        transcript = (speech_to_text(audio_file) or "").strip()
    except Exception as error:
        result["stt_error"] = str(error)
        return result

    result["transcript"] = transcript
    if not transcript:
        return result

    try:
        response = workflow(transcript)
    except Exception as error:
        result["workflow_error"] = str(error)
        return result

    result["response"] = response
    answer = str(response.get("answer", "")).strip()
    if not answer:
        result["tts_error"] = "The AI workflow returned an empty response."
        return result

    try:
        result["audio"] = text_to_speech(answer)
    except Exception as error:
        result["tts_error"] = str(error)
    return result