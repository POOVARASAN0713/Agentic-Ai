from utils.realtime_voice import process_realtime_voice_query
from utils.document_loader import load_faqs_from_bytes
from utils.text_to_speech import detect_speech_language
from utils.voice_pipeline import process_voice_message


class FakeAudio:
    name = "voice.webm"

    def getvalue(self):
        return b"audio"


def test_realtime_voice_query_processes_transcript_and_returns_audio():
    calls = []

    def workflow(query):
        calls.append(query)
        return {"agent": "faq", "answer": "Your order is out for delivery.", "confidence": 0.95}

    def text_to_speech(text, lang):
        return f"{lang}:{text}".encode("utf-8")

    result = process_realtime_voice_query(
        "Where is my order?",
        workflow=workflow,
        text_to_speech=text_to_speech,
    )

    assert calls == ["Where is my order?"]
    assert result["transcript"] == "Where is my order?"
    assert result["response"]["agent"] == "faq"
    assert result["audio"] == b"en:Your order is out for delivery."
    assert result["language"] == "en"
    assert result["error"] is None


def test_realtime_voice_empty_transcript_handling():
    result = process_realtime_voice_query("")
    assert result["transcript"] == ""
    assert result["response"] is None
    assert result["audio"] == b""
    assert "Empty" in result["error"]


def test_realtime_voice_multilingual_speech_detection():
    def workflow(query):
        return {"agent": "faq", "answer": "உங்கள் ஆர்டர் விநியோகத்தில் உள்ளது."}

    def text_to_speech(text, lang):
        return f"{lang}:{text}".encode("utf-8")

    result = process_realtime_voice_query(
        "கடவுச்சொல்லை எவ்வாறு மாற்றுவது?",
        workflow=workflow,
        text_to_speech=text_to_speech,
    )

    assert result["language"] == "ta"
    assert result["audio"].startswith(b"ta:")


def test_voice_pipeline_reuses_existing_workflow_and_creates_audio():
    calls = []

    def workflow(query):
        calls.append(query)
        return {"agent": "faq", "answer": "Your order is on the way."}

    result = process_voice_message(
        FakeAudio(),
        workflow=workflow,
        speech_to_text=lambda _audio: "Where is my order?",
        text_to_speech=lambda text: text.encode("utf-8"),
    )

    assert calls == ["Where is my order?"]
    assert result["transcript"] == "Where is my order?"
    assert result["response"]["answer"] == "Your order is on the way."
    assert result["audio"] == b"Your order is on the way."


def test_empty_audio_does_not_call_workflow():
    calls = []
    result = process_voice_message(
        None,
        workflow=lambda _query: calls.append("called"),
        speech_to_text=lambda _audio: "",
        text_to_speech=lambda _text: b"audio",
    )

    assert result["transcript"] == ""
    assert result["response"] is None
    assert calls == []


def test_stt_failure_is_returned_without_crashing():
    result = process_voice_message(
        FakeAudio(),
        speech_to_text=lambda _audio: (_ for _ in ()).throw(RuntimeError("STT down")),
    )

    assert result["stt_error"] == "STT down"
    assert result["response"] is None


def test_tts_failure_keeps_text_response():
    result = process_voice_message(
        FakeAudio(),
        workflow=lambda _query: {"answer": "Text remains available."},
        speech_to_text=lambda _audio: "Question",
        text_to_speech=lambda _text: (_ for _ in ()).throw(RuntimeError("TTS down")),
    )

    assert result["response"]["answer"] == "Text remains available."
    assert result["audio"] is None
    assert result["tts_error"] == "TTS down"


def test_workflow_failure_keeps_transcript_and_does_not_call_tts():
    result = process_voice_message(
        FakeAudio(),
        workflow=lambda _query: (_ for _ in ()).throw(RuntimeError("AI down")),
        speech_to_text=lambda _audio: "Where is my order?",
        text_to_speech=lambda _text: (_ for _ in ()).throw(AssertionError("TTS should not run")),
    )

    assert result["transcript"] == "Where is my order?"
    assert result["response"] is None
    assert result["workflow_error"] == "AI down"


def test_multilingual_tts_language_detection():
    assert detect_speech_language("உங்கள் ஆர்டர்") == "ta"
    assert detect_speech_language("आपका ऑर्डर") == "hi"
    assert detect_speech_language("Your order") == "en"


def test_uploaded_faq_csv_is_parsed():
    faqs = load_faqs_from_bytes(
        b"id,question,answer\n1,How do I track my order?,Use the tracking page.\n"
    )

    assert faqs == [
        {
            "id": "1",
            "question": "How do I track my order?",
            "answer": "Use the tracking page.",
            "category": "general",
        }
    ]


def test_uploaded_faq_csv_requires_question_and_answer():
    try:
        load_faqs_from_bytes(b"id,title\n1,Missing fields\n")
    except ValueError as error:
        assert "question and answer" in str(error)
    else:
        raise AssertionError("Invalid FAQ CSV should be rejected")