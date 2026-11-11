"""Customer-facing Streamlit chat UI for the existing support workflow."""

import base64
import html
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import process_support_query
from utils.speech import transcribe_audio
from utils.text_to_speech import detect_speech_language, synthesize_speech


st.set_page_config(
    page_title="AI Support",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#f4f5f7; --muted:#a7acb8; --line:#363944; --paper:#0d1017; --accent:#ff9800; --panel:#181b23; --sidebar:#262730; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] { background:#242631 !important; border-color:#242631 !important; }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input { color:#f4f5f7 !important; caret-color:#ff9800 !important; background:#242631 !important; }
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder { color:#a7acb8 !important; opacity:1 !important; }
    [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid #30323c; }
    [data-testid="stSidebar"] > div:first-child { padding:2rem 1.2rem; }
    .brand-mark { display:flex; align-items:center; gap:.65rem; margin-bottom:2rem; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.2rem; }
    .brand-dot { width:2rem; height:2rem; display:grid; place-items:center; color:white; background:var(--accent); border-radius:10px; }
    .page-shell { max-width:900px; margin:0 auto; padding:2rem 1.2rem 1rem; }
    .chat-header { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:2rem; }
    .chat-header h1 { margin:0; font-family:'Manrope',sans-serif; font-size:1.35rem; letter-spacing:-.04em; }
    .online { color:#63d297; font-size:.82rem; font-weight:600; }
    .welcome { text-align:center; padding:4.2rem 1rem 2rem; }
    .welcome h2 { font-family:'Manrope',sans-serif; font-size:2.15rem; letter-spacing:-.06em; margin:.5rem 0 .7rem; }
    .welcome p { color:var(--muted); margin:0 auto; max-width:540px; }
    .message-row { display:flex; gap:.8rem; margin:1.3rem 0; align-items:flex-start; }
    .message-row.user { justify-content:flex-end; }
    .avatar { flex:0 0 2rem; height:2rem; border-radius:9px; display:grid; place-items:center; font-size:.78rem; font-weight:700; }
    .avatar.ai { background:var(--accent); color:white; }
    .avatar.user { background:#ff4b4b; color:white; order:2; }
    .bubble { max-width:min(720px,82%); line-height:1.65; padding:.15rem 0; }
    .user .bubble { background:#191c24; border-radius:16px 16px 4px 16px; padding:.75rem 1rem; line-height:1.5; }
    .audio-action { margin-top:.7rem; }
    .history-label { color:#a7acb8; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.6rem 0 .65rem; }
    .history-item { color:#e1e3e8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding:.55rem .7rem; border-radius:9px; font-size:.88rem; }
    .history-item:hover { background:#30323d; }
    .composer-note { text-align:center; color:#858a98; font-size:.72rem; margin-top:.55rem; }
    div[data-testid="stButton"] > button { border-radius:10px; border:1px solid #3b3e49; background:#20232c; color:#f4f5f7; }
    div[data-testid="stButton"] > button:hover { border-color:var(--accent); color:var(--accent); }
    @media (max-width:700px) { .page-shell { padding:1rem .75rem; } .welcome { padding-top:2rem; } .welcome h2 { font-size:1.8rem; } .bubble { max-width:88%; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
def submit_query(query: str) -> None:
    clean_query = (query or "").strip()
    if not clean_query:
        return
    st.session_state.messages.append({"role": "user", "content": clean_query})
    response = process_support_query(clean_query)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "language": detect_speech_language(clean_query),
            "debug": {
                "agent": response.get("agent"),
                "route": response.get("route"),
                "confidence": response.get("confidence"),
                "reason": response.get("reason"),
                "sources": response.get("sources", []),
                "ticket_id": response.get("ticket_id"),
                "guardrail_passed": response.get("guardrail_passed"),
                "trace": response.get("trace", []),
            },
        }
    )


def render_audio_action(message_index: int, text: str, language: str = "en") -> None:
    audio_bytes = b""
    if message_index not in st.session_state.audio_cache:
        try:
            st.session_state.audio_cache[message_index] = synthesize_speech(text, language)
        except Exception:
            st.session_state.audio_cache[message_index] = b""

    audio_bytes = st.session_state.audio_cache[message_index]
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else ""
    quoted_text = json.dumps(text)
    components.html(
        f"""
        <style>
          body {{ margin:0; background:transparent; font-family:'DM Sans',sans-serif; }}
          button {{ border:0; background:transparent; color:#ffb74d; cursor:pointer; padding:5px 0; font-size:13px; font-weight:600; }}
          button.playing {{ color:#63d297; }}
        </style>
        <button id="audioButton" type="button">🔊 Hear response</button>
        <button id="stopButton" type="button" class="stop">■ Stop audio</button>
        <script>
          const button = document.getElementById('audioButton');
          const stopButton = document.getElementById('stopButton');
          const responseText = {quoted_text};
          const responseLanguage = {json.dumps({"en": "en-US", "ta": "ta-IN", "hi": "hi-IN", "ar": "ar-SA", "zh-CN": "zh-CN"}.get(language, "en-US"))};
          const audio = {f"new Audio('data:audio/mpeg;base64,{encoded_audio}')" if encoded_audio else "null"};
          const resetControls = () => {{
            button.disabled = false;
            button.className = '';
            button.textContent = '🔊 Hear response';
          }};
          const speakWithBrowser = () => {{
            if (!('speechSynthesis' in window)) {{
              button.textContent = '⚠️ Audio unavailable';
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(responseText);
            utterance.lang = responseLanguage;
            utterance.onend = resetControls;
            utterance.onerror = resetControls;
            window.speechSynthesis.speak(utterance);
            button.textContent = '🔊 Playing...';
          }};
          stopButton.onclick = () => {{
            if (audio) audio.pause();
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            resetControls();
          }};
          button.onclick = async () => {{
            button.disabled = true;
            button.className = 'playing';
            button.textContent = '⏳ Preparing audio...';
            try {{
              if (!audio) {{
                speakWithBrowser();
                return;
              }}
              await audio.play();
              button.textContent = '🔊 Playing...';
            }} catch (error) {{
              speakWithBrowser();
            }}
          }};
          if (audio) {{
            audio.onended = resetControls;
            audio.onerror = speakWithBrowser;
          }}
        </script>
        """,
        height=40,
        scrolling=False,
    )


def render_composer() -> None:
    submitted = st.chat_input(
        "Ask about orders, refunds, shipping...",
        accept_audio=True,
        audio_sample_rate=16000,
    )
    if submitted:
        if hasattr(submitted, "audio") and submitted.audio is not None:
            try:
                query = transcribe_audio(submitted.audio)
            except Exception as error:
                st.error(f"Voice typing failed: {error}")
                return
        else:
            query = submitted.text if hasattr(submitted, "text") else str(submitted)
        if query.strip():
            submit_query(query)
            st.rerun()
        else:
            st.warning("No speech was detected. Please try recording again.")


init_state()
query_params = st.query_params
incoming_query = query_params.get("chat_query", "") or query_params.get("voice_query", "")
if incoming_query:
    st.query_params.clear()
    submit_query(incoming_query)
    st.rerun()

with st.sidebar:
    st.markdown('<div class="brand-mark"><span class="brand-dot">✦</span> AI Support</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.audio_cache = {}
        st.rerun()
    st.markdown('<div class="history-label">Recent conversations</div>', unsafe_allow_html=True)
    user_queries = [message["content"] for message in st.session_state.messages if message["role"] == "user"]
    if user_queries:
        for query in reversed(user_queries[-6:]):
            st.markdown(f'<div class="history-item">{html.escape(query)}</div>', unsafe_allow_html=True)
    else:
        st.caption("Your recent questions will appear here.")

st.markdown('<div class="page-shell">', unsafe_allow_html=True)
st.markdown('<div class="chat-header"><h1>AI Support Assistant</h1><span class="online">● Online</span></div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><div class="brand-dot" style="margin:auto;">✦</div><h2>How can we help today?</h2><p>Ask about orders, delivery, returns, refunds, account access, and more.</p></div>',
        unsafe_allow_html=True,
    )
else:
    for message_index, message in enumerate(st.session_state.messages):
        role = message["role"]
        avatar = "AI" if role == "assistant" else "You"
        safe_content = html.escape(message["content"]).replace("\n", "<br>")
        avatar_class = "ai" if role == "assistant" else "user"
        st.markdown(
            f'<div class="message-row {role}"><div class="avatar {avatar_class}">{avatar}</div><div class="bubble">{safe_content}</div></div>',
            unsafe_allow_html=True,
        )
        if role == "assistant":
            render_audio_action(
                message_index,
                message["content"],
                message.get("language", "en"),
            )
            if message.get("debug"):
                with st.expander("Debug Details"):
                    st.json(message["debug"])

st.markdown('</div>', unsafe_allow_html=True)
render_composer()
