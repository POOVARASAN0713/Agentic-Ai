# AI Customer Support with Multi-Agent System 🤖💬

An intelligent, production-grade AI customer support application built using **LangGraph**, **LangChain**, **ChromaDB**, **Sentence Transformers**, **Streamlit**, **Groq**, and **Google Gemini**.

Designed for high accuracy, online-marketplace customer support, multilingual capability, and resilience with automated LLM fallback.

---

## 🌟 Key Architecture & Highlights

- **Multi-Agent Orchestration**: Intelligent query routing via LangGraph across specialized agents:
  - **Orchestrator Agent**: Analyzes intent and routes queries dynamically.
  - **FAQ Agent**: Fast semantic similarity matching against vector-indexed customer FAQs.
  - **RAG Agent**: Document-grounded retrieval using vector-indexed policy documents (PDF/TXT).
  - **Escalation Agent**: Detects low confidence, sensitive queries, or human handoff requests.
- **Resilient Fallback LLM Architecture**: Uses **Groq** (`llama-3.3-70b-versatile`) as the primary fast provider, with seamless fallback to **Google Gemini** (`gemini-2.0-flash`) if Groq hits rate limits or experiences downtime.
- **Multilingual Embeddings**: Powered by `paraphrase-multilingual-mpnet-base-v2` for cross-lingual understanding in English, Tamil, and Hindi.
- **Auditable Streamlit Frontend**: Includes a side-by-side Debug / viva-demonstration panel showing routing decisions, similarity scores, and retrieved source chunks.

---

## 📂 Project Structure

```text
Agentic Ai/
├── agents/                 # Multi-agent implementations (Orchestrator, FAQ, RAG, Escalation)
│   └── __init__.py
├── data/                   # Raw FAQ datasets and PDF policy documents
│   ├── faqs.csv            # 50-100 realistic customer support FAQs
│   ├── documents/          # Unified online marketplace policy document
│   └── processed/          # Preprocessed vector chunks
├── vectorstore/            # Chroma DB persistent vector collections
│   └── chroma/
├── utils/                  # Central utilities
│   ├── __init__.py
│   ├── config.py           # Centralized configuration & thresholds
│   ├── llm.py              # LLM Manager with primary (Groq) -> fallback (Gemini)
│   ├── logging_utils.py    # Structured logging utility
│   ├── embeddings.py       # SentenceTransformer embedding loader
│   └── document_loader.py  # PDF/TXT loader and chunker
├── evaluation/             # Test benchmark dataset & metric evaluation scripts
├── frontend/               # Streamlit web application
│   └── app.py
├── tests/                  # Pytest unit & integration test suite
├── notebooks/              # Exploratory design notebooks
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore configuration
├── requirements.txt        # Production python dependencies
├── main.py                 # Main CLI test & entry script
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11 installed on your computer.
- Git installed.

### 2. Create Python Virtual Environment
Open your terminal in the project directory:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create `.env`:
```bash
cp .env.example .env
```
Open `.env` in your code editor and add your free API keys:
- `GROQ_API_KEY`: Get from [Groq Console](https://console.groq.com/)
- `GEMINI_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/)

---

## 🚀 Running the Project

Run the complete CLI diagnostics:

```bash
python main.py
```

Run the automated tests:

```bash
pytest -q
```

Launch the Streamlit chat interface:

```bash
streamlit run frontend/app.py
```

Run the offline routing benchmark:

```bash
python evaluation/evaluate.py
```

To also evaluate the live orchestrator, use `python evaluation/evaluate.py --workflow`. Live RAG answers require a configured Groq or Gemini API key; without one, the system correctly escalates when answer generation is unavailable.

## Deployment

This project uses a FastAPI backend and a Streamlit frontend, so they should be deployed separately:

### FastAPI API on Vercel

1. Import the repository into Vercel.
2. Keep the project root at the repository root.
3. Add `GROQ_API_KEY` and/or `GEMINI_API_KEY` in Vercel Project Settings > Environment Variables.
4. Deploy. The API health check will be available at `/health` and chat requests at `/api/chat`.

The Vercel adapter is `api/index.py`. The local Chroma database is not suitable for serverless persistence; production deployments should use a hosted vector database or package a small read-only index with the deployment.

### Streamlit frontend

Deploy `frontend/app.py` on Streamlit Community Cloud, Render, or Railway. Configure the same LLM environment variables there. The current frontend calls the orchestration code locally, so use a Python host for the complete Streamlit experience.

Never commit real API keys. If credentials were previously placed in a tracked file, revoke and regenerate them before deployment.

---

## 📊 Phase Roadmap

- [x] **Phase 1**: Project Setup, Environment Isolation, Configuration & Resilient LLM Layer
- [x] **Phase 2**: Dataset Creation (FAQs & Document Processing)
- [x] **Phase 3**: Vector DB & Embeddings setup (ChromaDB & Sentence Transformers)
- [x] **Phase 4**: FAQ Specialist Agent
- [x] **Phase 5**: RAG Specialist Agent
- [x] **Phase 6**: Escalation Specialist Agent
- [x] **Phase 7**: LangGraph Orchestrator & State Management
- [x] **Phase 8**: Response Validation Guardrails
- [x] **Phase 9**: Streamlit Debug & Chat Frontend UI
- [x] **Phase 10**: Benchmark Evaluation Framework & Rule-Based Baseline Comparison
