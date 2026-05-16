<div align="center">
  <h1>⚖️ Indian Legal RAG System</h1>
  <p><strong>A Production-Ready Retrieval-Augmented Generation System for Indian Law</strong></p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/VectorDB-Pinecone-yellow.svg" alt="Pinecone">
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-orange.svg" alt="LLMs">
</div>

---

## 📖 Overview

The **Indian Legal RAG** is a full-stack AI system designed to answer complex legal questions based on the Indian legal framework. 

It uses a highly optimized **Hierarchical RAG architecture**, combining fast retrieval via Pinecone with a dual-LLM routing strategy using **Groq (LLaMA 3.3 70B)** for lightning-fast factual lookups and **Google Gemini 2.5 Flash** for deep constitutional reasoning. The user interacts with the system through a premium, interactive **Streamlit frontend**.

> **New to the project?** Please read the [HOW_TO_USE.md](./HOW_TO_USE.md) guide for step-by-step setup and ingestion instructions.

---

## 🌍 Live Demo

The application is fully deployed and accessible over the web:

- **Frontend (Streamlit):** [Live App](https://ashish3120-production-ready-rag-streamlit-app-wjwvt2.streamlit.app/)
- **Backend API (Render):** `https://indian-legal-rag-api.onrender.com`

---

## ✨ Key Features

1. **Hierarchical Chunking (Parent-Child Strategy)**
   - **Child Chunks (400 tokens):** Used strictly for high-precision vector similarity search.
   - **Parent Chunks (1500 tokens):** Dynamically fetched when the vector confidence score is below 0.75, giving the LLM the surrounding legal context to prevent hallucination.
   
2. **Intelligent Query Router**
   - **Groq Pipeline:** Automatically selected for direct queries (e.g., *"What is Section 302 of the IPC?"*). Delivers answers in < 1 second.
   - **Gemini Pipeline:** Automatically triggered via keyword detection for complex queries (e.g., *"Compare the old CrPC with the new BNSS"*).
   
3. **Dedicated Legal Tools**
   - **Section Lookup:** Directly fetch bare acts without passing through the LLM.
   - **Citation Generator:** Formats legal sources rigorously (e.g., *Section 302 of the Indian Penal Code*).
   - **Plain English Explainer:** Dedicated endpoint to demystify heavy legal jargon for standard users.

4. **Premium User Interface**
   - A fully responsive, dark-mode Streamlit dashboard featuring live SSE (Server-Sent Events) streaming, query routing visualization, and system health monitoring.

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│              Streamlit Frontend (Premium UI)             │
│   • SSE Streaming    • Tool UIs    • Health Dashboard    │
└──────────────────────────┬───────────────────────────────┘
                           │ POST /api/query
                           ▼
┌──────────────────────────────────────────────────────────┐
│               FastAPI Backend (port 8000)                │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Query Router│  │ RAG Engine   │  │ Tool Service   │  │
│  │ • Classify  │  │ • Embed query│  │ • IPC Lookup   │  │
│  │ • Route     │─▶│ • Search     │  │ • Bare Acts    │  │
│  └─────────────┘  │ • Parent Exp │  │ • Citations    │  │
│                   └──────┬───────┘  └────────────────┘  │
│                   ┌──────▼───────┐                      │
│                   │  LLM Layer   │                      │
│                   │ Groq | Gemini│                      │
│                   └──────────────┘                      │
└──────────────────────────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Pinecone Vector DB  │
              │   gemini-embedding-2    │
              │   768-dims, cosine      │
              └─────────────────────────┘
```

---

## 📂 Project Structure

```text
legal-rag/
├── .env                        # API keys (Groq, Gemini, Pinecone)
├── requirements.txt            # Python dependencies
├── main.py                     # FastAPI server entrypoint
├── streamlit_app.py            # Streamlit Frontend application
├── render.yaml                 # Render Infrastructure as Code
├── Dockerfile                  # Containerization for FastAPI
│
├── app/
│   ├── config.py               # Centralized Pydantic settings
│   ├── models.py               # Request/Response schemas
│   ├── rag.py                  # Core RAG logic (Embed + Retrieve + Parent Expand)
│   ├── llm.py                  # Groq & Gemini client wrappers & prompts
│   ├── router.py               # Classification routing (Groq vs Gemini)
│   └── tools/                  # Dedicated legal tools
│
├── data/                       # (Ignored in Git, created locally)
│   ├── raw/                    # Place raw legal PDFs here
│   └── chunks/                 # Generated parent/child JSONL chunks
│
├── scripts/
│   ├── parse_pdfs.py           # Extracts and cleans PDF text
│   ├── chunk_data.py           # Creates the Parent-Child chunk mapping
│   └── ingest_pinecone.py      # Rate-limited Pinecone batch ingestion
│
└── HOW_TO_USE.md               # Step-by-step setup and ingestion tutorial
```

---

## 🚀 Quick Start (Local Development)

Ensure you have your API keys ready in `.env` and `data/raw/` populated with legal PDFs. For full data ingestion instructions, refer to [HOW_TO_USE.md](./HOW_TO_USE.md).

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Start the FastAPI Backend Server
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Start the Streamlit Frontend (In a new terminal)
streamlit run streamlit_app.py
```

## ☁️ Deployment

- **Backend (Render):** The backend is containerized via `Dockerfile` and configured for 1-click deployment on Render using the included `render.yaml` Blueprint.
- **Frontend (Streamlit Community Cloud):** Deploy `streamlit_app.py` via Streamlit Cloud. Ensure you configure `.streamlit/secrets.toml` or Streamlit Cloud Secrets with `API_BASE_URL` pointing to your deployed backend.
