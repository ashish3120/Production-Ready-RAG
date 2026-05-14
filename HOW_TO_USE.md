# ⚖️ Indian Legal RAG — How to Use Guide

Welcome to the **Indian Legal RAG** system! This guide will walk you through the entire process: from downloading raw legal documents to successfully querying the API.

---

## 🛑 Prerequisites

Before you start, make sure you have:
- **Python 3.10+** installed.
- API Keys for the following services:
  - **Pinecone**: [Get API Key](https://app.pinecone.io/)
  - **Google Gemini**: [Get API Key](https://aistudio.google.com/app/apikey)
  - **Groq**: [Get API Key](https://console.groq.com/keys)

---

## Step 1: Environment Setup

1. **Install Dependencies**
   Open your terminal and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Open the `.env` file in the root directory and securely add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=indian-law
   PINECONE_REGION=us-east-1
   ```

---

## Step 2: Getting Legal Data

The system requires raw legal PDFs to process. 

1. Go to official government portals like [India Code](https://www.indiacode.nic.in/) or [Supreme Court of India](https://sci.gov.in/).
2. Download PDFs for acts you want to include (e.g., `Indian Penal Code 1860`, `Constitution of India`, `Code of Criminal Procedure 1973`).
3. Place these downloaded `.pdf` files into the following directory:
   ```text
   data/raw/
   ```
   *(Ensure the file names are descriptive, e.g., `ipc_1860.pdf`, `constitution_india.pdf`)*

---

## Step 3: Data Ingestion (One-Time Setup)

Once your PDFs are in the `data/raw/` folder, run the following scripts in order to build your vector database.

### A. Extract Text
Converts the raw PDFs into clean `.txt` files.
```bash
python scripts/parse_pdfs.py
```

### B. Chunk the Data
Splits the text into parent-child chunks optimized for legal context.
```bash
python scripts/chunk_data.py
```
*This will generate `parent_chunks.jsonl` and `child_chunks.jsonl` in the `data/chunks/` directory.*

### C. Embed and Upload to Pinecone
Embeds the child chunks using Gemini Embedding 2 and upserts them to your Pinecone index.
```bash
python scripts/ingest_pinecone.py
```
*(Note: This step may take a few minutes depending on the size of your PDFs and API rate limits. The script automatically handles rate-limiting and index creation.)*

---

## Step 4: Starting the API Server

Once ingestion is complete, you can start the FastAPI server:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*(The API will be available at `http://localhost:8000`)*

---

## Step 5: Interacting with the API

You can use **cURL**, **Postman**, or any frontend application to interact with the API endpoints.

### 1. Main Legal Query Endpoint (`/api/query`)
Ask complex or factual legal questions. The system auto-routes between Groq (for facts/speed) and Gemini (for complex reasoning).

**Request:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is the punishment for theft under IPC?",
        "context": "ipc",
        "detailed": false
      }'
```

### 2. Legal Term Explanation (`/api/explain`)
Get plain-language explanations of complex legal terms.

**Request:**
```bash
curl -X POST http://localhost:8000/api/explain \
  -H "Content-Type: application/json" \
  -d '{"term": "Habeas Corpus"}'
```

### 3. Bare Act Section Lookup (`/api/section/{act}/{section}`)
Directly retrieve the raw, exact text of a specific section.

**Request:**
```bash
curl -X GET http://localhost:8000/api/section/ipc/302
```

### 4. List Indexed Acts (`/api/acts`)
Check which legal acts have been successfully ingested into your Pinecone index.

**Request:**
```bash
curl -X GET http://localhost:8000/api/acts
```

---

## 🛠 Troubleshooting

- **Server Returns 500 or 404 for `indian-law` index:** You haven't completed Step 3 (Ingestion). Ensure `ingest_pinecone.py` finished successfully.
- **`ModuleNotFoundError: No module named 'pinecone'`:** Ensure you have installed dependencies from `requirements.txt`. The package is now `pinecone` (v9+), not `pinecone-client`.
- **API Rate Limits:** If `ingest_pinecone.py` fails due to Gemini rate limits, the script will automatically retry. If it persists, reduce `EMBED_BATCH_SIZE` in the script.
