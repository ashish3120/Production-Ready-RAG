"""
Pinecone Ingestion — Single key, simple and reliable.

Usage:
    python scripts/ingest_pinecone.py

Features:
- Uses GOOGLE_API_KEY from .env (single key)
- Auto-waits on rate limits using Google's suggested retry delay
- Checkpoint/resume — won't re-embed already done chunks
- Flat metadata (Pinecone compatible)
- Creates index if missing, polls until ready
"""

import json
import re
import sys
import os
import time
import uuid
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 50  # vectors per Pinecone upsert
RATE_LIMIT_DELAY = 0.6  # seconds between each embed call
MAX_RETRIES = 5  # max retries for network errors (not rate limits)
CHILD_CHUNKS_FILE = Path("data/chunks/child_chunks.jsonl")
CHECKPOINT_FILE = Path("data/chunks/ingest_checkpoint.txt")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        val = CHECKPOINT_FILE.read_text().strip()
        if val.isdigit():
            return int(val)
    return 0


def save_checkpoint(idx: int):
    CHECKPOINT_FILE.write_text(str(idx))


# ── Flatten metadata for Pinecone ──────────────────────────────────────────────
def flatten_metadata(chunk: dict, text: str, chunk_id: str) -> dict:
    """Pinecone rejects list values in metadata — convert to comma-joined strings."""
    flat = {"text": text, "chunk_id": chunk_id}
    for k, v in chunk.get("metadata", {}).items():
        flat[k] = ", ".join(str(x) for x in v) if isinstance(v, list) else v
    return flat


# ── Parse retry delay from 429 error ──────────────────────────────────────────
def parse_retry_delay(error_str: str) -> int:
    """Extract Google's suggested retryDelay from a 429 error (e.g. 'retryDelay': '53s')."""
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", error_str)
    if match:
        return int(match.group(1)) + 3  # add 3s safety buffer
    return 60  # fallback: wait 60s


# ── Embed single text ─────────────────────────────────────────────────────────
def embed_text(text: str, client, settings) -> list[float] | None:
    """
    Embed text with automatic rate-limit waiting.
    - 429: waits Google's suggested delay, then retries (infinite patience)
    - Network errors: retries up to MAX_RETRIES with backoff
    - Other errors: fails immediately
    """
    network_retries = 0
    while True:
        try:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=text,
                config={
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": settings.EMBEDDING_DIMENSIONS,
                },
            )
            return list(result.embeddings[0].values)

        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = parse_retry_delay(err)
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                # Don't increment — we wait patiently and retry

            elif any(x in err for x in ["503", "504", "Timeout", "handshaker"]):
                network_retries += 1
                if network_retries >= MAX_RETRIES:
                    print(f"  ❌ Network failed after {MAX_RETRIES} retries.")
                    return None
                wait = 15 * network_retries
                print(
                    f"  🌐 Network error. Retrying in {wait}s... ({network_retries}/{MAX_RETRIES})"
                )
                time.sleep(wait)

            else:
                print(f"  ❌ Unrecoverable error: {e}")
                return None


# ── Pinecone index creation ────────────────────────────────────────────────────
def create_index_if_needed(pc: Pinecone, name: str, dim: int):
    existing = [idx.name for idx in pc.list_indexes()]
    if name in existing:
        print(f"Index '{name}' already exists.")
        return

    print(f"Creating index '{name}' (dim={dim}, metric=cosine)...")
    pc.create_index(
        name=name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready", end="", flush=True)
    while not pc.describe_index(name).status["ready"]:
        print(".", end="", flush=True)
        time.sleep(2)
    print(" ✅")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    settings = get_settings()

    api_key = settings.google_api_key_resolved
    if not api_key:
        print("❌ No GOOGLE_API_KEY found in .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(f"✅ Using single API key.")
    print(
        f"   Embed model: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_DIMENSIONS}d)"
    )
    print(f"   Rate: 1 embed every {RATE_LIMIT_DELAY}s\n")

    if not CHILD_CHUNKS_FILE.exists():
        print(f"❌ {CHILD_CHUNKS_FILE} not found. Run scripts/chunk_data.py first.")
        sys.exit(1)

    # Load all chunks
    chunks = []
    with open(CHILD_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    print(f"Total child chunks: {total}")

    # Resume from checkpoint
    start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"♻️  Resuming from chunk {start_idx + 1}/{total}\n")
    else:
        print("Starting fresh.\n")

    # Pinecone setup
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    create_index_if_needed(
        pc, settings.PINECONE_INDEX_NAME, settings.EMBEDDING_DIMENSIONS
    )
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    batch_vectors = []
    total_upserted = start_idx

    for idx in range(start_idx, total):
        chunk = chunks[idx]
        text = chunk["text"]
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or str(uuid.uuid4())
        metadata = flatten_metadata(chunk, text, chunk_id)

        embedding = embed_text(text, client, settings)

        if embedding is None:
            print(f"  ⏭️  Skipping chunk {idx} — all retries failed.")
            save_checkpoint(idx + 1)
            continue

        batch_vectors.append(
            {
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata,
            }
        )

        print(f"[{idx + 1}/{total}] ✓ {chunk_id}")
        save_checkpoint(idx + 1)

        # Upsert when batch is full
        if len(batch_vectors) >= BATCH_SIZE:
            index.upsert(vectors=batch_vectors)
            total_upserted += len(batch_vectors)
            print(f"  ✅ Upserted batch — {total_upserted}/{total} done\n")
            batch_vectors = []

        time.sleep(RATE_LIMIT_DELAY)

    # Final batch
    if batch_vectors:
        index.upsert(vectors=batch_vectors)
        total_upserted += len(batch_vectors)

    print(
        f"\n🎉 Done! Ingested {total_upserted}/{total} chunks into '{settings.PINECONE_INDEX_NAME}'."
    )
    CHECKPOINT_FILE.unlink(missing_ok=True)

    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
