"""
Pinecone Ingestion — Embed child chunks and upsert to Pinecone.

Usage:
    python scripts/ingest_pinecone.py

Reads data/chunks/child_chunks.jsonl, embeds each chunk with
Gemini Embedding 2, and upserts to the Pinecone 'indian-law' index.

Includes rate limiting to stay within API quotas.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Import after env is loaded
from app.config import get_settings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

BATCH_SIZE = 50           # Vectors per upsert batch
EMBED_BATCH_SIZE = 20     # Texts per embedding API call
RATE_LIMIT_DELAY = 1.0    # Seconds between embedding batches
CHILD_CHUNKS_FILE = Path("data/chunks/child_chunks.jsonl")


def create_index_if_needed(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Create the Pinecone index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name in existing:
        print(f"Index '{index_name}' already exists.")
        return

    print(f"Creating index '{index_name}' (dim={dimension}, metric=cosine)...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    print("Waiting for index to be ready...")
    time.sleep(10)
    print("Index created.")


def embed_batch(texts: list[str], settings) -> list[list[float]]:
    """Embed a batch of texts using Gemini Embedding 2."""
    client = genai.Client(api_key=settings.google_api_key_resolved)

    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        embeddings.append(list(result.embeddings[0].values))

    return embeddings


def main():
    """Main ingestion pipeline."""
    settings = get_settings()

    if not CHILD_CHUNKS_FILE.exists():
        print(f"Error: {CHILD_CHUNKS_FILE} not found.")
        print("Run scripts/chunk_data.py first.")
        sys.exit(1)

    # Load chunks
    print(f"Loading chunks from {CHILD_CHUNKS_FILE}...")
    chunks = []
    with open(CHILD_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} child chunks.\n")

    # Initialize Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    create_index_if_needed(pc, settings.PINECONE_INDEX_NAME, settings.EMBEDDING_DIMENSIONS)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    # Process in batches
    total_upserted = 0
    batch_vectors = []

    for i, chunk in enumerate(chunks):
        # Embed one chunk at a time (with rate limiting per batch)
        if i > 0 and i % EMBED_BATCH_SIZE == 0:
            time.sleep(RATE_LIMIT_DELAY)

        try:
            text = chunk["text"]
            metadata = chunk["metadata"].copy()
            metadata["text"] = text  # Store text in metadata for retrieval

            embedding = embed_batch([text], settings)[0]

            batch_vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata,
            })

            # Upsert when batch is full
            if len(batch_vectors) >= BATCH_SIZE:
                index.upsert(vectors=batch_vectors)
                total_upserted += len(batch_vectors)
                print(f"  Upserted {total_upserted}/{len(chunks)} vectors...")
                batch_vectors = []

        except Exception as e:
            print(f"  Error processing chunk {chunk['id']}: {e}")
            time.sleep(2)  # Back off on errors
            continue

    # Upsert remaining vectors
    if batch_vectors:
        index.upsert(vectors=batch_vectors)
        total_upserted += len(batch_vectors)

    print(f"\nDone! Upserted {total_upserted} vectors to '{settings.PINECONE_INDEX_NAME}'.")

    # Verify
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
