"""
Hierarchical Chunking — Parent-Child chunk generation for legal documents.

Usage:
    python scripts/chunk_data.py

Reads .txt files from data/raw/ and outputs:
  - data/chunks/parent_chunks.jsonl
  - data/chunks/child_chunks.jsonl
"""

import json
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────────────────────────
# Chunk Parameters (tuned for legal text)
# ──────────────────────────────────────────────

CHILD_CHUNK_SIZE = 400      # tokens (~300 words) — for retrieval
CHILD_CHUNK_OVERLAP = 80    # ~20% overlap — prevents splitting mid-clause
PARENT_CHUNK_SIZE = 1500    # tokens (~1100 words) — for context expansion
PARENT_CHUNK_OVERLAP = 100  # smooth boundary between parents

RAW_DIR = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")

# ──────────────────────────────────────────────
# Act metadata extractor
# ──────────────────────────────────────────────

ACT_METADATA = {
    "ipc_1860": {"act": "Indian Penal Code", "year": 1860},
    "crpc_1973": {"act": "Code of Criminal Procedure", "year": 1973},
    "constitution_india": {"act": "Constitution of India", "year": 1950},
    "evidence_act_1872": {"act": "Indian Evidence Act", "year": 1872},
    "cpc_1908": {"act": "Civil Procedure Code", "year": 1908},
    "ibc_2016": {"act": "Insolvency and Bankruptcy Code", "year": 2016},
    "motor_vehicles_act": {"act": "Motor Vehicles Act", "year": 1988},
    "consumer_protection_act": {"act": "Consumer Protection Act", "year": 2019},
    "pocso_act": {"act": "POCSO Act", "year": 2012},
    "it_act_2000": {"act": "Information Technology Act", "year": 2000},
    "bns_2023": {"act": "Bharatiya Nyaya Sanhita", "year": 2023},
    "bnss_2023": {"act": "Bharatiya Nagarik Suraksha Sanhita", "year": 2023},
}


def infer_act_metadata(filename: str) -> dict:
    """Infer act name and year from filename."""
    stem = Path(filename).stem.lower()

    # Check exact matches
    for key, meta in ACT_METADATA.items():
        if key in stem:
            return meta.copy()

    # Fallback: use filename as act name
    return {
        "act": stem.replace("_", " ").title(),
        "year": 0,
    }


def extract_section_number(text: str) -> str:
    """Try to extract a section or article number from the beginning of a chunk."""
    # Match patterns like "Section 302.", "Art. 21", "302."
    patterns = [
        r"(?:Section|Sec\.?)\s*(\d+[A-Za-z]*)",
        r"(?:Article|Art\.?)\s*(\d+[A-Za-z]*)",
        r"^(\d+[A-Za-z]*)\.\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def extract_chapter(text: str) -> str:
    """Try to extract chapter from text."""
    match = re.search(
        r"(?:Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)",
        text[:500],
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return ""


# ──────────────────────────────────────────────
# Main Chunking
# ──────────────────────────────────────────────

def chunk_document(text: str, doc_id: str, source_file: str) -> tuple[list[dict], list[dict]]:
    """
    Split a document into hierarchical parent-child chunks.

    Returns:
        (parent_chunks, child_chunks) — lists of dicts with id, text, metadata
    """
    act_meta = infer_act_metadata(source_file)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,  # character-based (approximate tokens)
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    parent_chunks_out = []
    child_chunks_out = []

    parent_texts = parent_splitter.split_text(text)

    for i, parent_text in enumerate(parent_texts):
        parent_id = f"{doc_id}-parent-{i}"
        parent_section = extract_section_number(parent_text)
        parent_chapter = extract_chapter(parent_text)

        parent_chunks_out.append({
            "id": parent_id,
            "text": parent_text,
            "metadata": {
                "act": act_meta["act"],
                "year": act_meta["year"],
                "section": parent_section,
                "chapter": parent_chapter,
                "chunk_type": "parent",
                "source_file": source_file,
            },
        })

        # Split parent into child chunks
        child_texts = child_splitter.split_text(parent_text)
        for j, child_text in enumerate(child_texts):
            child_id = f"{doc_id}-child-{i}-{j}"
            child_section = extract_section_number(child_text) or parent_section
            child_chapter = extract_chapter(child_text) or parent_chapter

            child_chunks_out.append({
                "id": child_id,
                "text": child_text,
                "metadata": {
                    "act": act_meta["act"],
                    "year": act_meta["year"],
                    "section": child_section,
                    "chapter": child_chapter,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "source_file": source_file,
                },
            })

    return parent_chunks_out, child_chunks_out


def main():
    """Process all .txt files in data/raw/ and generate JSONL chunk files."""
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR} does not exist.")
        print("Run scripts/parse_pdfs.py first to extract text from PDFs.")
        sys.exit(1)

    txt_files = list(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {RAW_DIR}/")
        print("Run scripts/parse_pdfs.py first.")
        sys.exit(1)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    all_parents = []
    all_children = []

    print(f"Found {len(txt_files)} text file(s) in {RAW_DIR}/\n")

    for txt_path in sorted(txt_files):
        print(f"  Chunking: {txt_path.name}")
        text = txt_path.read_text(encoding="utf-8")
        doc_id = txt_path.stem

        parents, children = chunk_document(text, doc_id, txt_path.name)
        all_parents.extend(parents)
        all_children.extend(children)

        print(f"    → {len(parents)} parent chunks, {len(children)} child chunks\n")

    # Write output
    parent_file = CHUNKS_DIR / "parent_chunks.jsonl"
    child_file = CHUNKS_DIR / "child_chunks.jsonl"

    with open(parent_file, "w", encoding="utf-8") as f:
        for chunk in all_parents:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with open(child_file, "w", encoding="utf-8") as f:
        for chunk in all_children:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_parents)} parent chunks, {len(all_children)} child chunks")
    print(f"Saved to: {parent_file}")
    print(f"Saved to: {child_file}")
    print("\nNext step: python scripts/ingest_pinecone.py")


if __name__ == "__main__":
    main()
