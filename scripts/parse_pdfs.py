"""
PDF Parser — Extracts clean text from legal PDF documents.

Usage:
    python scripts/parse_pdfs.py

Reads all PDFs from data/raw/ and outputs .txt files alongside them.
"""

import re
import sys
from pathlib import Path

from PyPDF2 import PdfReader


RAW_DIR = Path("data/raw")


def clean_text(text: str) -> str:
    """Clean extracted PDF text for better chunking."""
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Fix broken words across lines (e.g., "pun-\nishment" → "punishment")
    text = re.sub(r"-\n\s*", "", text)

    # Collapse multiple newlines but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page headers/footers (common patterns in legal PDFs)
    text = re.sub(r"(?m)^Page \d+ of \d+\s*$", "", text)
    text = re.sub(r"(?m)^\d+\s*$", "", text)  # standalone page numbers

    return text.strip()


def parse_pdf(pdf_path: Path) -> str:
    """Extract text from a single PDF file."""
    print(f"  Parsing: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))

    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
        else:
            print(f"    Warning: Page {i + 1} has no extractable text")

    full_text = "\n\n".join(pages)
    return clean_text(full_text)


def main():
    """Parse all PDFs in data/raw/ and save as .txt files."""
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR} does not exist. Create it and add your PDFs.")
        sys.exit(1)

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}/")
        print("Add your legal PDFs (IPC, CrPC, Constitution, etc.) to data/raw/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) in {RAW_DIR}/\n")

    for pdf_path in sorted(pdf_files):
        try:
            text = parse_pdf(pdf_path)
            output_path = pdf_path.with_suffix(".txt")
            output_path.write_text(text, encoding="utf-8")
            print(f"    → Saved: {output_path.name} ({len(text):,} characters)\n")
        except Exception as e:
            print(f"    ✗ Error parsing {pdf_path.name}: {e}\n")

    print("Done! Text files are ready in data/raw/")
    print("Next step: python scripts/chunk_data.py")


if __name__ == "__main__":
    main()
