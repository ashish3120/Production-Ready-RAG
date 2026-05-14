"""
Bare Act passage retriever.
Fetches raw text of an entire act section from stored chunks.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_chunks_by_act(act_name: str) -> list[dict]:
    """Load all child chunks for a given act from the JSONL file."""
    child_file = Path("data/chunks/child_chunks.jsonl")
    if not child_file.exists():
        logger.warning("Child chunks file not found: %s", child_file)
        return []

    chunks = []
    act_lower = act_name.lower()

    with open(child_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunk_act = chunk.get("metadata", {}).get("act", "").lower()
            if act_lower in chunk_act or chunk_act in act_lower:
                chunks.append(chunk)

    return chunks


def get_bare_act_passage(act_name: str, section: str) -> str | None:
    """
    Retrieve the bare act text for a specific section.

    Args:
        act_name: Full or partial act name (e.g., "Indian Penal Code")
        section: Section number (e.g., "302")

    Returns:
        Raw text of the section, or None if not found.
    """
    chunks = _load_chunks_by_act(act_name)

    matching = [
        c for c in chunks
        if c.get("metadata", {}).get("section", "") == str(section)
    ]

    if not matching:
        return None

    # Sort by chunk index and join
    matching.sort(key=lambda c: c.get("id", ""))
    return "\n\n".join(c.get("text", "") for c in matching)


def list_indexed_acts() -> list[dict]:
    """
    List all unique acts present in the child chunks file.

    Returns:
        List of {name, year, section_count} dicts.
    """
    child_file = Path("data/chunks/child_chunks.jsonl")
    if not child_file.exists():
        return []

    acts: dict[str, dict] = {}

    with open(child_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            meta = chunk.get("metadata", {})
            act_name = meta.get("act", "Unknown")
            year = meta.get("year", 0)
            section = meta.get("section", "")

            if act_name not in acts:
                acts[act_name] = {
                    "name": act_name,
                    "year": year,
                    "sections": set(),
                }
            if section:
                acts[act_name]["sections"].add(section)

    return [
        {
            "name": info["name"],
            "year": info["year"],
            "section_count": len(info["sections"]),
        }
        for info in sorted(acts.values(), key=lambda a: a["name"])
    ]
