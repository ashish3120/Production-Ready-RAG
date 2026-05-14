"""
Legal citation formatter.
Generates properly formatted legal citations from source metadata.
"""


def format_citation(source: dict) -> str:
    """
    Format a single source into a standard Indian legal citation.

    Args:
        source: Dict with keys like act, section, chapter, page, source_file

    Returns:
        Formatted citation string.
    """
    act = source.get("act", "Unknown Act")
    section = source.get("section", "")
    chapter = source.get("chapter", "")
    page = source.get("page")

    parts = []

    if section:
        # Determine prefix: "Article" for Constitution, "Section" for others
        if "constitution" in act.lower():
            parts.append(f"Article {section}")
        else:
            parts.append(f"Section {section}")

    parts.append(f"of the {act}")

    if chapter:
        parts.append(f"(Chapter {chapter})")

    if page:
        parts.append(f"[Page {page}]")

    return " ".join(parts)


def format_citations(sources: list[dict]) -> list[str]:
    """Format multiple sources into a list of citations."""
    return [format_citation(s) for s in sources]


def build_citation_block(sources: list[dict]) -> str:
    """
    Build a citation block for inclusion in API responses.

    Returns a formatted markdown-style citation section.
    """
    if not sources:
        return ""

    lines = ["**Legal References:**"]
    for i, source in enumerate(sources, 1):
        citation = format_citation(source)
        lines.append(f"{i}. {citation}")

    return "\n".join(lines)
