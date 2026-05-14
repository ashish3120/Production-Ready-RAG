"""
Request and Response schemas for the Indian Legal RAG API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Main RAG query request."""
    query: str = Field(..., min_length=3, description="Legal question to answer")
    context: str = Field(
        default="general",
        description="Optional context filter: 'ipc', 'crpc', 'constitution', 'general'"
    )
    detailed: bool = Field(
        default=False,
        description="If true, use Gemini for deep reasoning instead of Groq"
    )


class ExplainRequest(BaseModel):
    """Request to explain a legal term in plain language."""
    term: str = Field(..., min_length=2, description="Legal term to explain")
    context: str = Field(default="", description="Optional context for the term")


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class SourceInfo(BaseModel):
    """A single retrieved source chunk."""
    act: str = ""
    section: str = ""
    chapter: str = ""
    text: str = ""
    score: float = 0.0
    page: Optional[int] = None
    expanded: bool = False  # True if parent chunk was used


class QueryResponse(BaseModel):
    """Response from the main RAG query endpoint."""
    answer: str
    sources: list[SourceInfo] = []
    llm_used: str = "groq"
    confidence: float = 0.0
    query_type: str = "simple"


class ExplainResponse(BaseModel):
    """Response from the explain endpoint."""
    term: str
    explanation: str
    related_sections: list[str] = []
    llm_used: str = "groq"


class SectionResponse(BaseModel):
    """Response for a bare act section lookup."""
    act: str
    section: str
    text: str
    chapter: str = ""
    source_file: str = ""


class ActInfo(BaseModel):
    """Information about an indexed act."""
    name: str
    year: int = 0
    section_count: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    pinecone_connected: bool = False
    index_name: str = ""
    vector_count: int = 0
