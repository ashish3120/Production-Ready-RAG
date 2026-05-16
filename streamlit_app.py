"""
Indian Legal RAG — Streamlit Frontend
Connects to the FastAPI backend and exposes all endpoints through a premium UI.
"""

import json
import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Legal RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { color: #f0e6ff; font-size: 2.2rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
.hero p { color: #b8a9d4; font-size: 1rem; margin-top: 0.5rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.metric-card .label { color: #8b8fa3; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #e0e0ff; font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }

/* Source cards */
.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: rgba(139, 92, 246, 0.5); }
.source-card .src-header { color: #a78bfa; font-weight: 600; font-size: 0.9rem; }
.source-card .src-score { color: #6ee7b7; font-size: 0.78rem; float: right; }
.source-card .src-text { color: #9ca3af; font-size: 0.82rem; margin-top: 0.5rem; line-height: 1.5; }

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-groq { background: rgba(99,102,241,0.2); color: #818cf8; }
.badge-gemini { background: rgba(251,191,36,0.2); color: #fbbf24; }
.badge-ok { background: rgba(52,211,153,0.2); color: #34d399; }
.badge-degraded { background: rgba(251,113,133,0.2); color: #fb7185; }

/* Act table */
.act-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.act-name { color: #c4b5fd; font-weight: 600; font-size: 0.92rem; }
.act-meta { color: #6b7280; font-size: 0.8rem; }

/* Section result */
.section-result {
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1a2e 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    line-height: 1.7;
    color: #d1d5db;
    font-size: 0.92rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────

def api_get(path: str, **kwargs):
    """GET request to the backend API."""
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Make sure `uvicorn main:app` is running on port 8000.")
        return None
    except requests.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} — {e.response.text}")
        return None


def api_post(path: str, payload: dict):
    """POST request to the backend API."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Make sure `uvicorn main:app` is running on port 8000.")
        return None
    except requests.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} — {e.response.text}")
        return None


def stream_query(payload: dict):
    """Stream tokens from the SSE endpoint."""
    try:
        with requests.post(f"{API_BASE}/api/query/stream", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API.")
    except Exception as e:
        st.error(f"Streaming error: {e}")


def render_llm_badge(llm: str):
    cls = "badge-gemini" if llm == "gemini" else "badge-groq"
    return f'<span class="badge {cls}">{llm.upper()}</span>'


def render_sources(sources: list):
    if not sources:
        return
    st.markdown("#### 📚 Sources")
    for s in sources:
        expanded_tag = " 🔄 expanded" if s.get("expanded") else ""
        score_pct = f"{s.get('score', 0) * 100:.1f}%"
        text_preview = s.get("text", "")[:300]
        st.markdown(f"""
        <div class="source-card">
            <span class="src-header">§ {s.get('section', 'N/A')} — {s.get('act', 'Unknown')}{expanded_tag}</span>
            <span class="src-score">Score: {score_pct}</span>
            <div class="src-text">{text_preview}{'…' if len(s.get('text', '')) > 300 else ''}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ Navigation")
    page = st.radio(
        "Go to",
        ["🔍 Legal Query", "📖 Section Lookup", "💡 Term Explainer", "📋 Indexed Acts", "🏥 System Health"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#6b7280;font-size:0.75rem;text-align:center;">'
        'Indian Legal RAG v1.0<br>Powered by Groq · Gemini · Pinecone</p>',
        unsafe_allow_html=True,
    )

# ── Hero Header ────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>⚖️ Indian Legal RAG</h1>
    <p>AI-powered legal research assistant for Indian Law — IPC, CrPC, Constitution & more</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 1: Legal Query (RAG)
# ══════════════════════════════════════════════════════════

if page == "🔍 Legal Query":
    st.markdown("### Ask a Legal Question")
    st.caption("Your question will be answered using Retrieval-Augmented Generation with cited Indian law sources.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Your legal question",
            placeholder="e.g., What is the punishment for murder under IPC Section 302?",
            height=100,
            key="query_input",
        )
    with col2:
        context_options = {
            "General (All Acts)": "general",
            "IPC": "ipc",
            "CrPC": "crpc",
            "Constitution": "constitution",
            "Evidence Act": "evidence",
            "CPC": "cpc",
            "BNS": "bns",
            "BNSS": "bnss",
            "IT Act": "it",
            "POCSO": "pocso",
        }
        context_label = st.selectbox("Filter by Act", list(context_options.keys()))
        context_val = context_options[context_label]
        detailed = st.checkbox("Deep reasoning (Gemini)", value=False)
        use_stream = st.checkbox("Stream response", value=True)

    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not query or len(query.strip()) < 3:
            st.warning("Please enter at least 3 characters.")
        else:
            payload = {"query": query.strip(), "context": context_val, "detailed": detailed}

            if use_stream:
                # ── Streaming Mode ──
                answer_placeholder = st.empty()
                full_answer = ""
                sources_data = None

                with st.spinner("Retrieving legal context…"):
                    for event in stream_query(payload):
                        if "token" in event:
                            full_answer += event["token"]
                            answer_placeholder.markdown(full_answer)
                        elif "sources" in event:
                            sources_data = event

                if sources_data:
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.markdown(f'<div class="metric-card"><div class="label">LLM Used</div><div class="value">{render_llm_badge(sources_data.get("llm_used", "groq"))}</div></div>', unsafe_allow_html=True)
                    with mcol2:
                        conf = sources_data.get("confidence", 0)
                        st.markdown(f'<div class="metric-card"><div class="label">Confidence</div><div class="value">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with mcol3:
                        st.markdown(f'<div class="metric-card"><div class="label">Query Type</div><div class="value" style="font-size:1rem;">{sources_data.get("query_type", "general")}</div></div>', unsafe_allow_html=True)
                    render_sources(sources_data.get("sources", []))

            else:
                # ── Blocking Mode ──
                with st.spinner("Generating answer…"):
                    resp = api_post("/api/query", payload)

                if resp:
                    st.markdown("#### 📝 Answer")
                    st.markdown(resp["answer"])
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.markdown(f'<div class="metric-card"><div class="label">LLM Used</div><div class="value">{render_llm_badge(resp.get("llm_used", "groq"))}</div></div>', unsafe_allow_html=True)
                    with mcol2:
                        conf = resp.get("confidence", 0)
                        st.markdown(f'<div class="metric-card"><div class="label">Confidence</div><div class="value">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with mcol3:
                        st.markdown(f'<div class="metric-card"><div class="label">Query Type</div><div class="value" style="font-size:1rem;">{resp.get("query_type", "general")}</div></div>', unsafe_allow_html=True)
                    render_sources(resp.get("sources", []))


# ══════════════════════════════════════════════════════════
# PAGE 2: Section Lookup
# ══════════════════════════════════════════════════════════

elif page == "📖 Section Lookup":
    st.markdown("### 📖 Bare Act Section Lookup")
    st.caption("Fetch the raw text of a specific section directly — no LLM involved.")

    col1, col2 = st.columns(2)
    with col1:
        act_id = st.selectbox(
            "Act",
            ["ipc", "crpc", "constitution", "evidence", "cpc", "bns", "bnss", "ibc", "motor", "consumer", "pocso", "it"],
            format_func=lambda x: {
                "ipc": "Indian Penal Code (IPC)",
                "crpc": "Code of Criminal Procedure (CrPC)",
                "constitution": "Constitution of India",
                "evidence": "Indian Evidence Act",
                "cpc": "Civil Procedure Code",
                "bns": "Bharatiya Nyaya Sanhita (BNS)",
                "bnss": "Bharatiya Nagarik Suraksha Sanhita (BNSS)",
                "ibc": "Insolvency & Bankruptcy Code",
                "motor": "Motor Vehicles Act",
                "consumer": "Consumer Protection Act",
                "pocso": "POCSO Act",
                "it": "Information Technology Act",
            }.get(x, x),
        )
    with col2:
        section_num = st.text_input("Section / Article Number", placeholder="e.g., 302")

    if st.button("📜 Fetch Section", type="primary", use_container_width=True):
        if not section_num.strip():
            st.warning("Please enter a section number.")
        else:
            with st.spinner("Looking up section…"):
                resp = api_get(f"/api/section/{act_id}/{section_num.strip()}")
            if resp:
                st.markdown(f'<div class="section-result">'
                            f'<strong style="color:#a78bfa;font-size:1.1rem;">§ {resp["section"]} — {resp["act"]}</strong>'
                            f'{"<br><em style=color:#6b7280>Chapter: " + resp["chapter"] + "</em>" if resp.get("chapter") else ""}'
                            f'<hr style="border-color:rgba(255,255,255,0.1);margin:0.8rem 0;">'
                            f'{resp["text"]}'
                            f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3: Term Explainer
# ══════════════════════════════════════════════════════════

elif page == "💡 Term Explainer":
    st.markdown("### 💡 Legal Term Explainer")
    st.caption("Get a plain-language explanation of any legal term using AI and cited sources.")

    term = st.text_input("Legal term", placeholder="e.g., Habeas Corpus, Cognizable Offence, Bail")

    if st.button("💬 Explain", type="primary", use_container_width=True):
        if not term or len(term.strip()) < 2:
            st.warning("Please enter at least 2 characters.")
        else:
            with st.spinner("Generating explanation…"):
                resp = api_post("/api/explain", {"term": term.strip()})

            if resp:
                st.markdown("#### 📝 Explanation")
                st.markdown(resp["explanation"])

                if resp.get("related_sections"):
                    st.markdown("#### 🔗 Related Sections")
                    for sec in resp["related_sections"]:
                        st.markdown(f"- {sec}")

                st.markdown(f'<div style="margin-top:1rem;">{render_llm_badge(resp.get("llm_used", "groq"))}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 4: Indexed Acts
# ══════════════════════════════════════════════════════════

elif page == "📋 Indexed Acts":
    st.markdown("### 📋 Indexed Acts")
    st.caption("All acts currently stored in the vector database.")

    with st.spinner("Fetching acts…"):
        resp = api_get("/api/acts")

    if resp is not None:
        if not resp:
            st.info("No acts are currently indexed. Run the ingestion pipeline first.")
        else:
            total_sections = sum(a.get("section_count", 0) for a in resp)
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(f'<div class="metric-card"><div class="label">Total Acts</div><div class="value">{len(resp)}</div></div>', unsafe_allow_html=True)
            with mcol2:
                st.markdown(f'<div class="metric-card"><div class="label">Total Sections</div><div class="value">{total_sections}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            for act in resp:
                year_str = f" ({act['year']})" if act.get("year") else ""
                st.markdown(
                    f'<div class="act-row">'
                    f'<span class="act-name">📕 {act["name"]}{year_str}</span>'
                    f'<span class="act-meta">{act.get("section_count", 0)} sections</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════
# PAGE 5: System Health
# ══════════════════════════════════════════════════════════

elif page == "🏥 System Health":
    st.markdown("### 🏥 System Health")
    st.caption("Real-time status of the backend services.")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    with st.spinner("Checking health…"):
        resp = api_get("/health")

    if resp:
        status = resp.get("status", "unknown")
        badge_cls = "badge-ok" if status == "ok" else "badge-degraded"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card"><div class="label">Status</div>'
                f'<div class="value"><span class="badge {badge_cls}" style="font-size:1.1rem;">{status.upper()}</span></div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            pc = "✅ Connected" if resp.get("pinecone_connected") else "❌ Disconnected"
            st.markdown(f'<div class="metric-card"><div class="label">Pinecone</div><div class="value" style="font-size:1rem;">{pc}</div></div>', unsafe_allow_html=True)
        with col3:
            vc = resp.get("vector_count", 0)
            st.markdown(f'<div class="metric-card"><div class="label">Vectors Indexed</div><div class="value">{vc:,}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="metric-card" style="margin-top:1rem;"><div class="label">Index Name</div><div class="value" style="font-size:1.1rem;">{resp.get("index_name", "N/A")}</div></div>', unsafe_allow_html=True)
