from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from app.agent.agent import LammAgent
from app.core.config import Settings, get_settings
from app.evaluation.report import write_markdown_report
from app.evaluation.runner import EvaluationRunner

# Page Configuration
st.set_page_config(
    page_title="LAMM: Long-Term Memory Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_agent(
    relevance_w: float,
    recency_w: float,
    confidence_w: float,
    redundancy_w: float,
    utility_w: float,
    red_thresh: float,
    arch_thresh: float,
    forg_thresh: float,
    top_k: int,
) -> LammAgent:
    base_settings = get_settings()
    custom_settings = Settings(
        gemini_api_key=base_settings.gemini_api_key,
        gemini_model=base_settings.gemini_model,
        embedding_provider=base_settings.embedding_provider,
        embedding_model=base_settings.embedding_model,
        embedding_dimension=base_settings.embedding_dimension,
        database_url=base_settings.database_url,
        faiss_index_path=base_settings.faiss_index_path,
        top_k=top_k,
        relevance_weight=relevance_w,
        recency_weight=recency_w,
        confidence_weight=confidence_w,
        redundancy_weight=redundancy_w,
        utility_weight=utility_w,
        redundancy_threshold=red_thresh,
        archive_threshold=arch_thresh,
        forget_threshold=forg_thresh,
    )
    return LammAgent(custom_settings)


# -------------------------------------------------------------
# SIDEBAR: Configuration, Weights, and Operational Status
# -------------------------------------------------------------
st.sidebar.title("🧠 LAMM Controller")
st.sidebar.caption("Lightweight Adaptive Memory Management")

base_cfg = get_settings()
status_color = "🟢" if base_cfg.gemini_available else "🟡"
mode_name = "Gemini LLM (Online)" if base_cfg.gemini_available else "Deterministic Mock (Offline)"
st.sidebar.markdown(f"**Mode:** {status_color} {mode_name}")
st.sidebar.markdown(f"**Embedding:** `{base_cfg.embedding_provider}` (`{base_cfg.embedding_model}`)")

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval & Lifecycle Parameters")
top_k = st.sidebar.slider("Top-K Retrieval", min_value=1, max_value=10, value=base_cfg.top_k)

st.sidebar.markdown("**Scoring Weights (Heuristic Composite)**")
w_rel = st.sidebar.slider("Relevance Weight", 0.0, 1.0, base_cfg.relevance_weight, 0.05)
w_rec = st.sidebar.slider("Recency Weight", 0.0, 1.0, base_cfg.recency_weight, 0.05)
w_conf = st.sidebar.slider("Confidence Weight", 0.0, 1.0, base_cfg.confidence_weight, 0.05)
w_red = st.sidebar.slider("Redundancy Weight", 0.0, 1.0, base_cfg.redundancy_weight, 0.05)
w_util = st.sidebar.slider("Utility Weight", 0.0, 1.0, base_cfg.utility_weight, 0.05)

st.sidebar.markdown("**Lifecycle Thresholds**")
t_red = st.sidebar.slider("Redundancy Threshold", 0.1, 1.0, base_cfg.redundancy_threshold, 0.02)
t_arch = st.sidebar.slider("Archive Threshold", 0.1, 0.9, base_cfg.archive_threshold, 0.02)
t_forg = st.sidebar.slider("Forget Threshold", 0.05, t_arch, min(base_cfg.forget_threshold, t_arch), 0.01)

agent = load_agent(w_rel, w_rec, w_conf, w_red, w_util, t_red, t_arch, t_forg, top_k)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Rebuild Vector Index", use_container_width=True):
    agent.manager.rebuild_index()
    st.sidebar.success(f"Index synchronized ({len(agent.manager.vector_store.ids)} active vectors).")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["💬 Agent Chat", "🗄️ Memory Store", "📊 Lifecycle Analytics", "🔬 Comparative Evaluation"],
)

# -------------------------------------------------------------
# PAGE 1: AGENT CHAT
# -------------------------------------------------------------
if page == "💬 Agent Chat":
    st.title("💬 LAMM Conversational Agent")
    st.markdown(
        "Interact with the agent. Incoming statements are parsed into candidate facts, "
        "scored via the **LAMM External Controller**, and governed across the 6 lifecycle operations: "
        "`Retain`, `Update`, `Merge`, `Compress`, `Archive`, `Forget`."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_retrieved" not in st.session_state:
        st.session_state.latest_retrieved = []
    if "latest_decisions" not in st.session_state:
        st.session_state.latest_decisions = []
    if "latest_stats" not in st.session_state:
        st.session_state.latest_stats = {}

    # Display prior turns
    for turn in st.session_state.messages:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    # Chat input
    user_input = st.chat_input("Enter your message (e.g. 'I prefer Python for backend development')...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing memory retrieval & lifecycle management..."):
                resp = agent.chat(user_input, conversation_id="web_session")
                st.write(resp.response)

                st.session_state.messages.append({"role": "assistant", "content": resp.response})
                st.session_state.latest_retrieved = resp.retrieved_memories
                st.session_state.latest_decisions = resp.lifecycle_decisions
                st.session_state.latest_stats = resp.stats

    # Inspection Panels for Most Recent Turn
    if st.session_state.latest_retrieved or st.session_state.latest_decisions:
        st.markdown("---")
        st.subheader("🔍 Turn Inspection: Memory & Lifecycle Decisions")

        col1, col2 = st.columns(2)
        with col1:
            with st.expander(f"📥 Relevant Retrieved Memories ({len(st.session_state.latest_retrieved)})", expanded=True):
                if st.session_state.latest_retrieved:
                    for item in st.session_state.latest_retrieved:
                        st.markdown(f"- **Text:** {item['text']}")
                        st.caption(f"Relevance Score: `{item['score']:.3f}` | ID: `{item['id']}`")
                else:
                    st.info("No relevant long-term memories retrieved for this query.")

        with col2:
            with st.expander(f"⚙️ Newly Extracted Facts & Decisions ({len(st.session_state.latest_decisions)})", expanded=True):
                if st.session_state.latest_decisions:
                    for d in st.session_state.latest_decisions:
                        op = d["operation"]
                        badge_color = {
                            "RETAIN": "green", "UPDATE": "blue", "MERGE": "purple",
                            "COMPRESS": "orange", "ARCHIVE": "gray", "FORGET": "red"
                        }.get(op, "blue")
                        st.markdown(f":{badge_color}[**{op}**] - {d['reason']}")
                        scores = d.get("scores", {})
                        st.caption(
                            f"Scores: Overall={scores.get('overall', 0):.2f} | "
                            f"Redundancy={scores.get('redundancy', 0):.2f} | "
                            f"Confidence={scores.get('confidence', 0):.2f} | "
                            f"Utility={scores.get('utility', 0):.2f}"
                        )
                else:
                    st.info("No candidate facts extracted from this turn.")

        if st.session_state.latest_stats:
            st.caption(
                f"⏱️ Retrieval Latency: `{st.session_state.latest_stats.get('retrieval_latency_seconds', 0)*1000:.1f}ms` | "
                f"Context Tokens: `{st.session_state.latest_stats.get('context_tokens', 'N/A')}` "
                f"({st.session_state.latest_stats.get('token_count_method', 'estimated')}) | "
                f"Active Memories in Store: `{st.session_state.latest_stats.get('active', 0)}`"
            )

# -------------------------------------------------------------
# PAGE 2: MEMORY STORE
# -------------------------------------------------------------
elif page == "🗄️ Memory Store":
    st.title("🗄️ Canonical Memory Store")
    st.markdown(
        "**SQLite is the canonical source of truth** for all memories and metadata. "
        "The **FAISS vector index is a derived index** holding vectors solely for `ACTIVE` memories."
    )

    stats = agent.manager.stats()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Total Stored", stats["total"])
    m_col2.metric("Active (Retrievable)", stats["active"])
    m_col3.metric("Archived", stats["archived"])
    m_col4.metric("Forgotten", stats["forgotten"])
    m_col5.metric("FAISS Index Count", stats["faiss_indexed_count"])

    st.markdown("---")
    st.subheader("Semantic Query Test")
    test_q = st.text_input("Enter a query to test live FAISS semantic retrieval:", "backend project programming language")
    if test_q:
        results = agent.manager.retrieve(test_q, top_k=5)
        if results:
            for rank, item in enumerate(results, start=1):
                st.markdown(
                    f"**{rank}.** {item.memory.text}  \n"
                    f"*(Similarity: `{item.score:.3f}` | Status: `{item.memory.lifecycle_status.value}` | "
                    f"Access Count: `{item.memory.access_count}` | ID: `{item.memory.id}`)*"
                )
        else:
            st.info("No active memories matched the query.")

    st.markdown("---")
    st.subheader("Stored Memory Records")
    filter_status = st.selectbox("Filter by Status", ["All", "ACTIVE", "ARCHIVED", "FORGOTTEN"])
    all_records = agent.manager.memories.list(include_inactive=True)

    if filter_status != "All":
        filtered = [m for m in all_records if m.lifecycle_status.value == filter_status]
    else:
        filtered = all_records

    if filtered:
        table_data = [
            {
                "ID": m.id[:8] + "...",
                "Full ID": m.id,
                "Text": m.text,
                "Status": m.lifecycle_status.value,
                "Confidence": round(m.confidence_score, 2),
                "Utility": round(m.utility_score, 2),
                "Recency Score": round(m.recency_score, 2),
                "Access Count": m.access_count,
                "Updated At": m.updated_at.strftime("%Y-%m-%d %H:%M:%S") if m.updated_at else "",
            }
            for m in filtered
        ]
        df = pd.DataFrame(table_data)
        st.dataframe(df.drop(columns=["Full ID"]), use_container_width=True)

        st.markdown("---")
        st.subheader("Explicit Privacy Deletion (User Request)")
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            del_id = st.selectbox("Select Memory to Forget:", [row["Full ID"] for row in table_data])
        with col_del2:
            st.write("")
            st.write("")
            if st.button("🗑️ Forget Memory", use_container_width=True):
                if agent.manager.forget(del_id):
                    st.success(f"Memory '{del_id[:8]}' removed from active FAISS and marked FORGOTTEN.")
                    st.rerun()
    else:
        st.info("No memory records found.")

# -------------------------------------------------------------
# PAGE 3: LIFECYCLE ANALYTICS
# -------------------------------------------------------------
elif page == "📊 Lifecycle Analytics":
    st.title("📊 Memory Lifecycle Analytics & Audit Log")
    st.markdown(
        "Every lifecycle decision made by LAMM is permanently recorded with full before/after state snapshots, "
        "score breakdowns, and human-interpretable reasoning."
    )

    events = agent.manager.events.list()
    if events:
        event_df = pd.DataFrame(events)
        ops_count = event_df["operation"].value_counts()

        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("Lifecycle Decisions Distribution")
            st.bar_chart(ops_count)

        with col2:
            st.subheader("Operations Summary")
            st.dataframe(
                pd.DataFrame({"Operation": ops_count.index, "Count": ops_count.values}),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Immutable Lifecycle Audit Trail")
        display_events = []
        for e in reversed(events):
            scores = json.loads(e.get("scores", "{}")) if isinstance(e.get("scores"), str) else e.get("scores", {})
            display_events.append({
                "Timestamp": e.get("timestamp", "")[:19].replace("T", " "),
                "Operation": e.get("operation"),
                "Reason": e.get("reason"),
                "Overall Score": round(scores.get("overall", 0.0), 2),
                "Redundancy": round(scores.get("redundancy", 0.0), 2),
                "Confidence": round(scores.get("confidence", 0.0), 2),
                "Memory ID": (e.get("memory_id") or "")[:8] + "...",
            })
        st.dataframe(pd.DataFrame(display_events), use_container_width=True)
    else:
        st.info("No lifecycle events recorded yet. Interact in Agent Chat to generate events.")

# -------------------------------------------------------------
# PAGE 4: COMPARATIVE EVALUATION
# -------------------------------------------------------------
else:
    st.title("🔬 Comparative Evaluation: LAMM vs Baselines")
    st.markdown(
        "Conducts a fair 3-way experimental comparison on identical synthetic scenario turns:  \n"
        "1. **LAMM**: Adaptive lifecycle management (relevance, recency, confidence, redundancy, utility)  \n"
        "2. **UNMANAGED_MEMORY**: Unconditional persistence without eviction (unbounded growth)  \n"
        "3. **RECENCY_ONLY**: Fixed LRU budget memory eviction without semantic signals"
    )

    if st.button("🚀 Run 3-Way Comparative Experiment", type="primary"):
        with st.spinner("Executing strictly identical multi-turn benchmark across all 3 systems..."):
            runner = EvaluationRunner()
            res = runner.run()
            write_markdown_report()
            st.success("Experiment completed! Figures and tables saved in results/.")

    # Load and display latest results if present
    report_json_path = Path("results/reports/evaluation_report.json")
    if report_json_path.exists():
        data = json.loads(report_json_path.read_text(encoding="utf-8"))
        st.markdown("---")
        st.subheader(f"Latest Experiment Results ({data.get('experiment_timestamp', 'Recent')})")

        comp = data.get("comparative_metrics", {})
        mem_red = comp.get("measured_active_memory_reduction_vs_unmanaged_pct", 0.0)
        tok_red = comp.get("measured_context_token_reduction_vs_unmanaged_pct", 0.0)

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Active Memory Reduction", f"{mem_red}%", help="vs UNMANAGED_MEMORY baseline")
        kpi2.metric("Prompt Token Savings", f"{tok_red}%", help="vs UNMANAGED_MEMORY baseline")
        kpi3.metric("LAMM Active vs Total", f"{data['final_memory_counts']['LAMM']['active']} / {data['final_memory_counts']['LAMM']['total']}")

        st.markdown("### Summary Performance Comparison")
        summary_csv = Path("results/tables/metrics_summary.csv")
        if summary_csv.exists():
            st.dataframe(pd.read_csv(summary_csv), use_container_width=True)

        st.markdown("### Generated Research Figures")
        fig_cols = st.columns(2)
        figs = [
            ("results/figures/memory_growth.png", "1. Total Memory Growth Over Turns"),
            ("results/figures/active_memory_count.png", "2. Active Memory Count Over Turns"),
            ("results/figures/token_context_consumption.png", "3. Context Token Load Over Turns"),
            ("results/figures/retrieval_latency.png", "4. Semantic Retrieval Latency Comparison"),
            ("results/figures/lifecycle_operations.png", "5. LAMM Lifecycle Decisions Distribution"),
            ("results/figures/retrieval_quality.png", "6. Retrieval Quality (Precision@k, Recall@k, MRR)"),
        ]
        for idx, (path_str, title) in enumerate(figs):
            p = Path(path_str)
            col = fig_cols[idx % 2]
            with col:
                if p.exists():
                    st.image(str(p), caption=title, use_container_width=True)
                else:
                    st.warning(f"Figure not found: {path_str}")

        # Show Markdown Report
        md_path = Path("results/reports/evaluation_report.md")
        if md_path.exists():
            with st.expander("📄 View Full Markdown Research Report", expanded=False):
                st.markdown(md_path.read_text(encoding="utf-8"))
    else:
        st.info("Click 'Run 3-Way Comparative Experiment' to generate experimental comparison data and plots.")
