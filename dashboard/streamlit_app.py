from __future__ import annotations

import pandas as pd
import streamlit as st

from app.agent.agent import LammAgent
from app.core.config import get_settings


@st.cache_resource
def get_agent():
    return LammAgent()


settings = get_settings()
agent = get_agent()

st.set_page_config(page_title="LAMM Demo", layout="wide")
st.sidebar.title("LAMM")
st.sidebar.write("Gemini available:", settings.gemini_available)
st.sidebar.write("Embedding model:", settings.embedding_model)
st.sidebar.write("Top-k:", settings.top_k)
st.sidebar.write("Redundancy threshold:", settings.redundancy_threshold)
st.sidebar.write("Archive threshold:", settings.archive_threshold)
st.sidebar.write("Forget threshold:", settings.forget_threshold)

page = st.sidebar.radio("Page", ["Chat", "Memory", "Analytics", "Comparison"])

if page == "Chat":
    st.title("LAMM Chat")
    message = st.text_input("User message", "I prefer Python for backend development.")
    if st.button("Send"):
        result = agent.chat(message)
        st.subheader("Response")
        st.write(result.response)
        st.subheader("Retrieved memories")
        st.json(result.retrieved_memories)
        st.subheader("Extracted memories")
        st.json(result.extracted_memories)
        st.subheader("Lifecycle decisions")
        st.json(result.lifecycle_decisions)
        st.subheader("Stats")
        st.json(result.stats)
elif page == "Memory":
    st.title("Memory Store")
    memories = [m.model_dump(mode="json") for m in agent.manager.memories.list()]
    st.dataframe(pd.DataFrame(memories), use_container_width=True)
elif page == "Analytics":
    st.title("Analytics")
    st.json(agent.manager.stats())
    st.dataframe(pd.DataFrame(agent.manager.events.list()), use_container_width=True)
else:
    st.title("LAMM vs Unmanaged Baseline")
    st.write("Run `python scripts/run_evaluation.py` to generate comparison artifacts.")
    st.write("Results are saved in `results/tables` and `results/reports`.")
