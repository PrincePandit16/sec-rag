import streamlit as st
import requests
import pandas as pd

# API Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SEC Filings RAG Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SEC Filings RAG Dashboard")
st.markdown("Query professional financial reports with AI-powered precision.")

# Sidebar for Filters & Admin
with st.sidebar:
    st.header("Search Filters")
    company = st.text_input("Company (optional)", placeholder="e.g. apple")
    year = st.number_input("Year (optional)", min_value=2000, max_value=2026, value=2024, step=1)
    form = st.selectbox("Form", options=["Any", "10-K", "10-Q"])

    st.divider()

    st.header("Settings")
    agentic = st.toggle("Enable Agentic Search", value=False, help="Slower but more accurate. Uses a self-correction loop.")
    top_k = st.slider("Number of Source Chunks", 1, 20, 10)

    st.divider()

    st.header("Admin")
    if st.button("🔄 Sync Database", use_container_width=True):
        with st.spinner("Triggering ingestion pipeline..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/upload")
                if resp.status_code == 200:
                    st.success("Ingestion triggered!")
                else:
                    st.error(f"Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Main Query Interface
query = st.text_input("What would you like to know about the filings?", placeholder="e.g. What was Apple's total revenue in 2023?")

if query:
    with st.spinner("Analyzing filings..."):
        try:
            # Prepare request payload
            payload = {
                "query": query,
                "company": company if company else None,
                "year": year if year != 2024 else None, # Simplified for this demo
                "form": form if form != "Any" else None,
                "agentic": agentic,
                "top_k": top_k
            }

            resp = requests.post(f"{API_BASE_URL}/query", json=payload)

            if resp.status_code == 200:
                data = resp.json()

                # AI Answer
                st.subheader("AI Answer")
                st.markdown(data["answer"] if data["answer"] else "No answer could be synthesized.")

                # Audit Report
                if data.get("audit_report"):
                    st.warning(f"**Citation Audit Warning:**\n\n{data['audit_report']}")

                # Source Chunks
                st.divider()
                st.subheader("Source Chunks")

                for i, source in enumerate(data["sources"], 1):
                    with st.expander(f"Chunk {i} | Score: {source.get('score', 'N/A'):.4f} | {source['metadata'].get('company', 'Unknown')}"):
                        st.markdown(source["document"])
                        st.json(source["metadata"])
            else:
                st.error(f"API Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# Health Status in Footer
try:
    health = requests.get(f"{API_BASE_URL}/health").json()
    st.caption(f"System Status: {health['status']} | Docs Indexed: {health['documents_count']:,}")
except:
    st.caption("System Status: API Offline")
