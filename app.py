import streamlit as st
from rag_pipeline import (
    load_faiss,
    load_tinyllama,
    load_gemini,
    search_faiss,
    build_context,
    generate_with_tinyllama,
    generate_with_gemini,
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot — TinyLlama & Gemini")
st.write("Ask questions based on your uploaded research PDFs using Retrieval-Augmented Generation (RAG).")

# ---------------- Load FAISS and Models ----------------
@st.cache_resource
def init_system():
    index, metadata, documents, embedder = load_faiss()
    tinyllama = load_tinyllama()
    gemini = load_gemini()
    return index, metadata, documents, embedder, tinyllama, gemini


try:
    index, metadata, documents, embedder, tinyllama, gemini = init_system()
except Exception as e:
    st.error(f"❌ System initialization failed: {e}")
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Select model:", ["TinyLlama", "Gemini"])
top_k = st.sidebar.slider("Number of retrieved chunks (Top-K)", 1, 10, 3)
show_context = st.sidebar.checkbox("Show retrieved context (for debugging)", False)

# ---------------- Session State ----------------
if "answers" not in st.session_state:
    st.session_state.answers = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "attempt" not in st.session_state:
    st.session_state.attempt = 0


# ---------------- Helper: Generate Answer ----------------
def generate_answer(query, attempt=1):
    """Generate an answer based on the query and attempt number."""
    results = search_faiss(query, embedder, index, metadata, documents, top_k)
    context = build_context(results)

    if model_choice == "TinyLlama":
        answer = generate_with_tinyllama(query, tinyllama, context, version=attempt)
    else:
        answer = generate_with_gemini(query, gemini, context, version=attempt)

    return {
        "answer": answer,
        "context": context,
        "sources": list({r["meta"]["source"] for r in results}),
    }


# ---------------- Main UI ----------------
query = st.text_area("Enter your question:", height=100)

if st.button("🔍 Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.query = query
    st.session_state.answers = []
    st.session_state.attempt = 1

    with st.spinner(f"Generating explanation #{st.session_state.attempt} using {model_choice}..."):
        answer = generate_answer(query, attempt=st.session_state.attempt)
        st.session_state.answers.append(answer)

    st.rerun()

# ---------------- Display Answers ----------------
if st.session_state.answers:
    for i, ans in enumerate(st.session_state.answers, start=1):
        st.markdown(f"### 💡 Explanation #{i}")
        st.success(ans["answer"])

        if show_context:
            with st.expander(f"🧩 View Retrieved Context for Explanation #{i}", expanded=False):
                st.write(ans["context"][:2000] + "..." if len(ans["context"]) > 2000 else ans["context"])

        st.markdown("### 📚 Sources")
        for src in ans["sources"]:
            st.write(f"- {src}")

        st.markdown("---")

    # "Didn't understand" button (max 3 attempts)
    if st.session_state.attempt < 3:
        if st.button("😕 I didn’t understand — show me a different explanation"):
            st.session_state.attempt += 1
            with st.spinner(f"Generating a new explanation (#{st.session_state.attempt})..."):
                new_answer = generate_answer(st.session_state.query, st.session_state.attempt)
                st.session_state.answers.append(new_answer)
            st.rerun()
    else:
        st.info("✅ You’ve reached the maximum of 3 explanations for this question.")

st.markdown("---")
st.caption("Built with TinyLlama 🦙 + Gemini 🌟 + FAISS 🔍")
