import os
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
from ingest import ingest_pdfs

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot — TinyLlama & Gemini")
st.write("Ask questions based on your uploaded research PDFs using Retrieval-Augmented Generation (RAG).")

DATA_DIR = "data"
INDEX_PATH = "faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Select model:", ["TinyLlama", "Gemini"])
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 3)
show_context = st.sidebar.checkbox("Show retrieved context", False)
st.sidebar.divider()

# ---------------- PDF Management ----------------
st.sidebar.subheader("📚 Indexed PDFs")
existing_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if existing_pdfs:
    for pdf in existing_pdfs:
        st.sidebar.write(f"• {pdf}")
else:
    st.sidebar.info("No PDFs currently indexed.")

uploaded_files = st.sidebar.file_uploader("📤 Upload PDFs", type=["pdf"], accept_multiple_files=True)

if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

if uploaded_files:
    new_files = [f.name for f in uploaded_files if f.name not in st.session_state.last_uploaded_files]
    if new_files:
        with st.spinner("📚 Processing uploaded PDFs..."):
            for file in uploaded_files:
                with open(os.path.join(DATA_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())
            ingest_pdfs(DATA_DIR)
            st.session_state.last_uploaded_files = [f.name for f in uploaded_files]
            st.success("✅ PDFs ingested successfully!")
            st.rerun()
    else:
        st.sidebar.info("ℹ️ These PDFs are already indexed.")

if st.sidebar.button("🧹 Clear all indexed PDFs"):
    for folder in [DATA_DIR, INDEX_PATH]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    st.session_state.last_uploaded_files = []
    st.sidebar.success("✅ All indexed PDFs cleared.")
    st.rerun()

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
    st.error(f"❌ Initialization failed: {e}")
    st.stop()

# ---------------- Session State ----------------
if "answers" not in st.session_state:
    st.session_state.answers = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "attempt" not in st.session_state:
    st.session_state.attempt = 0

# ---------------- Answer Generation ----------------
def generate_answer(query, attempt=1):
    results = search_faiss(query, embedder, index, metadata, documents, top_k)
    context = build_context(results)
    context_variant = context
    if attempt > 1:
        context_variant += f"\n\n(Note: Provide a different phrasing or reasoning approach. Attempt #{attempt})"

    if model_choice == "TinyLlama":
        answer = generate_with_tinyllama(query, tinyllama, context_variant, attempt)
    else:
        answer = generate_with_gemini(query, gemini, context_variant, attempt)

    return {"answer": answer, "context": context, "sources": list({r['meta']['source'] for r in results})}

# ---------------- Main UI ----------------
query = st.text_area("Enter your question:", height=100)

if st.button("🔍 Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.query = query
    st.session_state.answers = []
    st.session_state.attempt = 1

    with st.spinner(f"Generating answer using {model_choice}..."):
        answer = generate_answer(query, attempt=1)
        st.session_state.answers.append(answer)
    st.rerun()

# ---------------- Display Answers ----------------
if st.session_state.answers:
    for i, ans in enumerate(st.session_state.answers, start=1):
        st.markdown("### 💡 Answer" if i == 1 else "### 💬 Alternative explanation")
        st.success(ans["answer"])

        if show_context:
            with st.expander("🧩 Retrieved Context", expanded=False):
                st.write(ans["context"][:2000] + "..." if len(ans["context"]) > 2000 else ans["context"])

        st.markdown("### 📚 Sources")
        for src in ans["sources"]:
            st.write(f"- {src}")
        st.markdown("---")

    if st.session_state.attempt < 3:
        if st.button("😕 I didn’t understand — show me another explanation"):
            st.session_state.attempt += 1
            with st.spinner("Generating alternative explanation..."):
                new_ans = generate_answer(st.session_state.query, st.session_state.attempt)
                st.session_state.answers.append(new_ans)
            st.rerun()
    else:
        st.info("✅ You’ve reached the maximum of 3 explanations.")

st.markdown("---")
st.caption("Built with 🦙 TinyLlama + 🌟 Gemini + 🔍 FAISS")
