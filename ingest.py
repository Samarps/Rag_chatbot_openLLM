import os
import faiss
import pdfplumber
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

# ---------------- CONFIG ----------------
DATA_DIR = "data"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "docs.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.npy")
DOCS_FILE = os.path.join(INDEX_DIR, "documents.npy")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------------- TEXT CHUNKING ----------------
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks for more coherent retrieval.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdfs(pdf_dir):
    """
    Extract text from all PDFs in the data directory using pdfplumber.
    """
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("⚠️ No PDF files found in the 'data/' folder.")

    all_docs = []
    for pdf in pdf_files:
        file_path = os.path.join(pdf_dir, pdf)
        print(f"📄 Extracting text from: {pdf}")
        full_text = ""

        try:
            with pdfplumber.open(file_path) as pdf_reader:
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    page_text = page_text.replace("\n", " ").strip()
                    if page_text:
                        full_text += page_text + " "
        except Exception as e:
            print(f"⚠️ Error reading {pdf}: {e}")
            continue

        if full_text.strip():
            all_docs.append((pdf, full_text))
        else:
            print(f"⚠️ No extractable text found in: {pdf}")

    print(f"✅ Extracted text from {len(all_docs)} PDFs.")
    return all_docs


# ---------------- BUILD FAISS INDEX ----------------
def build_faiss_index(docs, embedder, chunk_size=500, overlap=100):
    """
    Create and save FAISS index, metadata, and document chunks.
    """
    all_chunks = []
    all_meta = []

    for pdf_name, text in docs:
        chunks = chunk_text(text, chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_meta.append({"source": pdf_name})

    print(f"🧩 Total chunks to embed: {len(all_chunks)}")

    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="🔍 Generating embeddings"):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(META_FILE, np.array(all_meta, dtype=object))
    np.save(DOCS_FILE, np.array(all_chunks, dtype=object))

    print("✅ FAISS index built and saved successfully.")
    print(f"   → Saved index: {INDEX_FILE}")
    print(f"   → Metadata entries: {len(all_meta)}")


# ---------------- MAIN FUNCTION ----------------
def ingest_pdfs(pdf_dir=DATA_DIR):
    """
    Main pipeline: extract text → chunk → embed → build FAISS → save.
    """
    print("\nStarting ingestion pipeline...\n")

    docs = extract_text_from_pdfs(pdf_dir)
    embedder = SentenceTransformer(EMBED_MODEL)
    build_faiss_index(docs, embedder)

    print("\nIngestion complete! Your FAISS index is ready.\n")


# ---------------- CLI ENTRYPOINT ----------------
if __name__ == "__main__":
    ingest_pdfs()
