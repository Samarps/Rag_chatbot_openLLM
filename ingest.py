import os
import faiss
import numpy as np
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Path where PDFs will be stored (upload folder)
DATA_PATH = "data"
# Path where FAISS index will be saved
INDEX_PATH = "faiss_index"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

# Load embedding model (MiniLM is light and works great)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_path}: {e}")
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for better context retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_faiss_index(pdf_folder=DATA_PATH):
    """Extract, chunk, embed, and store all PDF data into a FAISS index."""
    documents = []
    metadata = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in 'data/' folder.")
        return None, None, None

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
    for file_name in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder, file_name)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        for chunk in chunks:
            documents.append(chunk)
            metadata.append({"source": file_name})

    print("Generating embeddings...")
    embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, os.path.join(INDEX_PATH, "docs.index"))
    np.save(os.path.join(INDEX_PATH, "metadata.npy"), metadata)
    np.save(os.path.join(INDEX_PATH, "documents.npy"), documents)

    print(f"Ingestion complete! {len(documents)} chunks stored in FAISS.")
    return index, documents, metadata


if __name__ == "__main__":
    build_faiss_index()
