import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from google import genai
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TINYLLAMA_MODEL,
    EMBED_MODEL,
    DEVICE,
)

# -------------------- Paths --------------------
INDEX_PATH = "faiss_index"
INDEX_FILE = os.path.join(INDEX_PATH, "docs.index")
META_FILE = os.path.join(INDEX_PATH, "metadata.npy")
DOCS_FILE = os.path.join(INDEX_PATH, "documents.npy")


# -------------------- Load FAISS --------------------
def load_faiss():
    print("Loading FAISS index...")
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("❌ FAISS index not found. Run 'ingest.py' first.")

    index = faiss.read_index(INDEX_FILE)
    metadata = np.load(META_FILE, allow_pickle=True)
    documents = np.load(DOCS_FILE, allow_pickle=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"✅ Loaded FAISS index with {len(metadata)} chunks.")
    return index, metadata, documents, embedder


# -------------------- Search FAISS --------------------
def search_faiss(query, embedder, index, metadata, documents, top_k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx].item() if isinstance(metadata[idx], np.ndarray) else metadata[idx]
        text = documents[idx].item() if isinstance(documents[idx], np.ndarray) else documents[idx]
        results.append({"meta": meta, "text": text})
    return results


# -------------------- Load TinyLlama --------------------
def load_tinyllama():
    print(f"Loading TinyLlama model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
        max_new_tokens=256,
        temperature=0.5,  # slightly higher for variety
        do_sample=True,
    )
    print("✅ TinyLlama loaded successfully.")
    return gen_pipe


# -------------------- Load Gemini --------------------
def load_gemini():
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API client ready.")
    return client


# -------------------- Build Context --------------------
def build_context(results):
    chunks = []
    for r in results:
        src = r["meta"]["source"]
        text = r["text"]
        chunks.append(f"From {src}:\n{text}")
    return "\n\n".join(chunks[:3])


# -------------------- Generate: TinyLlama --------------------
def generate_with_tinyllama(query, gen_pipe, context, version=1):
    """Generate response with TinyLlama. version controls explanation style (1–3)."""
    styles = {
        1: "Answer clearly and concisely based on the context.",
        2: "Re-explain the same concept differently, using simpler terms and analogies to help a student understand.",
        3: "Provide another distinct explanation focusing on intuitive understanding, reasoning, or step-by-step logic.",
    }

    prompt = f"""
You are a helpful research assistant.
{styles.get(version, styles[1])}

Context:
{context}

Question: {query}

Answer:
"""
    response = gen_pipe(prompt, pad_token_id=gen_pipe.tokenizer.eos_token_id)
    return response[0]["generated_text"].split("Answer:")[-1].strip()


# -------------------- Generate: Gemini --------------------
def generate_with_gemini(query, client, context, version=1):
    """Generate response with Gemini. version controls explanation style (1–3)."""
    styles = {
        1: "Answer clearly and concisely in an academic tone.",
        2: "Re-explain the same idea using simpler words, examples, or analogies suitable for a student.",
        3: "Give another unique explanation focusing on conceptual understanding, intuition, or real-world reasoning.",
    }

    prompt = f"""
You are a helpful research assistant. {styles.get(version, styles[1])}

Context:
{context}

Question: {query}

Answer:
"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


# -------------------- CLI MODE (for debugging) --------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    index, metadata, documents, embedder = load_faiss()
    tinyllama = load_tinyllama()
    gemini = load_gemini()

    print("\nSystem ready! Type your query below.")
    print("Choose model: [1 = TinyLlama | 2 = Gemini]\n")

    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        mode = input("Choose model [1=TinyLlama, 2=Gemini]: ").strip()
        version = int(input("Explanation version [1–3]: ").strip() or 1)

        results = search_faiss(query, embedder, index, metadata, documents)
        context = build_context(results)

        if mode == "1":
            answer = generate_with_tinyllama(query, tinyllama, context, version)
        elif mode == "2":
            answer = generate_with_gemini(query, gemini, context, version)
        else:
            print("Invalid choice.")
            continue

        print("\n" + "=" * 50)
        print("Answer:\n", answer)
        print("=" * 50)
