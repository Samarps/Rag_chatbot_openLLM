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
    INDEX_FILE,
    META_FILE,
    DOCS_FILE,
)

# -------------------- Load FAISS --------------------
def load_faiss():
    print("📥 Loading FAISS index...")
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("❌ FAISS index not found. Run 'ingest.py' first.")

    index = faiss.read_index(INDEX_FILE)
    metadata = np.load(META_FILE, allow_pickle=True)
    documents = np.load(DOCS_FILE, allow_pickle=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"✅ Loaded FAISS index with {len(metadata)} chunks.")
    return index, metadata, documents, embedder


# -------------------- Search --------------------
def search_faiss(query, embedder, index, metadata, documents, top_k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx].item() if isinstance(metadata[idx], np.ndarray) else metadata[idx]
        text = documents[idx].item() if isinstance(documents[idx], np.ndarray) else documents[idx]
        results.append({"meta": meta, "text": text})
    return results


# -------------------- Load Models --------------------
def load_tinyllama():
    print(f"🦙 Loading TinyLlama on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINYLLAMA_MODEL, torch_dtype=torch.float32)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    print("✅ TinyLlama loaded.")
    return gen_pipe


def load_gemini():
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API client ready.")
    return client


# -------------------- Context Builder --------------------
def build_context(results):
    chunks = [f"From {r['meta']['source']}:\n{r['text']}" for r in results]
    return "\n\n".join(chunks[:3])


# -------------------- Generate Responses --------------------
def generate_with_tinyllama(query, gen_pipe, context, version=1):
    styles = {
        1: "Answer clearly and concisely based on the context.",
        2: "Re-explain the same concept differently, using simpler terms and analogies to help a student understand.",
        3: "Provide another distinct explanation focusing on intuitive understanding, reasoning, or step-by-step logic.",
    }

    tokenizer = gen_pipe.tokenizer
    encoded = tokenizer(context, truncation=True, max_length=1400)
    truncated_context = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    prompt = f"""
You are a helpful research assistant.
{styles.get(version, styles[1])}

Context:
{truncated_context}

Question: {query}

Answer:
"""
    response = gen_pipe(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9)
    return response[0]["generated_text"].split("Answer:")[-1].strip()


def generate_with_gemini(query, client, context, version=1):
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
