import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm
import os

# --- 1. Configuration & File Paths ---
# Use CPU for the embedding model to avoid potential memory conflicts with SLM training on GPU
device = "cpu"
torch.set_default_device(device)

# Define the local file paths
TRAIN_FILE_PATH = r"C:\Users\anilsagar\Desktop\SLM_rag_project\wikitext-2-raw\wiki.train.raw" # Your local validation file from Kaggle
INDEX_FILE_PATH = "wikitext_faiss_index.bin"
CORPUS_FILE_PATH = "wikitext_corpus_chunks.npy"


# --- 2. Load Knowledge Corpus (Using Local File) ---
# CHANGE 2: Loading the training file instead of the validation file
print(f"Loading local RAG corpus from {TRAIN_FILE_PATH}...")

ds = load_dataset(
    "text",
    data_files=TRAIN_FILE_PATH,  
)

# Access the first (and only) split from the loaded local file
train_split = ds[list(ds.keys())[0]] # Renamed for clarity

# Filter out short/empty lines to create clean, meaningful chunks
corpus = [text.strip() for text in train_split['text'] if len(text.strip()) > 50]
print(f"Corpus loaded. Total chunks: {len(corpus)}")

# --- 3. Load Embedding Model ---
# 'all-MiniLM-L6-v2' is a lightweight yet effective model for semantic search (384 dimensions)
print("Loading Sentence Transformer model on CPU...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- 4. Create Embeddings ---
print("Creating embeddings for the corpus...")
# Embed the entire corpus. Output is a NumPy array.
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
dimension = corpus_embeddings.shape[1] # Should be 384

# --- 5. Create and Save FAISS Index ---
# IndexFlatL2 uses standard Euclidean distance for similarity search
index = faiss.IndexFlatL2(dimension)
# Add the embeddings (vectors) to the index. FAISS requires float32.
index.add(corpus_embeddings.astype(np.float32))

# Save the index and the raw text chunks (corpus)
faiss.write_index(index, "wikitext_faiss_index.bin")
np.save("wikitext_corpus_chunks.npy", np.array(corpus))

print("\nIndexing complete.")
print(f"FAISS index saved to wikitext_faiss_index.bin (Dimension: {dimension})")
print(f"Raw corpus chunks saved to wikitext_corpus_chunks.npy (Total {len(corpus)} chunks)")
