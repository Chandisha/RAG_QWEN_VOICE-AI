import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

# --- 1. Configuration & File Paths ---
# The paths where the RAG files will be saved and the data will be read from.
COMPANY_DATA_FILE = r"C:\Users\user\OneDrive\Documents\RenataVoiceAgent\renata_data.json"
INDEX_FILE_PATH = "company_rag_index.bin"
CORPUS_FILE_PATH = "company_corpus_chunks.npy"
# The temporary file that holds the plain text to be indexed
KNOWLEDGE_BASE_TEXT_FILE = "knowledge_base.txt" 
device = "cpu"
torch.set_default_device(device)


def load_company_data():
    """Load the structured company data from JSON."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, COMPANY_DATA_FILE)

    if not os.path.exists(file_path):
        logger.error(f"❌ {COMPANY_DATA_FILE} not found at {file_path}")
        # NOTE: You will need to create a company_data.json file in the same directory!
        raise FileNotFoundError(f"❌ Required file {COMPANY_DATA_FILE} is missing. Please create it.")

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"❌ {COMPANY_DATA_FILE} is not valid JSON.")
        raise ValueError(f"❌ {COMPANY_DATA_FILE} is not valid JSON.")


def create_knowledge_base_text(data: dict) -> list[str]:
    """
    Formats the structured JSON data into a plain text document, 
    then chunks it by paragraph for the RAG system.
    
    Returns:
        list[str]: A list of clean text chunks (paragraphs).
    """
    logger.info("📄 Generating plain text for knowledge base...")

    company_name = data.get("company_name", "Renata AI")
    founder = data.get("co_founder", "an esteemed co-founder")
    mission = data.get("mission", "connecting the physical and digital world to unlock new operational efficiencies.")
    founded_year = data.get("founded_year", "2018")
    industry = data.get("industry", "Industrial IoT and Enterprise Automation")
    customers = ", ".join(data.get("customers", ["Global 500 Manufacturing Co.", "Major Logistics Firm"]))
    verticals = data.get("solution_verticals", [])

    # Start building the long-form text
    text_document = [
        f"Renata AI is a leading company in the {industry} sector, founded in {founded_year}. "
        f"The co-founder of Renata is {founder}.",

        f"The core mission of Renata AI is: {mission}. We are committed to pioneering "
        "the next generation of smart enterprise solutions by leveraging advanced AI and IoT technologies.",

        f"Renata AI offers a comprehensive suite of services and solutions across various verticals. "
        f"Our primary solution categories are: {', '.join([v['name'] for v in verticals])}.",
        
        f"A detailed list of the capabilities and services provided by Renata includes: "
        f"{', '.join([cap for v in verticals for cap in v['capabilities']])}.",
        
        "Renata AI is proud to partner with industry-leading clients, including: "
        f"{customers}. Our customer base spans manufacturing, logistics, smart cities, and energy sectors, "
        "demonstrating the versatility and impact of our IIoT platform."
    ]

    # Add detailed descriptions for each solution vertical as separate paragraphs
    for vertical in verticals:
        text_document.append(
            f"The '{vertical['name']}' vertical, which focuses on {vertical['description']}, offers key capabilities like: "
            f"{', '.join(vertical['capabilities'])}."
        )

    # Filter out empty or too short lines, and clean up the text
    corpus = [line.strip() for line in text_document if len(line.strip()) > 50]
    
    # Save the text to a file for review (optional, but good practice)
    with open(KNOWLEDGE_BASE_TEXT_FILE, "w") as f:
        f.write("\n\n---\n\n".join(corpus))
        
    logger.info(f"✅ Text knowledge base saved to {KNOWLEDGE_BASE_TEXT_FILE}.")
    return corpus


def create_faiss_index(corpus: list[str]):
    """Creates embeddings, builds a FAISS index, and saves the index and corpus."""
    logger.info("Loading Sentence Transformer model on CPU...")
    # 'all-MiniLM-L6-v2' is a lightweight yet effective model for semantic search (384 dimensions)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # --- Create Embeddings ---
    logger.info("Creating embeddings for the corpus...")
    # Embed the entire corpus. Output is a NumPy array.
    corpus_embeddings = embedder.encode(
        corpus, 
        convert_to_tensor=False, 
        show_progress_bar=True
    )
    dimension = corpus_embeddings.shape[1]  # Should be 384

    # --- Create and Save FAISS Index ---
    logger.info("Building FAISS index...")
    # IndexFlatL2 uses standard Euclidean distance for similarity search
    index = faiss.IndexFlatL2(dimension)
    # Add the embeddings (vectors) to the index. FAISS requires float32.
    index.add(corpus_embeddings.astype(np.float32))

    # Save the index and the raw text chunks (corpus)
    faiss.write_index(index, INDEX_FILE_PATH)
    np.save(CORPUS_FILE_PATH, np.array(corpus))

    logger.info("\nIndexing complete.")
    logger.info(f"FAISS index saved to {INDEX_FILE_PATH} (Dimension: {dimension})")
    logger.info(f"Raw corpus chunks saved to {CORPUS_FILE_PATH} (Total {len(corpus)} chunks)")


if __name__ == "__main__":
    try:
        # Load the structured data
        company_data = load_company_data()
        
        # Convert structured data into RAG-friendly text chunks
        corpus_chunks = create_knowledge_base_text(company_data)
        
        # Build the vector index
        create_faiss_index(corpus_chunks)
        
    except Exception as e:
        logger.error(f"An error occurred during index creation: {e}")