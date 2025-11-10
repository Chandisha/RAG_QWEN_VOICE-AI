import torch
import os
import numpy as np
import time
import uuid
import base64
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from typing import List, Optional

# Hugging Face Transformers and Sentence Transformers for RAG components
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# FAISS for vector search
from faiss import read_index, Index

# gTTS for Text-to-Speech (requires pip install gTTS)
from gtts import gTTS

# librosa for audio loading (requires FFmpeg to be on the system PATH)
import librosa

# --- Global Configuration and Initialization ---

# Check for CUDA availability and set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Server will attempt to run GPU-heavy models on: {DEVICE}")

# File Paths (MUST be correct for the server to start)
FAISS_INDEX_PATH = "renata_faiss_index.bin"
CORPUS_FILE_PATH = "renata_corpus_chunks.npy"

# Model Configurations
LLM_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
STT_MODEL_NAME = "openai/whisper-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global Variables (Initialized on startup)
global_index: Optional[Index] = None
global_embedder: Optional[SentenceTransformer] = None
global_llm_model: Optional[AutoModelForCausalLM] = None
global_llm_tokenizer: Optional[AutoTokenizer] = None
global_stt_model: Optional[WhisperForConditionalGeneration] = None
global_stt_processor: Optional[WhisperProcessor] = None
global_raw_chunks: Optional[List[str]] = None

# Initialize FastAPI app
# >>> THIS LINE IS CRUCIAL AND DEFINES THE 'app' ATTRIBUTE <<<
app = FastAPI(
    title="Audio-to-RAG-to-TTS Service",
    description="Processes an audio query through STT, performs RAG using Qwen and FAISS, and returns the answer as text and TTS audio (Base64 MP3)."
)
# >>> END CRUCIAL LINE <<<


# --- Pydantic Schemas ---

class AudioQueryRequest(BaseModel):
    """Schema for the incoming audio query request."""
    audio_file_path: str = Field(
        r"C:\Users\anilsagar\Desktop\SLM_rag_project\sample-000021.mp3",
        description="Local path to the MP3 audio file to be transcribed."
    )
    top_k_chunks: int = 5

class AudioQueryResponse(BaseModel):
    """Schema for the API response."""
    original_audio_path: str
    transcribed_query: str
    generated_answer: str
    context_chunks_used: List[str]
    answer_audio_base64: str = Field(description="Base64 encoded MP3 audio of the generated answer.")


# --- 1. RAG Component Loading (Startup Event) ---
@app.on_event("startup")
async def load_rag_components():
    """Loads all models, tokenizer, index, and raw data on startup."""
    global global_index, global_embedder, global_llm_model, global_llm_tokenizer, \
           global_stt_model, global_stt_processor, global_raw_chunks

    print(f"\n--- 1. Loading RAG Components on Device: {DEVICE} ---")

    # A. Load raw text chunks
    try:
        raw_chunks_np = np.load(CORPUS_FILE_PATH, allow_pickle=True)
        global_raw_chunks = raw_chunks_np.tolist()
        print(f"Loaded {len(global_raw_chunks)} raw text chunks.")
    except Exception as e:
        print(f"Error loading raw text chunks from {CORPUS_FILE_PATH}: {e}")
        raise RuntimeError("Failed to load RAG corpus. Check CORPUS_FILE_PATH.")

    # B. Load FAISS Index
    try:
        global_index = read_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index (Dimension: {global_index.d}).")
    except Exception as e:
        print(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
        raise RuntimeError("Failed to load FAISS index. Check FAISS_INDEX_PATH.")

    # C. Load Sentence Transformer (Embedding Model) - For retrieval, typically run on CPU
    try:
        global_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME).to("cpu")
        print(f"Loaded Embedding Model ({EMBEDDING_MODEL_NAME}) on CPU.")
    except Exception as e:
        print(f"Error loading Sentence Transformer: {e}")
        raise RuntimeError("Failed to load Sentence Transformer model.")

    # D. Load LLM (Qwen) - 4-bit Quantization on GPU if available
    print("\n--- 2A. Loading LLM (Qwen/Qwen1.5-1.8B-Chat 4-bit) ---")
    try:
        global_llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            global_llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config
            )
            print(f"LLM loaded in 4-bit on GPU: {global_llm_model.device}")
        else:
            global_llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
            print("LLM loaded on CPU.")

    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise RuntimeError("Failed to load LLM model. Check VRAM and dependencies.")

    # E. Load STT Model (Whisper)
    print("\n--- 2B. Loading STT Model (openai/whisper-base) ---")
    try:
        global_stt_model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_NAME).to(DEVICE)
        global_stt_processor = WhisperProcessor.from_pretrained(STT_MODEL_NAME)
        print(f"STT Model loaded on device: {DEVICE}")
    except Exception as e:
        print(f"Error loading STT Model: {e}")
        raise RuntimeError("Failed to load Whisper STT model.")

    print("\n--- All Components Ready. Server is Live. ---")

# --- 2. STT Function (Audio to Text) ---
def transcribe_audio(audio_path):
    """Transcribes the given audio file using the global Whisper model."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"Transcribing audio file: {audio_path}...")
    
    # librosa handles audio loading and resampling (requires FFmpeg)
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process the audio data
    input_features = global_stt_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    input_features = input_features.to(global_stt_model.device)

    # Generate the transcription
    predicted_ids = global_stt_model.generate(
        input_features,
        forced_decoder_ids=global_stt_processor.get_decoder_prompt_ids(language="english", task="transcribe")
    )

    # Decode the IDs to text
    transcription = global_stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# --- 3. RAG Query Function (Text Generation) ---
def rag_query(query: str, top_k: int):
    """Performs the RAG query: Embed query, search index, retrieve context, generate answer."""
    
    # Embed the query
    query_vector = global_embedder.encode(query)
    query_vector = query_vector.astype('float32').reshape(1, -1)

    # Search the FAISS index
    distances, indices = global_index.search(query_vector, top_k)

    # Retrieve the context text
    context_chunks = [global_raw_chunks[i] for i in indices[0]]
    context_text = "\n\n".join(context_chunks)

    # Prepare the Prompt for Qwen
    system_prompt = (
        "You are an intelligent RAG chatbot. Your task is to answer the user's "
        "query based ONLY on the provided context below. Do not use external knowledge. "
        "If the context does not contain the answer, state that you could not find "
        "enough information in the provided context."
    )

    # Use Qwen's chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n---\n{context_text}\n---\n\nUser Query: {query}"}
    ]

    text = global_llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the Answer
    model_inputs = global_llm_tokenizer([text], return_tensors="pt").to(global_llm_model.device)

    generated_ids = global_llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
    )
    
    # Remove the prompt tokens from the generated sequence
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]

    response = global_llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip(), context_chunks

# --- 4. TTS Function (Text to Audio for API) ---
def text_to_audio_api(text: str) -> str:
    """
    Converts text to MP3 using gTTS, saves it temporarily, reads it as bytes,
    and returns the Base64 encoded string.
    """
    temp_filename = f"temp_rag_response_{uuid.uuid4()}.mp3"
    
    # Clean up the text (replaces @-@ with a hyphen for better pronunciation)
    cleaned_text = text.replace("@-@", "-")
    
    # Create a gTTS object and save the audio file temporarily
    tts = gTTS(text=cleaned_text, lang='en', tld='com', slow=False)
    tts.save(temp_filename)
    
    # Read the file content as binary data
    with open(temp_filename, "rb") as f:
        audio_bytes = f.read()
        
    # Clean up the temporary file
    os.remove(temp_filename)
    
    # Encode the audio bytes to Base64
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    return base64_audio

# --- 5. API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint to confirm component readiness."""
    return {
        "status": "ok", 
        "device": DEVICE, 
        "llm_loaded": global_llm_model is not None,
        "stt_loaded": global_stt_model is not None,
        "rag_data_loaded": global_index is not None and global_raw_chunks is not None
    }

@app.post("/query_audio", response_model=AudioQueryResponse)
async def run_audio_rag_process(request: AudioQueryRequest):
    """
    Handles the full pipeline: STT -> RAG -> TTS, returning the result as text and Base64 audio.
    """
    # 1. Component Check
    if not all([global_index, global_embedder, global_llm_model, global_llm_tokenizer, global_stt_model, global_stt_processor, global_raw_chunks]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core RAG/STT components failed to load during startup."
        )

    audio_path = request.audio_file_path

    try:
        # STAGE 1: STT (Audio to Text)
        stt_query = transcribe_audio(audio_path)
        
        if not stt_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="STT transcription failed or resulted in an empty query."
            )

        # STAGE 2: RAG (Text to Answer)
        print(f"\nRunning RAG Query with STT Result: {stt_query}")
        final_answer, context_list = rag_query(stt_query, request.top_k_chunks)
        
        # STAGE 3: TTS (Answer to Audio)
        base64_audio = text_to_audio_api(final_answer)

        # 4. Return Response
        return AudioQueryResponse(
            original_audio_path=audio_path,
            transcribed_query=stt_query,
            generated_answer=final_answer,
            context_chunks_used=context_list,
            answer_audio_base64=base64_audio
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        print(f"Internal processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during the RAG process: {e}"
        )
