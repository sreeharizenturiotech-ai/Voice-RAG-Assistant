from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import whisper
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS

import os
import imageio_ffmpeg

# =========================
# FIX FFMPEG
# =========================
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

# =========================
# CREATE APP
# =========================
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL VARIABLES (LAZY LOAD)
# =========================
stt_model = None
embedder = None
index = None
chunks = None

# =========================
# LOAD MODELS ONLY WHEN NEEDED
# =========================
def load_models():
    global stt_model, embedder, index, chunks

    if stt_model is None:
        print("🔥 Loading Whisper...")
        stt_model = whisper.load_model("tiny")

    if embedder is None:
        print("🔥 Loading embeddings...")

        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        with open("rag_documents.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        texts = [doc["text"] for doc in documents]

        def chunk_text(text, chunk_size=400, overlap=80):
            words = text.split()
            chunks_local = []
            i = 0
            while i < len(words):
                chunk = words[i:i + chunk_size]
                chunks_local.append(" ".join(chunk))
                i += chunk_size - overlap
            return chunks_local

        chunks_local = []
        for t in texts:
            chunks_local.extend(chunk_text(t))

        embeddings = embedder.encode(chunks_local)
        embeddings = np.array(embeddings).astype("float32")

        index_local = faiss.IndexFlatL2(embeddings.shape[1])
        index_local.add(embeddings)

        index = index_local
        chunks = chunks_local

# =========================
# API ROUTE
# =========================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):
    try:
        load_models()  # 🔥 IMPORTANT

        audio_path = "input.wav"

        # Save audio
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        print("✅ Audio received")

        # 🎤 Speech → Text
        result = stt_model.transcribe(audio_path)
        question = result.get("text", "").strip()

        print("🧑 User:", question)

        if not question:
            return {
                "question": "No speech detected",
                "answer": "Please try again",
                "audio": ""
            }

        # 🔍 RAG Retrieval
        query_embedding = embedder.encode([question]).astype("float32")
        D, I = index.search(query_embedding, 5)

        docs = [chunks[i] for i in I[0]]
        context = "\n".join(docs)

        if not context:
            answer = "I don't know based on the provided documents."
        else:
            answer = context[:300]

        print("🤖 Bot:", answer)

        # 🔊 Text → Speech
        audio_file = "response.mp3"
        tts = gTTS(answer)
        tts.save(audio_file)

        return {
            "question": question,
            "answer": answer,
            "audio": f"https://voice-rag-assistant-1.onrender.com/{audio_file}"
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "question": "error",
            "answer": str(e),
            "audio": ""
        }

# =========================
# ROOT CHECK (OPTIONAL)
# =========================
@app.get("/")
def home():
    return {"message": "Voice RAG Assistant is running 🚀"}

# =========================

