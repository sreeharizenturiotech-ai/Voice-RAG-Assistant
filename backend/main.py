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
# LOAD MODELS
# =========================
print("Loading Whisper...")
stt_model = whisper.load_model("tiny")  # 🔥 faster

print("Loading documents...")
with open("rag_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [doc["text"] for doc in documents]

# =========================
# CHUNKING
# =========================
def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

chunks = []
for t in texts:
    chunks.extend(chunk_text(t))

# =========================
# EMBEDDINGS
# =========================
print("Loading embeddings...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================
# RETRIEVER
# =========================
def retrieve_top_k(question, k=5):
    query_embedding = embedder.encode([question]).astype("float32")
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# =========================
# SIMPLE ANSWER (NO QWEN ❌)
# =========================
def generate_answer(context, question):
    if not context:
        return "I don't know based on the provided documents."
    
    return context[:300]  # simple + fast

# =========================
# API
# =========================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):
    try:
        audio_path = "input.wav"

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

        # 🔍 RAG
        docs = retrieve_top_k(question)
        context = "\n".join(docs)

        answer = generate_answer(context, question)

        print("🤖 Bot:", answer)

        # 🔊 TTS
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
# RUN SERVER (IMPORTANT)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
