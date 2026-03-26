from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS

import os
import imageio_ffmpeg

# Set ffmpeg path manually
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

# =========================
# ✅ CORS FIX (IMPORTANT)
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
stt_model = whisper.load_model("base")

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
# EMBEDDINGS + FAISS
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

def retrieve_top_k(question, k=5, threshold=1.0):
    query_embedding = embedder.encode([question]).astype("float32")
    D, I = index.search(query_embedding, k)

    if D[0][0] > threshold:
        return []

    return [chunks[i] for i in I[0]]

# =========================
# LOAD QWEN MODEL
# =========================

print("Loading Qwen...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

# =========================
# GENERATE ANSWER
# =========================

def generate_answer(context, question):
    prompt = f"""
You are a helpful assistant.

Answer using ONLY the given context.
Max 2 short sentences.
If not found, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()

    return text.strip()

# =========================
# API ROUTE
# =========================

@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):

    audio_path = "input.wav"

    # Save incoming audio
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # 🎤 Speech → Text
    result = stt_model.transcribe(audio_path)
    question = result["text"].strip()

    print("User:", question)

    # 🔍 RAG retrieval
    docs = retrieve_top_k(question)

    if not docs:
        answer = "I don't know based on the provided documents."
    else:
        context = "\n\n".join(docs)
        answer = generate_answer(context, question)

    print("Bot:", answer)

    # 🔊 Text → Speech
    tts = gTTS(answer)
    audio_file = "response.mp3"
    tts.save(audio_file)

    return {
        "question": question,
        "answer": answer,
        "audio": f"http://127.0.0.1:8000/{audio_file}"
    }
