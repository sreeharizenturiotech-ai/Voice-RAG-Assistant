# =========================
# ✅ FIX FFmpeg (MUST BE FIRST)
# =========================
import os
import imageio_ffmpeg

os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

# =========================
# IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import whisper
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# ✅ CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ✅ Serve audio files
# =========================
app.mount("/", StaticFiles(directory="."), name="static")

# =========================
# LOAD MODELS
# =========================
print("Loading Whisper...")
stt_model = whisper.load_model("base")  # ⚠️ use "tiny" for faster

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
# LOAD LLM (QWEN)
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
async def voice_chat(request: Request, file: UploadFile = File(...)):

    audio_path = "input.wav"

    # Save audio
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # 🎤 Speech → Text
    result = stt_model.transcribe(audio_path)
    question = result["text"].strip()

    print("User:", question)

    # 🔍 Retrieve
    docs = retrieve_top_k(question)

    if not docs:
        answer = "I don't know based on the provided documents."
    else:
        context = "\n\n".join(docs)
        answer = generate_answer(context, question)

    print("Bot:", answer)

    # 🔊 Text → Speech
    audio_file = "response.mp3"
    tts = gTTS(answer)
    tts.save(audio_file)

    # ✅ Dynamic URL (IMPORTANT for Render)
    base_url = str(request.base_url)

    return {
        "question": question,
        "answer": answer,
        "audio": base_url + audio_file
    }
