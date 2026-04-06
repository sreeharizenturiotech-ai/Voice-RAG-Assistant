# =========================
# IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import requests
import json
import os
from gtts import gTTS

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# CORS (IMPORTANT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SERVE AUDIO FILES
# =========================
app.mount("/", StaticFiles(directory="."), name="static")

# =========================
# ENV VARIABLE
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# LOAD DOCUMENTS (RAG)
# =========================
with open("rag_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

def get_context():
    return "\n\n".join([doc["text"] for doc in documents[:3]])

# =========================
# SPEECH → TEXT (API)
# =========================
def speech_to_text(audio_path):
    url = "https://api.openai.com/v1/audio/transcriptions"

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            files={
                "file": f,
                "model": (None, "whisper-1")
            }
        )

    return response.json()["text"]

# =========================
# LLM RESPONSE (API)
# =========================
def generate_answer(question):
    context = get_context()

    url = "https://api.openai.com/v1/chat/completions"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer using ONLY given context. Max 2 short sentences."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# =========================
# API ROUTE
# =========================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):
    try:
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

        # 🔍 RAG
        docs = retrieve_top_k(question)

        if not docs:
            answer = "I don't know based on the provided documents."
        else:
            context = "\n\n".join(docs)
            answer = generate_answer(context, question)

        print("🤖 Bot:", answer)

        # 🔊 Text → Speech
        audio_file = "response.mp3"
        tts = gTTS(answer)
        tts.save(audio_file)

        # ✅ IMPORTANT: return FULL URL
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
