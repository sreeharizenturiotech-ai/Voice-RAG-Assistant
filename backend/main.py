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
async def voice_chat(request: Request, file: UploadFile = File(...)):

    audio_path = "input.wav"

    # Save uploaded audio
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Default values (important for debugging)
    question = "❌ No transcription"
    answer = "❌ No answer generated"
    audio_url = ""

    try:
        # 🎤 Speech → Text
        question = speech_to_text(audio_path)
        print("User:", question)
    except Exception as e:
        print("STT ERROR:", e)

    try:
        # 🤖 Generate answer
        answer = generate_answer(question)
        print("Bot:", answer)
    except Exception as e:
        print("LLM ERROR:", e)

    try:
        # 🔊 Text → Speech
        audio_file = "response.mp3"
        tts = gTTS(answer)
        tts.save(audio_file)

        base_url = str(request.base_url)
        audio_url = base_url + audio_file
    except Exception as e:
        print("TTS ERROR:", e)

    return {
        "question": question,
        "answer": answer,
        "audio": audio_url
    }
