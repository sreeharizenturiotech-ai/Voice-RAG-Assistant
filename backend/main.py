from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
from gtts import gTTS

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
# ENV VARIABLES
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# LOAD DOCUMENTS
# =========================
with open("rag_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Simple context join (lightweight RAG)
def get_context():
    return "\n\n".join([doc["text"] for doc in documents[:3]])

# =========================
# SPEECH → TEXT (OpenAI Whisper API)
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
# GENERATE ANSWER (LLM API)
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
# ROUTE
# =========================
@app.post("/voice")
async def voice_chat(file: UploadFile = File(...)):

    audio_path = "input.wav"

    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # 🎤 STT
    question = speech_to_text(audio_path)

    # 🤖 LLM
    answer = generate_answer(question)

    # 🔊 TTS
    tts = gTTS(answer)
    audio_file = "response.mp3"
    tts.save(audio_file)

    return {
        "question": question,
        "answer": answer,
        "audio": audio_file
    }
