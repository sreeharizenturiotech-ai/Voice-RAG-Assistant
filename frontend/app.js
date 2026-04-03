const API = "https://voice-rag-assistant-1.onrender.com";

let mediaRecorder;
let audioChunks = [];

const startBtn = document.getElementById("startBtn");
const chat = document.getElementById("chat");
const status = document.getElementById("status");

startBtn.onclick = async () => {

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {

        status.innerText = "⏳ Processing...";

        const blob = new Blob(audioChunks, { type: "audio/wav" });

        const formData = new FormData();
        formData.append("file", blob, "audio.wav");

        try {
           const res = await fetch(`${API}/voice`, {
    method: "POST",
    body: formData
});

const data = await res.json();

console.log("FULL RESPONSE:", data);

// 🔥 SHOW RAW DATA (DEBUG)
addMessage("TRANSCRIPTION: " + data.question, "user");
addMessage("ANSWER: " + data.answer, "bot");

// 🔊 PLAY AUDIO (only if exists)
if (data.audio) {
    const audioPlayer = document.getElementById("audioPlayer");
    audioPlayer.src = data.audio;
    audioPlayer.play();
} else {
    console.log("No audio received");
}
        } catch (err) {
            console.error(err);
            status.innerText = "❌ Error";
        }
    };

    mediaRecorder.start();
    status.innerText = "🎙 Recording...";
    startBtn.classList.add("recording");

    setTimeout(() => {
        mediaRecorder.stop();
        startBtn.classList.remove("recording");
    }, 5000);
};

function addMessage(text, type) {
    const div = document.createElement("div");
    div.className = `msg ${type}`;
    div.innerText = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}
