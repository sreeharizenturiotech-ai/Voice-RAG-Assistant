const API = "https://voice-rag-assistant-1.onrender.com";

let mediaRecorder;
let audioChunks = [];

const micBtn = document.getElementById("micBtn");
const chat = document.getElementById("chat");
const status = document.getElementById("status");

// Add message to chat
function addMessage(text, type, audioUrl = null) {
    const div = document.createElement("div");
    div.className = `msg ${type}`;
    div.innerText = text;

    // If bot message → add play button
    if (type === "bot" && audioUrl) {
        const btn = document.createElement("button");
        btn.innerText = "🔊 Play";
        btn.className = "audio-btn";

        btn.onclick = () => {
            const audio = new Audio(audioUrl);
            audio.play();
        };

        div.appendChild(document.createElement("br"));
        div.appendChild(btn);
    }

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

// Mic click
micBtn.onclick = async () => {

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {

        status.innerText = "⏳ Processing...";
        micBtn.classList.remove("recording");

        const blob = new Blob(audioChunks, { type: "audio/wav" });

        const formData = new FormData();
        formData.append("file", blob, "audio.wav");

        try {
            const res = await fetch(`${API}/voice`, {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            // Show user message
            addMessage(data.question, "user");

            // Move mic up
            micBtn.classList.add("moved");

            // Show bot message
            addMessage(data.answer, "bot", data.audio);

            status.innerText = "Idle";

        } catch (err) {
            console.error(err);
            status.innerText = "Error!";
        }
    };

    mediaRecorder.start();

    status.innerText = "🎤 Recording...";
    micBtn.classList.add("recording");

    setTimeout(() => {
        mediaRecorder.stop();
    }, 5000);
};
