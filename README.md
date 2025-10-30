# 🎤 Multimodal-Emotion-Aware-Chatbot
This project is an AI-powered multimodal chatbot that detects emotions from text, audio, and video (facial expressions) in real-time, and generates empathetic responses using a language model.
It integrates speech, vision, and natural language understanding to create an emotionally intelligent interaction experience.

🚀 Features

🎙️ Speech-to-Text using Groq Whisper API

🧠 Text Emotion Recognition using DistilBERT model

🔊 Audio Emotion Recognition using Wav2Vec2 model

🎥 Facial Emotion Detection using DeepFace

💬 Empathetic Response Generation using BlenderBot

🔁 Text-to-Speech (TTS) with gTTS for voice-based replies

🪶 Built with Streamlit for a simple and interactive web UI


🧩 Architecture Overview

User speaks → Audio recorded via microphone

Groq Whisper → Converts speech to text

Emotion Detection:

Text emotion → via DistilBERT

Audio emotion → via Wav2Vec2

Video emotion → via DeepFace (webcam frame)

Fusion Module → Combines multimodal emotions with weighted averaging

Response Generation → LLM generates an empathetic response

Voice Output → Response is converted to speech (TTS)

⚙️ Installation   
1️⃣ Clone the repository
git clone https://github.com/your-username/multimodal-emotion-chatbot.git
cd multimodal-emotion-chatbot

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set up environment variables

Create a .env file in the project root and add your Groq API key:

GROQ_API_KEY=your_groq_api_key_here

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then open the displayed local URL (e.g., http://localhost:8501
) in your browser.

Click “🎙️ Start Talking” and interact with the chatbot in real time!

🧠 Models Used
Modality   	Model	                                                    Source
Text	      bhadresh-savani/distilbert-base-uncased-emotion         	Hugging Face
Audio     	audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim	    Hugging Face
Vision    	DeepFace	                                                OpenCV + DeepFace
Response	  facebook/blenderbot-400M-distill	                        Hugging Face
STT	        whisper-large-v3 (via Groq API)                         	Groq

🛠️ Requirements

Python 3.8+

Microphone & webcam access

FFmpeg installed (for pydub and deepface)

