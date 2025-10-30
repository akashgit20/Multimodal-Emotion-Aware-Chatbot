import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sounddevice as sd
import streamlit as st
import threading
import platform
import subprocess
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import logging
import speech_recognition as sr
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, pipeline
)
from deepface import DeepFace
from dotenv import load_dotenv
from groq import Groq  

# ---------------------- Load API Keys ---------------------- #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------- Load Models ---------------------- #
# Text Model
text_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_model.to("cpu")
text_emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]

# Audio Emotion Model
audio_model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_model.to("cpu")
audio_emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# LLM Response
llm_pipeline = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

# ---------------------- Config ---------------------- #
emotion_weights = {"text": 0.4, "audio": 0.3, "video": 0.3}
logging.basicConfig(level=logging.INFO)

# ---------------------- Speech-to-Text with GROQ Whisper ---------------------- #
def record_audio(file_path, timeout=20, phrase_time_limit=10):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("🎙️ Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            audio_segment = AudioSegment.from_raw(BytesIO(wav_data), sample_width=2, frame_rate=16000, channels=1)
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            return True
    except sr.WaitTimeoutError:
        st.warning("⏱️ Timed out while waiting for speech.")
    except Exception as e:
        st.error(f"🎤 Recording error: {e}")
    return False

def transcribe_with_groq(audio_filepath, stt_model="whisper-large-v3"):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_filepath, "rb") as f:
            transcription = client.audio.transcriptions.create(model=stt_model, file=f, language="en")
        return transcription.text
    except Exception as e:
        st.error(f"📝 Transcription failed: {e}")
        return ""

# ---------------------- Text-to-Speech ---------------------- #
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    audio = AudioSegment.from_mp3(filename)
    wav_filename = filename.replace(".mp3", ".wav")
    audio.export(wav_filename, format="wav")

    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_filename}").PlaySync();'])
        elif os_name == "Darwin":
            subprocess.run(['afplay', wav_filename])
        else:
            subprocess.run(['aplay', wav_filename])
    except Exception as e:
        st.error(f"Error playing audio: {e}")

# ---------------------- Emotion Detection ---------------------- #
def predict_text_emotion(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    return text_emotion_labels[pred_class]

def predict_audio_emotion(audio_data):
    inputs = audio_processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = audio_model(**inputs)
    pred_class = torch.argmax(outputs.logits, dim=1).item()
    return audio_emotion_labels[pred_class]


def record_audio_for_emotion(duration=8, sr=16000):
    st.info("🔊 Recording audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def predict_video_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "No face detected"
    cv2.imwrite("temp_frame.jpg", frame)
    try:
        result = DeepFace.analyze(img_path="temp_frame.jpg", actions=["emotion"], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        return "Error"

# ---------------------- Combine Emotions ---------------------- #
def calculate_weighted_emotion(emotions):
    score_map = {e: 0 for e in text_emotion_labels}
    for src, emotion in emotions.items():
        if emotion in score_map:
            score_map[emotion] += emotion_weights[src]
    return max(score_map, key=score_map.get)

# ---------------------- Generate LLM Response ---------------------- #
def generate_response(text, emotion):
    prompt = f"The user is feeling {emotion}. Respond empathetically to: {text}"
    result = llm_pipeline(prompt, max_length=100, truncation=True)
    return result[0]["generated_text"]

# ---------------------- Streamlit App ---------------------- #
st.title("🎤 Real time Multimodal Emotion Aware Chatbot")

if st.button("🎙️ Start Talking"):
    st.info("🎙️ Speak now...")
    audio_path = "temp_input.mp3"
    if record_audio(audio_path):
        spoken_text = transcribe_with_groq(audio_path)
        st.success(f"🗣️ You said: {spoken_text}")
        
        if spoken_text:
            results = {}

            def task_text():
                results["text"] = predict_text_emotion(spoken_text)

            def task_audio():
                audio_data = record_audio_for_emotion()
                results["audio"] = predict_audio_emotion(audio_data)

            def task_video():
                results["video"] = predict_video_emotion()

            t1 = threading.Thread(target=task_text)
            t2 = threading.Thread(target=task_audio)
            t3 = threading.Thread(target=task_video)

            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()

            final_emotion = calculate_weighted_emotion(results)
            response = generate_response(spoken_text, final_emotion)

            st.subheader("🧾 Detected Emotions")
            for mode, emo in results.items():
                st.write(f"**{mode.capitalize()} Emotion:** {emo}")
            st.write(f"🟢 **Final Weighted Emotion:** {final_emotion}")

            st.subheader("🤖 Response")
            st.write(response)
            text_to_speech(response)
