import os
from dotenv import load_dotenv
import wave
import pyaudio
from scipy.io import wavfile
import numpy as np
import whisper
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
import edge_tts
import pygame

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def is_silence(data, max_amplitude_threshold=3000):
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    num_chunks = int(16000 / 1024 * chunk_length)
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return True

def load_whisper():
    return whisper.load_model("tiny")

def transcribe_audio(model, file_path):
    print("Transcribing...")
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path)
        return results['text']
    else:
        return None

def load_prompt():
    return PromptTemplate.from_template("""
    You are a AI conversational teacher acting as '{role}'. The user has chosen the topic '{topic}' and language '{language}'. Act naturally, respond to the user's question or input with a relevant answer, and always follow up with a related question (max 10 words) to keep the conversation going. If the user's input is short (e.g., 'yes', 'no') or empty, ask a new question based on the topic. If user says 'stop' or 'change topic', end or switch accordingly. Every 5 exchanges, ask if the user wants to continue, switch topics, or stop.

    Language-specific instructions:
    - If language is 'en', respond naturally in English.
    - If language is 'zh', respond in Mandarin Chinese, paying attention to tone and word order for natural conversation (e.g., "你想吃什么菜？").
    - If language is 'ko', respond in Korean with polite forms (e.g., use '요' or '습니다' endings, like "어떤 음식을 드시고 싶으세요?").
    - If language is 'ja', respond in Japanese with polite forms (e.g., use 'です/ます' endings, like "どのような料理がお好きですか？").
    - If language is 'vi', respond in Vietnamese with natural tone and accents (e.g., "Bạn muốn ăn món gì?").

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
""")

def load_llm():
    return ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

def get_response_llm(user_question, memory, topic="", language="en", role="AI conversational teacher"):
    prompt = load_prompt()
    llm = load_llm()

    print(f"Processing with topic: {topic}, language: {language}, role: {role}")
    chat_history = []
    if memory is not None:
        try:
            chat_history = memory.load_memory_variables({})["chat_history"]
            print(f"Memory chat_history: {chat_history}")
        except KeyError as e:
            print(f"Error accessing chat_history: {e}")
            chat_history = []  # Fallback if chat_history doesn't exist

    chain = (
        {
            "question": RunnablePassthrough(),
            "topic": lambda x: topic,
            "language": lambda x: language,
            "chat_history": lambda x: chat_history,
            "role": lambda x: role
        }
        | prompt
        | llm
    )

    try:
        response = chain.invoke(user_question)
        return response.content
    except Exception as e:
        print(f"Error in LLM chain: {str(e)}")
        raise

async def play_text_to_speech(text, language='en'):
    voice = {
        "en": "en-US-AriaNeural",
        "vi": "vi-VN-AnNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "zh": "zh-CN-XiaoxiaoNeural"
    }.get(language, "en-US-AriaNeural")
    tts = edge_tts.Communicate(text, voice=voice)
    temp_audio_file = "temp_audio.mp3"
    await tts.save(temp_audio_file)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_audio_file)