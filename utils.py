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
import re
from langdetect import detect, DetectorFactory
import uuid
import json
from flask import jsonify
import nltk
from pydub import AudioSegment

nltk.download('words')  # Tải từ vựng để nhận diện

DetectorFactory.seed = 0

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
audio_dir = "audio"  # Define audio directory

# Define CORS headers
cors_headers = {"Access-Control-Allow-Origin": "*"}

def load_prompt(chat_mode: str = "general"):
    if chat_mode == "topic_based":
        return PromptTemplate.from_template("""
       You are acting as a '{role}' for the topic '{topic}' in English. Respond naturally to the user's input with a relevant answer related to the topic, and follow up with a related question (max 150 words) to keep the conversation going. If the user's input is short (e.g., 'yes', 'no') or empty, ask a new question based on the topic. If the user says 'stop' or 'change topic', end or switch accordingly. Every 50 exchanges, ask if the user wants to continue, switch topics, or stop.

        Your role includes:
        - Acting like a real person: If the user asks about locations (e.g., "Where is the restaurant?"), use their location data (if provided) or suggest realistic options. For other queries (e.g., weather, news), provide accurate answers if possible.
        - Engaging with humor or encouragement (e.g., "Great job! Let’s keep practicing!").

        Previous conversation:
        {chat_history}

        New human question: {question}
        Response:
        """)
    else:  # chat_mode == "general"
        return PromptTemplate.from_template("""
        Your name is Emma. You are an AI conversational English teacher acting as '{role}'. The user has chosen the topic '{topic}' and must use English. Act naturally, respond to the user's question or input with a relevant answer in English, and always follow up with a related question (max 150 words) to keep the conversation going. If the user's input is short (e.g., 'yes', 'no') or empty, ask a new question based on the topic. If the user says 'stop' or 'change topic', end or switch accordingly. Every 50 exchanges, ask if the user wants to continue, switch topics, or stop.

        Your role includes:
        - Acting like a real person: If the user asks about locations (e.g., "Where is the restaurant?"), use their location data (if provided) or suggest realistic options. For other queries (e.g., weather, news), provide accurate answers if possible.
        - Engaging with humor or encouragement (e.g., "Great job! Let’s keep practicing!").

        Previous conversation:
        {chat_history}

        New human question: {question}
        Response:
        """)

def load_llm():
    return ChatGroq(temperature=0.81, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

def generate_audio(text: str, audio_path: str, language: str, topic: str, role: str):
    try:
        voice = "en-US-AriaNeural"
        tts = edge_tts.Communicate(text, voice=voice)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts.save(audio_path))
        loop.close()
        print(f"Audio generated with edge_tts at {audio_path}")
        # Compress audio
        audio = AudioSegment.from_file(audio_path, format="mp3")
        compressed_path = audio_path.replace(".mp3", "_compressed.mp3")
        audio.export(compressed_path, format="mp3", bitrate="32k")
        os.remove(audio_path)
        os.rename(compressed_path, audio_path)
        print(f"Audio compressed at {audio_path}")
    except Exception as e:
        print(f"edge_tts failed: {str(e)}. Falling back to gTTS.")
        from gtts import gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(audio_path)
        # Compress audio
        audio = AudioSegment.from_file(audio_path, format="mp3")
        compressed_path = audio_path.replace(".mp3", "_compressed.mp3")
        audio.export(compressed_path, format="mp3", bitrate="32k")
        os.remove(audio_path)
        os.rename(compressed_path, audio_path)
        print(f"Audio generated and compressed with gTTS at {audio_path}")

def get_response(user_text: str, memory, topic: str = "", language: str = "en", role: str = "AI conversational teacher", chat_mode: str = "general") -> str:
    prompt = load_prompt(chat_mode=chat_mode)
    llm = load_llm()
    print(f"Processing with topic: {topic}, language: {language}, role: {role}, chat_mode: {chat_mode}")
    chat_history_str = ""
    if memory is not None:
        try:
            chat_history_str = memory.load_memory_variables({})["chat_history"]
            print(f"Memory chat_history: {chat_history_str}")
        except KeyError as e:
            print(f"Error accessing chat_history: {e}. Defaulting to empty chat history.")
            chat_history_str = ""
    chain = (
        {
            "question": RunnablePassthrough(),
            "topic": lambda x: topic,
            "language": lambda x: "en",
            "chat_history": lambda x: chat_history_str,
            "role": lambda x: role
        }
        | prompt
        | llm
    )
    try:
        response = chain.invoke(user_text)
        return response.content
    except Exception as e:
        print(f"Error in LLM chain: {str(e)}")
        return "An internal error occurred. Please try again later."

async def play_text_to_speech(text: str, language: str = 'en'):
    voice = "en-US-AriaNeural"
    tts = edge_tts.Communicate(text, voice=voice)
    temp_audio_file = "temp_audio.mp3"
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        await tts.save(temp_audio_file)
        loop.close()
        pygame.mixer.init()
        sound = pygame.mixer.Sound(temp_audio_file)
        sound.play()
        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing text-to-speech: {e}")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

if __name__ == "__main__":
    class DummyMemory:
        def load_memory_variables(self, variables):
            return {"chat_history": "Human: Hi there!\nAI: Hello! How can I help you today?"}

    dummy_memory = DummyMemory()

    print("--- Test 1: Nonsensical input ---")
    senseless_input = "jdasfskahsdadfsdjdasfskahsdadfsdjdasfskahsdadfsdjdasfskahsda"
    response1 = get_response(senseless_input, dummy_memory, topic="Travel", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for senseless input: {response1}\n")

    print("--- Test 2: Non-English input (Vietnamese) ---")
    non_english_input = "Tôi muốn đi Hà Nội"
    response2 = get_response(non_english_input, dummy_memory, topic="Travel", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for non-English input: {response2}\n")

    print("--- Test 3: Valid English input ---")
    valid_input = "What's the capital of France?"
    response3 = get_response(valid_input, dummy_memory, topic="Geography", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for valid input: {response3}\n")

    print("--- Test 4: Short valid input ---")
    short_valid_input = "yes"
    response4 = get_response(short_valid_input, dummy_memory, topic="Daily Life", language="English Teacher", role="en", chat_mode="general")
    print(f"Response for short valid input: {response4}\n")

    print("--- Test 5: Empty input ---")
    empty_input = ""
    response5 = get_response(empty_input, dummy_memory, topic="Daily Life", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for empty input: {response5}\n")

    print("--- Test 6: 'stop' command ---")
    stop_command = "stop"
    response6 = get_response(stop_command, dummy_memory, topic="Daily Life", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for 'stop' command: {response6}\n")

    print("--- Test 7: Non-English input (Chinese) ---")
    chinese_input = "我想去北京"
    response7 = get_response(chinese_input, dummy_memory, topic="Travel", language="en", role="English Teacher", chat_mode="general")
    print(f"Response for Chinese input: {response7}\n")