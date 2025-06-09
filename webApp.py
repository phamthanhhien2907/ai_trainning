from flask import Flask, render_template, request, jsonify, send_file, make_response
import os
from flask_cors import CORS
import json
import uuid
from langchain.memory import ConversationBufferWindowMemory
from utils import get_response, generate_audio
import edge_tts
from pydub import AudioSegment
from gtts import gTTS
import pandas as pd
import random
import logging
import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample
from better_profanity import profanity
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk

# Download required NLTK resources
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.corpus import words, wordnet
from nltk.tokenize import word_tokenize

# Khởi tạo english_words
english_words = set(words.words())

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": ["*"]}})

# Directory for audio files
audio_dir = os.path.join(os.getcwd(), "audio")
os.makedirs(audio_dir, exist_ok=True)

# Thiết lập seed cho langdetect
DetectorFactory.seed = 0

# Load word pronunciation dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_DATA_PATH = os.path.join(BASE_DIR, "./databases/data_word_en.csv")
print(f"Loading data_word_en.csv from: {WORD_DATA_PATH}")
try:
    data_word_en = pd.read_csv(WORD_DATA_PATH)
    print(f"Loaded data_word_en.csv with columns: {data_word_en.columns.tolist()}")
    if data_word_en.empty:
        print("Warning: data_word_en.csv is empty")
except FileNotFoundError:
    print(f"Error: data_word_en.csv not found at {WORD_DATA_PATH}")
    data_word_en = pd.DataFrame()
except Exception as e:
    print(f"Error loading data_word_en.csv: {str(e)}")
    data_word_en = pd.DataFrame()

# Load topic dataset
TOPIC_DATA_PATH = os.path.join(BASE_DIR, "./databases/data_topic_en.csv")
print(f"Loading data_topic_en.csv from: {TOPIC_DATA_PATH}")
try:
    data_topic_en = pd.read_csv(TOPIC_DATA_PATH)
    print(f"Loaded data_topic_en.csv with columns: {data_topic_en.columns.tolist()}")
    if data_topic_en.empty:
        print("Warning: data_topic_en.csv is empty")
except FileNotFoundError:
    print(f"Error: data_topic_en.csv not found at {TOPIC_DATA_PATH}")
    data_topic_en = pd.DataFrame()
except Exception as e:
    print(f"Error loading data_topic_en.csv: {str(e)}")
    data_topic_en = pd.DataFrame()

# Conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=100
)

# CORS headers
cors_headers = {
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
}

# Cache SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache greeting_words từ WordNet
greeting_words = None
def init_greeting_words():
    global greeting_words
    if greeting_words is None:
        greeting_words = set(['hello', 'hi', 'hey', 'excuse', 'sorry', 'please', 'greetings', 'apology'])
        greeting_concepts = ['greeting.n.01', 'apology.n.01', 'please.v.01', 'salutation.n.01']
        greeting_synsets = [wordnet.synset(concept) for concept in greeting_concepts]
        for synset in greeting_synsets:
            for lemma in synset.lemmas():
                greeting_words.add(lemma.name().lower())
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    greeting_words.add(lemma.name().lower())
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    greeting_words.add(lemma.name().lower())
init_greeting_words()

# Hàm kiểm tra câu xã giao
def is_greeting(text):
    if not text:
        return False
    tokens = word_tokenize(text.lower())
    # Kiểm tra WordNet
    if any(token in greeting_words for token in tokens):
        return True
    # Fallback: Kiểm tra tương đồng ngữ nghĩa với câu mẫu
    user_embedding = sentence_model.encode(text, convert_to_tensor=True)
    greeting_embedding = sentence_model.encode("Hello, how can I assist you?", convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, greeting_embedding).item()
    return similarity > 0.7

# Routes
rootPath = ''

@app.route(rootPath + '/', methods=['GET'])
@app.route("/GetAccuracyFromRecordedAudio", methods=["POST", "OPTIONS"])
def get_accuracy_from_recorded_audio():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    try:
        request_data = request.get_json(force=False)
        if request_data is None:
            logging.error("Invalid or missing JSON payload")
            return _corsify_actual_response(
                jsonify({"error": "No JSON data found. Please send a valid request."})
            ), 400
        logging.info(f"Received request data: {request_data}")
        recorded_audio = request_data.get("recorded_audio") or request_data.get("base64Audio")
        text = request_data.get("text") or request_data.get("title")
        language = request_data.get("language", "en")
        if not recorded_audio or not text:
            logging.error(f"Missing fields: recorded_audio/base64Audio={bool(recorded_audio)}, text/title={bool(text)}")
            return _corsify_actual_response(
                jsonify({"error": "Please provide both audio and text to proceed."})
            ), 400
        import base64
        try:
            audio_data = recorded_audio
            if ',' in recorded_audio:
                audio_data = recorded_audio.split(',')[1]
            base64.b64decode(audio_data)
        except Exception as e:
            logging.error(f"Invalid base64 audio: {str(e)}")
            return _corsify_actual_response(
                jsonify({"error": f"Invalid audio format: {str(e)}. Please check your audio data."})
            ), 400
        if language != "en":
            return _corsify_actual_response(
                jsonify({"error": "We only support English for now. Set language to 'en'."})
            ), 400
        event = {
            "body": json.dumps({
                "base64Audio": recorded_audio,
                "text": text,
                "language": language
            })
        }
        context = {}
        response = lambdaSpeechToScore.lambda_handler(event, context)
        response_data = json.loads(response['body'])
        if response['statusCode'] != 200:
            logging.error(f"Lambda error: {response_data.get('error')}")
            return _corsify_actual_response(
                jsonify({"error": response_data.get("error", "Something went wrong with audio processing.")})
            ), response['statusCode']
        logging.info(f"Successful response: {response_data}")
        return _corsify_actual_response(jsonify(response_data))
    except Exception as e:
        logging.error(f"Error processing accuracy request: {str(e)}")
        return _corsify_actual_response(
            jsonify({"error": f"Oops, our server hiccuped: {str(e)}. Try again?"})
        ), 500

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/text-and-respond/', methods=['POST'])
def text_and_respond():
    try:
        data = request.get_json(force=True)
        user_text = data.get('user_text', '').strip()
        topic = data.get('topic', '').strip()
        language = data.get('language', 'en').strip()
        role = data.get('role', 'English Teacher').strip()
        chat_mode = data.get('chat_mode', 'general')
        recorded_audio = data.get('recorded_audio')
        user_location = data.get('location', None)

        print(f"Received user_text: '{user_text}', topic: {topic}, role: {role}, language: {language}, chat_mode: {chat_mode}, audio: {bool(recorded_audio)}, location: {user_location}")

        # Kiểm tra đầu vào rỗng hoặc quá ngắn
        if not user_text and not recorded_audio:
            return jsonify({
                "error": "Nothing to say? Share something about the topic!",
                "hint": f"How about a sentence related to '{topic}'?" if topic else "Start with a simple question!"
            }), 400, cors_headers
        if user_text and len(user_text) < 2:
            return jsonify({
                "error": "That’s a bit short! Can you say more?",
                "hint": f"Try a full sentence about '{topic}'." if topic else "Add a few more words!"
            }), 400, cors_headers

        # Kiểm tra ngôn ngữ và xử lý tiếng Việt
        is_vietnamese = False
        try:
            if user_text and detect(user_text) == 'vi':
                is_vietnamese = True
        except LangDetectException:
            pass  

        # Kiểm tra từ chối lời lẽ tục tĩu
        if profanity.contains_profanity(user_text or ''):
            return jsonify({
                "error": "Whoa, let's keep it friendly! Try a different phrase.",
                "hint": "Use polite words to continue the conversation."
            }), 400, cors_headers

        # Kiểm tra input bậy bạ (nhấn linh tinh trên bàn phím) bằng regex
        if re.match(r'^[a-zA-Z\s]{0,2}$|^[^a-zA-Z]*$', user_text):
            # Bỏ qua kiểm tra này nếu là câu xã giao trong topic_based
            if chat_mode != 'topic_based' or not is_greeting(user_text):
                return jsonify({
                    "error": "Oops, that looks like a typo! Please type a valid message.",
                    "hint": f"Try something related to '{topic}' if you have one, or start with a greeting!"
                }), 400, cors_headers

        # Xử lý theo chat_mode
        if chat_mode == 'topic_based' and topic:
            # Cho phép câu chào hỏi hoặc phản hồi hợp lệ
            if is_greeting(user_text):
                response_llm = f"Hi! Let's talk about '{topic}'. What would you like to say?"
            else:
                # Kiểm tra nội dung có liên quan đến topic bằng SentenceTransformer
                filtered_data = data_topic_en[data_topic_en['word'].str.contains(topic, case=False, na=False) |
                                             data_topic_en['exampleSentence'].str.contains(topic, case=False, na=False)]
                if not filtered_data.empty:
                    sample_sentence = filtered_data['exampleSentence'].iloc[0] if not pd.isna(filtered_data['exampleSentence'].iloc[0]) else filtered_data['word'].iloc[0]
                    user_embedding = sentence_model.encode(user_text, convert_to_tensor=True)
                    topic_embedding = sentence_model.encode(sample_sentence, convert_to_tensor=True)
                    similarity = util.cos_sim(user_embedding, topic_embedding).item()
                    if similarity < 0.6:
                        return jsonify({
                            "error": "Sorry, that doesn’t fit the topic. Please stay on topic!",
                            "hint": f"Say something related to '{topic}' like 'I’d like to order pasta'."
                        }), 400, cors_headers
                response_llm = get_response(user_text, memory, topic, language, role, chat_mode)
        elif not topic and chat_mode == 'topic_based':
            return jsonify({
                "error": "Hey, you forgot to pick a topic! Choose one to get started.",
                "hint": "Try topics like 'Ordering a taxi' or 'Job interview'."
            }), 400, cors_headers
        else:
            # Chế độ general với tên Emma và vai trò teacher AI
            role = "Emma your teacher AI"
            if not topic and is_greeting(user_text):
                response_llm = f"Let's chat! What would you like to learn today? {('Note: Please use English for the best experience!' if is_vietnamese else '')}"
            else:
                response_llm = get_response(user_text, memory, topic, language, role, chat_mode)

        # Cảnh báo nếu là tiếng Việt
        if is_vietnamese:
            response_llm = f"{response_llm} [Warning: Detected non-English input. Please switch to English for better assistance!]"

        memory.save_context({'input': user_text or 'Start conversation'}, {'output': response_llm})

        max_text_length = 300
        # Cắt câu hoàn chỉnh trong 50 từ
        response_words = response_llm.split()
        if len(response_words) > 50:
            truncated_text = " ".join(response_words[:50])
            last_punctuation = max(truncated_text.rfind('.'), truncated_text.rfind('?'))
            if last_punctuation != -1:
                response_llm = truncated_text[:last_punctuation + 1]
            else:
                response_llm = truncated_text
        response_llm = response_llm[:max_text_length] if len(response_llm) > max_text_length else response_llm

        audio_filename = f"audio_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)
        generate_audio(response_llm, audio_path, language, topic, role)

        return jsonify({
            "user_text": user_text,
            "ai_response": response_llm,
            "audio_url": f"/audio/{audio_filename}",
            "language": language,
            "message": "Great job! Here's my response. Keep the conversation going!",
            "location": user_location if user_location else None
        }), 200, cors_headers

    except Exception as e:
        error_msg = f"Error in text-and-respond endpoint: {str(e)}"
        print(error_msg)
        return jsonify({
            "error": "Yikes, something went wrong! Can you try again?",
            "hint": "Please check your input or try a different topic."
        }), 500, cors_headers

@app.route('/suggest-response/', methods=['POST'])
def suggest_response():
    try:
        data = request.get_json(force=True)
        print(f"Request body: {data}")
        ai_response = data.get('ai_response', '').strip()
        topic = data.get('topic', '').strip()
        language = data.get('language', 'en').strip()
        role = data.get('role', 'English Teacher').strip()
        suggest_for = data.get('suggest_for', 'user').strip()
        chat_mode = data.get('chat_mode', 'general')

        if not topic and chat_mode == 'topic_based':
            return jsonify({
                "error": "No topic selected! Pick one to continue.",
                "hint": "Choose a topic like 'Coffee shop' or 'Job interview'."
            }), 400, cors_headers

        if language != 'en':
            return jsonify({
                "error": "English only, please! Set language to 'en'.",
                "hint": "Update your language setting and try again."
            }), 400, cors_headers

        if not ai_response:
            return jsonify({
                "error": "No AI response provided. Please include one!",
                "hint": "Send the AI's last message to get a suggestion."
            }), 400, cors_headers

        if suggest_for.lower() not in ['user', 'ai']:
            return jsonify({
                "error": "Invalid suggestion type! Use 'user' or 'ai'.",
                "hint": "Check the 'suggest_for' field in your request."
            }), 400, cors_headers

        question_type_hint = ""
        ai_response_lower = ai_response.lower()
        if "what time" in ai_response_lower or "when" in ai_response_lower:
            question_type_hint = "The question is asking for a time. Suggest a response like 'At 8:00 PM' or 'Tomorrow morning'."
        elif "where" in ai_response_lower or "which" in ai_response_lower:
            question_type_hint = "The question is asking for a place or choice. Suggest a response like 'At the airport' or 'The Italian restaurant'."
        elif "what" in ai_response_lower:
            question_type_hint = "The question is asking for a choice or item. Suggest a response like 'I’d like a coffee' or 'A new laptop'."
        elif "how many" in ai_response_lower or "how much" in ai_response_lower:
            question_type_hint = "The question is asking for a quantity. Suggest a response like 'Two tickets, please' or 'About $50'."
        elif "why" in ai_response_lower or "how" in ai_response_lower:
            question_type_hint = "The question is asking for a reason or method. Suggest a response like 'Because I’m late' or 'By following the manual'."
        else:
            question_type_hint = "The question type is unclear. Suggest a response that fits the topic and context."

        topic_context = f"The topic '{topic}' involves a scenario where the user is interacting with a {role}. " if topic else ""

        if suggest_for.lower() == 'user':
            suggestion_prompt = (
                f"You are a {role} in {language}. The AI has asked: '{ai_response}'. "
                f"Suggest a detailed response (max 50 words) that the user can say to answer this question, using varied vocabulary and natural phrasing. "
                f"{topic_context}"
                f"{question_type_hint} "
                f"Do not continue the conversation or ask a new question. "
                f"Focus on providing a direct, natural response for the user."
            )
            response_llm = get_response(suggestion_prompt, None, topic, language, role)
        else:
            global memory
            if memory is None:
                print("Memory is None, reinitializing...")
                memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=100
                )
            memory.save_context({'input': 'AI'}, {'output': ai_response})
            suggestion_prompt = (
                f"Act as a {role} in {language}. Based on the AI's last message '{ai_response}', "
                f"suggest a relevant response (max 50 words) for the topic '{topic}', using varied vocabulary. "
                f"{topic_context}"
            )
            response_llm = get_response(suggestion_prompt, memory, topic, language, role)

        # Cắt câu hoàn chỉnh trong 50 từ
        response_words = response_llm.split()
        if len(response_words) > 50:
            truncated_text = " ".join(response_words[:50])
            last_punctuation = max(truncated_text.rfind('.'), truncated_text.rfind('?'))
            if last_punctuation != -1:
                response_llm = truncated_text[:last_punctuation + 1]
            else:
                response_llm = truncated_text
        elif "?" in response_llm:
            response_llm = "I’m not sure, maybe later."

        max_text_length = 300
        response_llm = response_llm[:max_text_length] if len(response_llm) > max_text_length else response_llm

        return jsonify({
            "ai_response": ai_response,
            "suggested_response": response_llm,
            "language": language,
            "message": "Here's a suggested response. Give it a try!"
        }), 200, cors_headers
    except Exception as e:
        error_msg = f"Error in suggest-response endpoint: {str(e)}"
        print(error_msg)
        return jsonify({
            "error": "Something broke! Can you try that again?",
            "hint": "Check your request data or try later."
        }), 500, cors_headers

@app.route('/suggest-response/', methods=['GET'])
def suggest_response_get():
    return jsonify({
        "error": "Wrong method! Use POST to get suggestions.",
        "hint": "Send a POST request with your data."
    }), 405, cors_headers

def generate_audio(text: str, file_path: str, language: str, topic: str, role: str):
    try:
        role_voice_mapping = {
            "Taxi driver": "en-US-GuyNeural",
            "AI Taxi Driver": "en-US-GuyNeural",
            "Doctor Dreyfus": "en-US-GuyNeural",
            "AI Doctor": "en-US-GuyNeural",
            "Bank employee": "en-US-GuyNeural",
            "Real estate agent": "en-US-GuyNeural",
            "Gym employee": "en-US-GuyNeural",
            "Repairman": "en-US-GuyNeural",
            "Mechanic": "en-US-GuyNeural",
            "Veterinarian": "en-US-GuyNeural",
            "AI Veterinarian": "en-US-GuyNeural",
            "Your boss": "en-US-GuyNeural",
            "Vendor": "en-US-GuyNeural",
            "Company owner": "en-US-GuyNeural",
            "AI Company Owner": "en-US-GuyNeural",
            "HR person": "en-US-GuyNeural",
            "AI HR Person": "en-US-GuyNeural",
            "Company representative": "en-US-GuyNeural",
            "AI Company Representative": "en-US-GuyNeural",
            "Tech support representative": "en-US-GuyNeural",
            "AI Tech Support": "en-US-GuyNeural",
            "HR manager": "en-US-GuyNeural",
            "Travel agent": "en-US-GuyNeural",
            "AI Travel Agent": "en-US-GuyNeural",
            "Sales representative": "en-US-GuyNeural",
            "Hotel manager": "en-US-JennyNeural",
            "AI Hotel Manager": "en-US-JennyNeural",
            "Cashier": "en-US-JennyNeural",
            "AI Cashier": "en-US-JennyNeural",
            "Store employee": "en-US-JennyNeural",
            "AI Store Employee": "en-US-JennyNeural",
            "Waiter": "en-US-JennyNeural",
            "Barista": "en-US-JennyNeural",
            "AI Barista": "en-US-JennyNeural",
            "Salon manager": "en-US-JennyNeural",
            "Dry cleaner clerk": "en-US-JennyNeural",
            "Post office clerk": "en-US-JennyNeural",
            "Policewoman": "en-US-JennyNeural",
            "Consultant": "en-US-JennyNeural",
            "Airline representative": "en-US-JennyNeural",
            "Roommate": "en-US-JennyNeural",
            "Your roommate": "en-US-JennyNeural",
            "Best friend": "en-US-JennyNeural",
            "Neighbor": "en-US-JennyNeural",
            "New employee": "en-US-JennyNeural",
            "Your partner": "en-US-JennyNeural",
            "Intriguing Stranger": "en-US-JennyNeural",
            "Your partner's mother": "en-US-JennyNeural",
            "Your potential partner": "en-US-JennyNeural",
            "AI Potential Partner": "en-US-JennyNeural",
            "Celebrity": "en-US-JennyNeural",
            "Interviewer": "en-US-JennyNeural",
            "Co-worker": "en-US-JennyNeural",
            "Your co-worker": "en-US-JennyNeural",
            "Teacher": "en-US-JennyNeural",
            "Michelle": "en-US-JennyNeural",
            "Your manager": "en-US-JennyNeural",
            "Customer": "en-US-JennyNeural",
            "AI Customer": "en-US-JennyNeural",
            "Sandra": "en-US-JennyNeural",
            "Anita": "en-US-JennyNeural",
            "Customer support": "en-US-JennyNeural",
            "AI Customer Support": "en-US-JennyNeural"
        }

        default_voice = {
            "en": "en-US-AriaNeural"
        }.get(language, "en-US-AriaNeural")

        role_from_topic = topic.split(" - ")[-1] if " - " in topic else role
        voice = role_voice_mapping.get(role_from_topic, default_voice)

        if role_from_topic not in role_voice_mapping:
            print(f"Role {role_from_topic} not found in role_voice_mapping. Using default voice for language {language}: {default_voice}")

        tts = edge_tts.Communicate(text, voice=voice, rate="+50%")
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(tts.save(file_path))
        loop.close()
        print(f"Audio generated with edge_tts at {file_path} using voice {voice}")
    except Exception as e:
        print(f"edge_tts failed: {str(e)}. Falling back to gTTS.")
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(file_path)
        print(f"Audio generated with gTTS at {file_path}")

        audio = AudioSegment.from_file(file_path, format="mp3")
        compressed_path = file_path.replace(".mp3", "_compressed.mp3")
        audio.export(compressed_path, format="mp3", bitrate="32k")
        os.remove(file_path)
        os.rename(compressed_path, file_path)

@app.route('/audio/<audio_file_name>', methods=['GET'])
def get_audio(audio_file_name):
    file_path = os.path.join(audio_dir, audio_file_name)
    if not os.path.exists(file_path):
        print(f"Audio file not found: {audio_file_name}")
        return jsonify({
            "error": "Audio file not found. It might have expired.",
            "hint": "Try generating a new audio response."
        }), 404, cors_headers
    response = make_response(send_file(file_path, mimetype="audio/mpeg"))
    for key, value in cors_headers.items():
        response.headers[key] = value
    return response

if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    app.run(host="127.0.0.1", port=3000, debug=True)