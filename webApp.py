from flask import Flask, render_template, request, jsonify, send_file, make_response
import os
from flask_cors import CORS
import json
import uuid
from langchain.memory import ConversationBufferWindowMemory
from utils import get_response_llm
import edge_tts
from pydub import AudioSegment
from gtts import gTTS
import pandas as pd
import random
import logging
import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": ["*"]}})

# Directory for audio files
audio_dir = os.path.join(os.getcwd(), "audio")
os.makedirs(audio_dir, exist_ok=True)

# Load word pronunciation dataset with absolute path
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

# Load data_topic_en.csv (for topics)
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

# Conversation memory for langchain
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5
)

# Standard CORS headers for all responses
cors_headers = {
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
}

# Routes from webApp.py
rootPath = ''

@app.route(rootPath + '/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route(rootPath + '/getAudioFromText', methods=['POST'])
def get_audio_from_text():
    try:
        event = {'body': json.dumps(request.get_json(force=True))}
        response = lambdaTTS.lambda_handler(event, [])
        if isinstance(response, dict):
            body = response.get('body', '{}')
        else:
            body = response or '{}'
        return jsonify(json.loads(body)), 200, cors_headers
    except Exception as e:
        print(f'Error in /getAudioFromText: {str(e)}')
        return jsonify({'error': str(e)}), 500, cors_headers

@app.route(rootPath + '/getSample', methods=['POST'])
def get_next():
    try:
        data = request.get_json(force=True)
        language = data.get('language', 'en')
        mode = data.get('mode', 'sentence')  # word_pronunciation, sentence, phrasal_verb
        topic = data.get('topic', '')  # Optional for topic-based modes

        logging.info(f"getSample request: language={language}, mode={mode}, topic={topic}")

        if language != "en":
            # Fall back to lambdaGetSample for non-English requests
            event = {'body': json.dumps(data)}
            response = lambdaGetSample.lambda_handler(event, [])
            body = response.get('body', '{}') if isinstance(response, dict) else response or '{}'
            return jsonify(json.loads(body)), 200, cors_headers

        if mode == "word_pronunciation":
            # Use data_word_en.csv for pronunciation
            if data_word_en.empty or "id" not in data_word_en.columns:
                raise Exception("data_word_en.csv is empty or missing 'id' column")
            
            sample = data_word_en.sample(1).iloc[0]
            response = {
                "id": str(sample["id"]),
                "word": sample["word"],
                "ipa": sample["ipa"],
                "ending_sound": sample["ending_sound"],
                "language": language,
                "mode": mode,
                "text": sample["word"]  # Text for pronunciation
            }
        else:
            # Use data_topic_en.csv for sentence or phrasal_verb modes
            if data_topic_en.empty or "word" not in data_topic_en.columns:
                raise Exception("data_topic_en.csv is empty or missing 'word' column")
            
            # Filter by topic if provided
            filtered_data = data_topic_en
            if topic:
                filtered_data = data_topic_en[data_topic_en['word'].str.contains(topic, case=False, na=False) |
                                              data_topic_en['exampleSentence'].str.contains(topic, case=False, na=False)]
                if filtered_data.empty:
                    raise Exception(f"No samples found for topic '{topic}'")

            sample = filtered_data.sample(1).iloc[0]
            sample_id = str(uuid.uuid4()) if 'id' not in sample else str(sample['id'])
            text = sample["exampleSentence"] if mode == "sentence" else sample["word"]

            response = {
                "id": sample_id,
                "word": sample["word"],
                "phonetic": sample["phonetic"] if pd.notna(sample["phonetic"]) else "",
                "exampleSentence": sample["exampleSentence"] if pd.notna(sample["exampleSentence"]) else "",
                "language": language,
                "mode": mode,
                "text": text
            }

        return jsonify(response), 200, cors_headers

    except Exception as e:
        logging.error(f'Error in /getSample: {str(e)}')
        return jsonify({'error': str(e)}), 500, cors_headers

@app.route("/GetAccuracyFromRecordedAudio", methods=["POST", "OPTIONS"])
def get_accuracy_from_recorded_audio():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    try:
        request_data = request.get_json(force=False)
        if request_data is None:
            logging.error("Invalid or missing JSON payload")
            return _corsify_actual_response(
                jsonify({"error": "Invalid or missing JSON payload"})
            ), 400
        
        logging.info(f"Received request data: {request_data}")
        recorded_audio = request_data.get("recorded_audio") or request_data.get("base64Audio")
        text = request_data.get("text") or request_data.get("title")
        language = request_data.get("language", "en")
        
        if not recorded_audio or not text:
            logging.error(f"Missing fields: recorded_audio/base64Audio={bool(recorded_audio)}, text/title={bool(text)}")
            return _corsify_actual_response(
                jsonify({"error": "Missing recorded_audio/base64Audio or text/title"})
            ), 400
        
        # Validate base64 audio
        import base64
        try:
            audio_data = recorded_audio
            if ',' in recorded_audio:
                audio_data = recorded_audio.split(',')[1]
            base64.b64decode(audio_data)
        except Exception as e:
            logging.error(f"Invalid base64 audio: {str(e)}")
            return _corsify_actual_response(
                jsonify({"error": f"Invalid base64-encoded audio: {str(e)}"})
            ), 400

        # Construct event for Lambda
        event = {
            "body": json.dumps({
                "base64Audio": recorded_audio,
                "text": text,  # Pass text directly
                "language": language
            })
        }
        context = {}
        
        response = lambdaSpeechToScore.lambda_handler(event, context)
        response_data = json.loads(response['body'])
        
        if response['statusCode'] != 200:
            logging.error(f"Lambda error: {response_data.get('error')}")
            return _corsify_actual_response(
                jsonify({"error": response_data.get("error", "Unknown error")})
            ), response['statusCode']
        
        logging.info(f"Successful response: {response_data}")
        return _corsify_actual_response(jsonify(response_data))
    except Exception as e:
        logging.error(f"Error processing accuracy request: {str(e)}")
        return _corsify_actual_response(
            jsonify({"error": f"Server error: {str(e)}"})
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
        role = data.get('role', 'AI conversational teacher').strip()

        print(f"Received topic: {topic}, role: {role}, language: {language}")

        if not topic:
            return jsonify({"error": "Thiếu hoặc chủ đề trống"}), 400, cors_headers

        if user_text.lower() in ["fun", "interesting", "decide"]:
            suggestion_prompt = {
                "fun": f"Act as a {role} in {language}. Suggest a fun question (max 10 words) based on the topic '{topic}' to engage the user.",
                "interesting": f"Act as a {role} in {language}. Suggest an interesting question (max 10 words) based on the topic '{topic}' to engage the user.",
                "decide": f"Act as a {role} in {language}. Suggest a random question (max 10 words) based on the topic '{topic}' to engage the user."
            }.get(user_text.lower(), None)
            if suggestion_prompt:
                response_llm = get_response_llm(suggestion_prompt, memory, topic, language, role)
            else:
                response_llm = get_response_llm(user_text, memory, topic, language, role)
        elif not user_text or len(user_text.split()) <= 2:
            initial_prompt = f"Act as a {role} on the topic '{topic}' in {language}. Ask a relevant question (max 10 words) based on the topic to start or continue the conversation."
            response_llm = get_response_llm(initial_prompt, memory, topic, language, role)
        else:
            response_llm = get_response_llm(user_text, memory, topic, language, role)

        memory.save_context({'input': user_text or 'Start conversation'}, {'output': response_llm})

        max_text_length = 300
        response_llm = response_llm[:max_text_length] if len(response_llm) > max_text_length else response_llm

        audio_filename = f"audio_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)
        generate_audio(response_llm, audio_path, language, topic, role)

        return jsonify({
            "user_text": user_text,
            "ai_response": response_llm,
            "audio_url": f"/audio/{audio_filename}",
            "language": language
        }), 200, cors_headers

    except Exception as e:
        error_msg = f"Error in text-and-respond endpoint: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500, cors_headers

@app.route('/suggest-response/', methods=['POST'])
def suggest_response():
    try:
        data = request.get_json(force=True)
        print(f"Request body: {data}")
        ai_response = data.get('ai_response', '').strip()
        topic = data.get('topic', '').strip()
        language = data.get('language', 'en').strip()
        role = data.get('role', 'AI conversational teacher').strip()
        suggest_for = data.get('suggest_for', 'user').strip()

        if not topic:
            return jsonify({"error": "Thiếu hoặc chủ đề trống"}), 400, cors_headers

        if not ai_response:
            return jsonify({"error": "Thiếu ai_response để gợi ý"}), 400, cors_headers

        if suggest_for.lower() not in ['user', 'ai']:
            return jsonify({"error": "suggest_for must be 'user' or 'ai'"}), 400, cors_headers

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

        topic_context = f"The topic '{topic}' involves a scenario where the user is interacting with a {role}. "

        if suggest_for.lower() == 'user':
            suggestion_prompt = (
                f"You are a {role} in {language}. The AI has asked: '{ai_response}'. "
                f"Suggest a concise response (max 10 words) that the user can say to answer this question. "
                f"{topic_context}"
                f"{question_type_hint} "
                f"Do not continue the conversation or ask a new question. "
                f"Focus on providing a direct, natural response for the user."
            )
            response_llm = get_response_llm(suggestion_prompt, None, topic, language, role)
        else:
            global memory
            if memory is None:
                print("Memory is None, reinitializing...")
                memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=5
                )
            memory.save_context({'input': 'AI'}, {'output': ai_response})
            suggestion_prompt = (
                f"Act as a {role} in {language}. Based on the AI's last message '{ai_response}', "
                f"suggest a relevant response (max 10 words) for the topic '{topic}'. "
                f"{topic_context}"
            )
            response_llm = get_response_llm(suggestion_prompt, memory, topic, language, role)

        response_words = response_llm.split()
        if len(response_words) > 10:
            response_llm = " ".join(response_words[:10])
        if "?" in response_llm:
            response_llm = "I’m not sure, maybe later."

        max_text_length = 300
        response_llm = response_llm[:max_text_length] if len(response_llm) > max_text_length else response_llm

        return jsonify({
            "ai_response": ai_response,
            "suggested_response": response_llm,
            "language": language
        }), 200, cors_headers
    except Exception as e:
        error_msg = f"Error in suggest-response endpoint: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500, cors_headers

@app.route('/suggest-response/', methods=['GET'])
def suggest_response_get():
    return jsonify({"error": "Method not allowed. Use POST to suggest a response."}), 405, cors_headers

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
            "en": "en-US-AriaNeural",
            "vi": "vi-VN-AnNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "fr": "fr-FR-DeniseNeural",
            "es": "es-ES-ElviraNeural",
            "it": "it-IT-ElsaNeural",
            "pt": "pt-BR-FranciscaNeural",
            "ru": "ru-RU-SvetlanaNeural"
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
        return jsonify({"error": "File âm thanh không tồn tại"}), 404, cors_headers

    response = make_response(send_file(file_path, mimetype="audio/mpeg"))
    for key, value in cors_headers.items():
        response.headers[key] = value
    return response

if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    app.run(host="127.0.0.1", port=3000, debug=True)