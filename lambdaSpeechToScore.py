import torch
import json
import os
import WordMatching as wm
import utilsFileIO
import pronunciationTrainer
import base64
import time
import audioread
import numpy as np
from torchaudio.transforms import Resample
import io
import tempfile
import pandas as pd
import logging

TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR

trainer_SST_lambda = {}
trainer_SST_lambda['en'] = pronunciationTrainer.getTrainer("en")

transform = Resample(orig_freq=48000, new_freq=16000)

# Set up logging
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_DATA_PATH = os.path.join(BASE_DIR, "./databases/data_word_en.csv")
logging.info(f"Loading data_word_en.csv from: {WORD_DATA_PATH}")
try:
    word_data = pd.read_csv(WORD_DATA_PATH)
    logging.info(f"Loaded data_word_en.csv with columns: {word_data.columns.tolist()}")
    if word_data.empty:
        logging.warning("data_word_en.csv is empty")
except FileNotFoundError:
    logging.error(f"data_word_en.csv not found at {WORD_DATA_PATH}")
    word_data = pd.DataFrame()
except Exception as e:
    logging.error(f"Error loading data_word_en.csv: {str(e)}")
    word_data = pd.DataFrame()

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])
        recorded_audio = data.get('base64Audio') or data.get('recorded_audio')
        text = data.get('text')
        language = data.get('language', 'en')

        if language != 'en':
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                },
                'body': json.dumps({'error': 'Only English (en) is supported'})
            }

        if not recorded_audio or not text:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                },
                'body': json.dumps({'error': 'Missing recorded_audio/base64Audio or text'})
            }

        # Use provided text directly
        real_text = text.strip()
        logging.info(f"Processing text: {real_text}")

        # Lookup IPA from data_word_en.csv if text is a single word
        real_ipa = None
        if len(real_text.split()) == 1:
            word_info = word_data[word_data['word'].str.lower() == real_text.lower()]
            if not word_info.empty:
                real_ipa = word_info.iloc[0]['ipa']
                logging.info(f"Found IPA for '{real_text}': {real_ipa}")
            else:
                logging.warning(f"No IPA found for '{real_text}' in data_word_en.csv")
        else:
            logging.info("Text is multi-word; skipping IPA lookup")

    except Exception as e:
        logging.error(f"Error parsing request: {str(e)}")
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': True,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': f"Invalid request: {str(e)}"})
        }

    tmp_name = os.path.join(TEMP_DIR, f"audio_{int(time.time() * 1000)}.ogg")
    try:
        # Handle data URL format
        audio_data = recorded_audio
        if ',' in recorded_audio:
            audio_data = recorded_audio.split(',')[1]
        with open(tmp_name, "wb") as tmp_file:
            tmp_file.write(base64.b64decode(audio_data))
        signal, fs = audioread_load(tmp_name)
    except Exception as e:
        logging.error(f"Error processing audio file {tmp_name}: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': 'true',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': f"Failed to process audio file: {str(e)}"})
        }
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception as e:
                logging.error(f"Error deleting temp file {tmp_name}: {str(e)}")

    signal = torch.tensor(signal, dtype=torch.float32)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    if signal.shape[0] != 1:
        signal = signal[0:1, :]
    signal = transform(signal)

    try:
        result = trainer_SST_lambda[language].processAudioForGivenText(signal, real_text)
        logging.info(f"Raw result from processAudioForGivenText: {result}")
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': 'true',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': f"Failed to process audio: {str(e)}"})
        }

    start = time.time()
    real_transcripts_ipa = ' '.join([word[0] for word in result['real_and_transcribed_words_ipa']])
    matched_transcripts_ipa = ' '.join([word[1] for word in result['real_and_transcribed_words_ipa']])
    real_transcripts = ' '.join([word[0] for word in result['real_and_transcribed_words']])
    matched_transcripts = ' '.join([word[1] for word in result['real_and_transcribed_words']])
    logging.info(f"Real transcripts: {real_transcripts}")
    logging.info(f"Matched transcripts: {matched_transcripts}")
    logging.info(f"Real IPA: {real_transcripts_ipa}")
    logging.info(f"Matched IPA: {matched_transcripts_ipa}")

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):
        mapped_letters, mapped_letters_indices = wm.get_best_mapped_words(mapped_words[idx], word_real)
        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(word_real, mapped_letters)
        is_letter_correct_all_words += ''.join([str(int(is_correct)) for is_correct in is_letter_correct]) + ' '

    pair_accuracy_category = ' '.join([str(category) for category in result['pronunciation_categories']])
    logging.info(f"Time to post-process results: {time.time() - start}")

    accuracy = float(result['pronunciation_accuracy'])
    print(accuracy)
    logging.info(f"Before adjustment - Pronunciation accuracy: {accuracy}")
    accuracy = max(0, accuracy)
    logging.info(f"After adjustment - Pronunciation accuracy: {accuracy}")

    ipa_accuracy = {word[0]: 1.0 if word[0] == word[1] else 0.0 for word in result['real_and_transcribed_words_ipa']}

    res = {
        'accuracy': accuracy,
        'ipa_accuracy': ipa_accuracy,
        'ipa_transcript': matched_transcripts_ipa,
        'pronunciation_accuracy': str(accuracy),
        'real_transcripts_ipa': real_transcripts_ipa,
        'matched_transcripts_ipa': matched_transcripts_ipa,
        'is_letter_correct_all_words': is_letter_correct_all_words.strip(),
        'pair_accuracy_category': pair_accuracy_category,
        'start_time': result['start_time'],
        'end_time': result['end_time']
    }

    logging.info(f"Response sent to client: {res}")

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(res)
    }

def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = int(np.round(sr_native * offset)) * n_channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)
        n = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)
            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[: s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[(s_start - n_prev):]
            y.append(frame)
    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)
    return y, sr_native

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))
    fmt = "<i{:d}".format(n_bytes)
    return scale * np.frombuffer(x, fmt).astype(dtype)