import torch
import torch.nn as nn
import pickle
from ModelInterfaces import IASRModel
from AIModels import NeuralASR 
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import ModelInterfaces as mi

class WhisperASRModel(mi.IASRModel):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.audio = None
        self.transcript = ""
        self.word_locations = []

    def processAudio(self, audio):
        self.audio = audio
        inputs = self.processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        generated_ids = self.model.generate(inputs["input_features"], output_attentions=False)
        self.transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.word_locations = self._get_word_locations()

    def getTranscript(self):
        return self.transcript

    def getWordLocations(self):
        return self.word_locations

    def _get_word_locations(self):
        return [{"start_ts": 0, "end_ts": 1000} for _ in self.transcript.split()]

def getASRModel(language: str, use_whisper: bool = True, **kwargs) -> IASRModel:
    if use_whisper:
        model_name = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, **kwargs)
        return WhisperASRModel(model, processor)
    
    if language == 'de':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='de',
                                               device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)

    elif language == 'en':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en',
                                               device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)
    elif language == 'fr':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='fr',
                                               device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)
    else:
        raise ValueError('Language not implemented')

def getTTSModel(language: str) -> nn.Module:
    if language == 'de':
        speaker = 'thorsten_v2'
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)
    elif language == 'en':
        speaker = 'lj_16khz'
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)
    else:
        raise ValueError('Language not implemented')
    return model

# def getTranslationModel(language: str) -> nn.Module:
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    if language == 'de':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        with open('translation_model_de.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_de.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
    else:
        raise ValueError('Language not implemented')
    return model, tokenizer