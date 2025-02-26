import openai
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import cv2
import requests
import yaml

class ModelLoader:
    def __init__(self, config_path="config/api_keys.yaml"):
        with open(config_path, 'r') as file:
            self.api_keys = yaml.safe_load(file)

        # Initialize models and APIs
        self.init_openai_api()
        self.init_huggingface_models()
        self.init_opencv()

    def init_openai_api(self):
        openai.api_key = self.api_keys['openai_api_key']

    def init_huggingface_models(self):
        self.nlp_model = pipeline('text-classification', model='bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vision_model = AutoModelForSequenceClassification.from_pretrained('resnet-50')

    def init_opencv(self):
        self.cv_model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    def get_openai_response(self, prompt):
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=50)
        return response.choices[0].text.strip()

    def get_huggingface_nlp_response(self, text):
        return self.nlp_model(text)

    def get_huggingface_vision_response(self, image_path):
        image = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.cv_model.setInput(blob)
        detections = self.cv_model.forward()
        return detections

    def get_google_speech_to_text(self, audio_path):
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_keys['google_speech_to_text_api_key']}"
        headers = {"Content-Type": "application/json"}
        audio_data = open(audio_path, "rb").read()
        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US"
            },
            "audio": {
                "content": audio_data.decode("base64")
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def get_google_text_to_speech(self, text):
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_keys['google_text_to_speech_api_key']}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-D"},
            "audioConfig": {"audioEncoding": "MP3"}
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.content