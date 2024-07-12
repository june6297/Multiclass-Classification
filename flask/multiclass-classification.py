from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Emotion Prediction API',
    description='API for predicting emotions from images and videos')

# 모델 및 프로세서 로드
model_name = "facebook/convnext-tiny-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained("./model_classification_5")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 라벨 정의
labels = ['anger', 'happy', 'neutral', 'panic', 'sad']

def preprocess_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.convert('L')
    image = image.resize((224, 224))
    image = np.repeat(np.array(image)[:, :, np.newaxis], 3, axis=2)
    return image

def predict(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return labels[predicted_class_idx]

def process_image(file_content):
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)
    emotion = predict(processed_image)
    return jsonify({'emotion': emotion})

def process_video(file_content):
    temp_path = 'temp_video.mp4'
    with open(temp_path, 'wb') as f:
        f.write(file_content)
    
    cap = cv2.VideoCapture(temp_path)
    
    emotion_counts = {label: 0 for label in labels}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_image(frame)
        emotion = predict(processed_frame)
        emotion_counts[emotion] += 1
    
    cap.release()
    os.remove(temp_path)
    
    return jsonify({'emotion_counts': emotion_counts})

# Swagger 모델 정의
emotion_model = api.model('Emotion', {
    'emotion': fields.String(description='Predicted emotion')
})

emotion_counts_model = api.model('EmotionCounts', {
    'emotion_counts': fields.Nested(api.model('Counts', {
        'anger': fields.Integer,
        'happy': fields.Integer,
        'neutral': fields.Integer,
        'panic': fields.Integer,
        'sad': fields.Integer
    }))
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

# API 라우트 정의
@api.route('/predict')
class PredictEmotion(Resource):
    @api.expect(api.parser().add_argument('file', location='files', type='file', required=True))
    @api.response(200, 'Success (Image)', emotion_model)
    @api.response(200, 'Success (Video)', emotion_counts_model)
    @api.response(400, 'Bad Request', error_model)
    def post(self):
        """Predict emotion from image or video"""
        file = None
        for key in request.files:
            file = request.files[key]
            break

        if not file:
            return jsonify({'error': 'No file provided'}), 400

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        _, file_extension = os.path.splitext(file.filename)
        file_extension = file_extension.lower()

        file_content = file.read()

        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            return process_image(file_content)
        elif file_extension in ['.mp4', '.avi', '.mov']:
            return process_video(file_content)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/')
def home():
    return "Welcome to Emotion Prediction API. Please use /predict endpoint for predictions."

@app.route('/docs')
def get_docs():
    return api.__schema__

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10005, debug=True)