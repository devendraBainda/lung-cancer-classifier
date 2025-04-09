from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('Lung_cancer_prediction.keras')
class_names = ["Lung Benign Tissue", "Lung Squamous Cell Carcinoma", "Lung Adenocarcinoma"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_and_collect(filepaths):
    predictions = []
    for filepath in filepaths:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][predicted_class]) * 100
        predictions.append((predicted_class, confidence))
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    filepaths = []
    filenames = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)
            filenames.append(filename)
        else:
            return jsonify({'error': f'Invalid file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    predictions = predict_and_collect(filepaths)

    results = []
    for filepath, filename, (pred_idx, confidence) in zip(filepaths, filenames, predictions):
        predicted_label = class_names[pred_idx]
        results.append({
            'filepath': filepath,
            'filename': filename,
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })

    return render_template('results.html', results=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
