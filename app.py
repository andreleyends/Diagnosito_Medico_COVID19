import os
import io
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

import torch
import torchvision.transforms as transforms
from torchvision import models

from logistic_regression import LogisticRegressionManual

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_covid.pkl')

app = Flask(__name__)


# ---------------- Carga del modelo ----------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    modelo_data = joblib.load(MODEL_PATH)
    model = modelo_data['model']          # LogisticRegressionManual
    feature_extractor = modelo_data['feature_extractor']
    feature_extractor.eval()
    return model, feature_extractor, modelo_data


# Cargar al importar la app (una sola vez, para gunicorn)
MODEL, FEATURE_EXTRACTOR, MODEL_INFO = load_model()


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def preprocess_image(image):
    img_rgb = image.convert('RGB')
    img_tensor = TRANSFORM(img_rgb)
    return img_tensor.unsqueeze(0)


def extract_features(model, img_tensor):
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy()


def predict_image(image):
    """Devuelve (prediction: 0 o 1, prob_covid: float)."""
    img_tensor = preprocess_image(image)
    features = extract_features(FEATURE_EXTRACTOR, img_tensor)
    # La regresion manual devuelve P(y=1) como array 1D
    prob_covid = float(MODEL.predict_proba(features)[0])
    prediction = int((prob_covid >= 0.5))
    return prediction, prob_covid


# ---------------- HTML (pagina principal) ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COVID-19 X-ray Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f6;
            color: #0e1117;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .card {
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            padding: 40px;
            max-width: 720px;
            width: 100%;
        }
        h1 { text-align: center; font-size: 28px; margin-bottom: 4px; }
        .subtitle { text-align: center; color: #566073; margin-bottom: 28px; font-size: 15px; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 28px;
        }
        .metric {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        .metric .label { font-size: 13px; color: #566073; }
        .metric .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
        .dropzone {
            border: 2px dashed #9aa4b2;
            border-radius: 12px;
            padding: 36px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            background: #fafbfc;
        }
        .dropzone:hover { border-color: #4b8bff; }
        .dropzone p { color: #566073; font-size: 15px; }
        #fileInput { display: none; }
        .preview { max-width: 100%; max-height: 320px; margin-top: 16px; border-radius: 8px; display: none; }
        .btn {
            margin-top: 16px;
            background: #4b8bff;
            color: #fff;
            border: none;
            padding: 12px 28px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover { background: #3b74d9; }
        .btn:disabled { background: #9aa4b2; cursor: not-allowed; }
        #result {
            margin-top: 24px;
            display: none;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        }
        #result .answer { font-size: 26px; font-weight: 700; margin-bottom: 8px; }
        #result .detail { font-size: 15px; margin-bottom: 16px; }
        .positive { background: #ffe0e0; border: 2px solid #ff4b4b; color: #cc0000; }
        .negative { background: #e0ffe0; border: 2px solid #4bff4b; color: #006600; }
        .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .bar-label { width: 110px; text-align: right; font-size: 14px; color: #333; }
        .bar-track { flex: 1; background: #e9ebef; border-radius: 6px; height: 20px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 6px; }
        .bar-pos { background: #ff4b4b; }
        .bar-neg { background: #31b04f; }
        .bar-pct { width: 60px; text-align: left; font-size: 14px; font-weight: 600; }
        .note {
            margin-top: 28px;
            background: #f0f2f6;
            border-left: 4px solid #566073;
            padding: 14px 18px;
            border-radius: 6px;
            font-size: 13px;
            color: #566073;
        }
        .loading { text-align: center; font-size: 15px; color: #566073; margin-top: 16px; display: none; }
        .error { margin-top: 16px; background: #ffe0e0; color: #cc0000; padding: 12px; border-radius: 8px; display: none; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h1>&#129657; COVID-19 X-ray Classifier</h1>
        <p class="subtitle">Clasificacion binaria de radiografias usando Regresion Logistica + CNN
            <br>(regresion logistica implementada desde cero, sin librerias ML)</p>

        <div class="metrics">
            <div class="metric"><div class="label">Accuracy</div><div class="value">{{ accuracy }}</div></div>
            <div class="metric"><div class="label">Precision</div><div class="value">{{ precision }}</div></div>
            <div class="metric"><div class="label">Recall</div><div class="value">{{ recall }}</div></div>
            <div class="metric"><div class="label">F1-Score</div><div class="value">{{ f1 }}</div></div>
        </div>

        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
            <p>&#128065; Haz clic o arrastra una radiografia de torax aqui<br>
            <small>PNG, JPG, JPEG, BMP o TIFF</small></p>
        </div>
        <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.bmp,.tiff,image/*">
        <img class="preview" id="preview" alt="Preview">

        <div class="loading" id="loading">Analizando radiografia con IA&#8230;</div>
        <div class="error" id="error"></div>
        <button class="btn" id="analyzeBtn" onclick="analyze()" disabled>Analizar radiografia</button>

        <div id="result"></div>

        <div class="note">
            <strong>Nota:</strong> Este sistema es una herramienta de apoyo diagnostico.
            No sustituye el criterio de un profesional de salud.
        </div>
    </div>

<script>
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let selectedFile = null;

    fileInput.addEventListener('change', () => {
        selectedFile = fileInput.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(selectedFile);
            analyzeBtn.disabled = false;
            document.getElementById('result').style.display = 'none';
        }
    });

    async function analyze() {
        if (!selectedFile) return;
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const result = document.getElementById('result');
        loading.style.display = 'block';
        error.style.display = 'none';
        result.style.display = 'none';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const resp = await fetch('/predict', { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || 'Error al procesar');
            result.innerHTML = buildResult(data);
            result.style.display = 'block';
        } catch (err) {
            error.textContent = 'Error: ' + err.message;
            error.style.display = 'block';
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    }

    function buildResult(data) {
        const isPos = data.prediction === 1;
        const cls = isPos ? 'positive' : 'negative';
        const icon = isPos ? '&#128308;' : '&#128994;';
        const title = isPos ? 'COVID-19 POSITIVO' : 'COVID-19 NEGATIVO';
        const sub = isPos ? 'Presencia de condicion detectada' : 'Ausencia de condicion detectada';
        const pPos = (data.prob_covid * 100).toFixed(1);
        const pNeg = (data.prob_no_covid * 100).toFixed(1);
        return `
            <div class="${cls}">
                <div class="answer">${icon} ${title}</div>
                <div class="detail">${sub}</div>
                <div class="bar-row"><span class="bar-label">Positivo (COVID)</span>
                    <div class="bar-track"><div class="bar-fill bar-pos" style="width:${pPos}%"></div></div>
                    <span class="bar-pct">${pPos}%</span></div>
                <div class="bar-row"><span class="bar-label">Negativo (Normal)</span>
                    <div class="bar-track"><div class="bar-fill bar-neg" style="width:${pNeg}%"></div></div>
                    <span class="bar-pct">${pNeg}%</span></div>
                <div style="margin-top:12px;font-size:13px;">Umbral de decision: 0.50 &middot; Si P(positivo) &ge; 0.5 se clasifica como Positivo</div>
            </div>`;
    }
</script>
</body>
</html>
"""


# ---------------- Rutas ----------------
@app.route('/', methods=['GET'])
def index():
    if MODEL_INFO is None:
        return "<h3>Modelo no encontrado. Ejecuta entrenar_modelo.py primero.</h3>", 500

    def pct(v):
        return f"{float(v)*100:.1f}%"

    return render_template_string(
        HTML_PAGE,
        accuracy=pct(MODEL_INFO.get('accuracy', 0)),
        precision=pct(MODEL_INFO.get('precision', 0)),
        recall=pct(MODEL_INFO.get('recall', 0)),
        f1=pct(MODEL_INFO.get('f1', 0)),
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envio una imagen'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacio'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Imagen invalida: {e}'}), 400

    try:
        prediction, prob_covid = predict_image(image)
        prob_no_covid = 1.0 - prob_covid
        return jsonify({
            'prediction': prediction,
            'prob_covid': prob_covid,
            'prob_no_covid': prob_no_covid,
            'clase': 'covid_positive' if prediction == 1 else 'covid_negative'
        })
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {e}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
