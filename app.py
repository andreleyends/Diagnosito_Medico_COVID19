import os
import numpy as np
import streamlit as st
import joblib
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_covid.pkl')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')

st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
    .result-positive {
        background-color: #ffe0e0;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #cc0000;
    }
    .result-negative {
        background-color: #e0ffe0;
        border: 2px solid #4bff4b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #006600;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #0e1117;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        modelo_data = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        raise
    log_reg = modelo_data['log_reg']
    feature_extractor = modelo_data['feature_extractor']
    feature_extractor.eval()
    return log_reg, feature_extractor, modelo_data


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


def main():
    st.title("🩺 COVID-19 X-ray Classifier")
    st.markdown("**Clasificación binaria de radiografías usando Regresión Logística + CNN**")
    st.markdown("---")

    # Cargar modelo
    model_data = load_model()
    if model_data is None:
        st.error(
            "⚠️ No se encontró el modelo entrenado. "
            "Ejecuta primero `python entrenar_modelo.py` para generar el archivo `modelo_covid.pkl`."
        )
        return

    log_reg, feature_extractor, modelo_info = model_data

    # Sidebar con info del modelo
    with st.sidebar:
        st.header("📊 Información del Modelo")
        st.markdown(f"**Algoritmo:** Regresión Logística")
        st.markdown(f"**Extractor de features:** MobileNetV2")
        st.markdown(f"**ROC-AUC:** {modelo_info.get('roc_auc', 0):.4f}")
        st.markdown(f"**Accuracy:** {modelo_info.get('accuracy', 0):.4f}")
        st.markdown("---")
        st.markdown("**Clases:**")
        st.markdown("- 🟢 COVID Negativo (Ausencia)")
        st.markdown("- 🔴 COVID Positivo (Presencia)")

    # Mostrar métricas del modelo
    st.subheader("📈 Métricas del Modelo Entrenado")
    col1, col2, col3, col4 = st.columns(4)

    metrics_path = os.path.join(RESULTS_DIR, 'metricas_reporte.txt')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            content = f.read()

        def extract_metric(text, label):
            for line in text.split('\n'):
                stripped = line.strip()
                if stripped.startswith(label + ':'):
                    try:
                        val = stripped.split(':')[1].strip().split()[0]
                        return float(val)
                    except (ValueError, IndexError):
                        pass
            return 0.0

        with col1:
            st.metric("Accuracy", f"{extract_metric(content, 'Accuracy'):.1%}")
        with col2:
            st.metric("Precision", f"{extract_metric(content, 'Precision'):.1%}")
        with col3:
            st.metric("Recall", f"{extract_metric(content, 'Recall'):.1%}")
        with col4:
            st.metric("F1-Score", f"{extract_metric(content, 'F1-Score'):.1%}")
    else:
        st.warning("No se encontraron métricas detalladas. Mostrando ROC-AUC del modelo.")
        with col1:
            st.metric("ROC-AUC", f"{modelo_info.get('roc_auc', 0):.1%}")
        with col2:
            st.metric("Accuracy", f"{modelo_info.get('accuracy', 0):.1%}")
        with col3:
            st.metric("Imágenes", modelo_info.get('total', 'N/A'))
        with col4:
            st.metric("Clases", len(modelo_info.get('clases', [])))

    st.markdown("---")

    # Subir imagen
    st.subheader("🔍 Subir Radiografía para Diagnóstico")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de radiografía (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Sube una radiografía de tórax para clasificar COVID-19 positivo o negativo"
    )

    if uploaded_file is not None:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            image = Image.open(uploaded_file)
            st.image(image, caption="Radiografía subida", use_container_width=True)

        with col_result:
            st.markdown("### Resultado del Diagnóstico")

            try:
                with st.spinner("Analizando radiografía con IA..."):
                    img_tensor = preprocess_image(image)
                    features = extract_features(feature_extractor, img_tensor)
                    prediction = log_reg.predict(features)[0]
                    probability = log_reg.predict_proba(features)[0]

                    prob_covid = float(probability[1])
                    prob_no_covid = float(probability[0])
            except Exception as e:
                st.error(f"⚠️ Ocurrió un error al procesar la imagen: {e}")
                return

            if prediction == 1:
                st.markdown(
                    '<div class="result-positive">'
                    '🔴 COVID-19 POSITIVO<br>'
                    '<span style="font-size:16px;">Presencia de condición detectada</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-negative">'
                    '🟢 COVID-19 NEGATIVO<br>'
                    '<span style="font-size:16px;">Ausencia de condición detectada</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")
            st.markdown("#### Probabilidades")
            st.progress(prob_covid)
            st.markdown(f"**COVID Positivo:** {prob_covid:.2%}")
            st.progress(prob_no_covid)
            st.markdown(f"**COVID Negativo:** {prob_no_covid:.2%}")

            st.markdown("")
            st.info(
                f"**Umbral de decisión:** 0.50\n\n"
                f"La regresión logística asigna una probabilidad. "
                f"Si P(positivo) ≥ 0.5, se clasifica como **Positivo**."
            )

    st.markdown("---")
    st.markdown(
        "**Nota:** Este sistema es una herramienta de apoyo diagnóstico. "
        "No sustituye el criterio de un profesional de salud."
    )


if __name__ == '__main__':
    main()
