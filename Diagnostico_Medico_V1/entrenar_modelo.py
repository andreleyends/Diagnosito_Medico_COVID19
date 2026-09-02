import os
import sys
import numpy as np
import joblib

try:
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models
try:
    from torchvision.models import MobileNet_V2_Weights
except ImportError:
    try:
        from torchvision.models.mobilenet_v2 import MobileNet_V2_Weights
    except ImportError:
        MobileNet_V2_Weights = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados')
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_covid.pkl')

os.makedirs(RESULTS_DIR, exist_ok=True)

CLASES = ['covid_negative', 'covid_positive']
IMG_SIZE = 224
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


def get_feature_extractor():
    if MobileNet_V2_Weights is not None:
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_images_from_folder(folder_path):
    images = []
    labels = []
    transform = get_transform()

    for label_idx, clase in enumerate(CLASES):
        clase_dir = os.path.join(folder_path, clase)
        if not os.path.exists(clase_dir):
            print(f"  Carpeta no encontrada: {clase_dir}")
            continue

        archivos = [f for f in os.listdir(clase_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        print(f"  Cargando {len(archivos)} imágenes de {clase}...")

        for archivo in archivos:
            img_path = os.path.join(clase_dir, archivo)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(label_idx)
            except Exception as e:
                print(f"    Error cargando {img_path}: {e}")

    if len(images) == 0:
        raise ValueError(
            f"No se cargaron imágenes desde {folder_path}. "
            f"Verifica que las carpetas existan y contengan imágenes."
        )

    return torch.stack(images), np.array(labels)


def extract_features(model, images_tensor):
    all_features = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(images_tensor), BATCH_SIZE):
            batch = images_tensor[i:i+BATCH_SIZE].to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())
            processed = min(i + BATCH_SIZE, len(images_tensor))
            print(f"    Features extraídas: {processed}/{len(images_tensor)}")

    return np.concatenate(all_features, axis=0)


def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Negativo', 'Positivo'],
           yticklabels=['Negativo', 'Positivo'],
           xlabel='Predicción', ylabel='Real',
           title='Matriz de Confusión')
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada en: {save_path}")
    return cm


def save_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'Curva ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Clasificador aleatorio')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    ax.set_title('Curva ROC - Regresión Logística + CNN')
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Curva ROC guardada en: {save_path}")
    return roc_auc


def save_metrics_report(cm, accuracy, precision, recall, f1, roc_auc, save_path):
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    report = f"""
{'='*60}
   REPORTE DE MÉTRICAS - COVID-19 X-ray Classifier
{'='*60}

MODELO: Regresión Logística + MobileNetV2 (Feature Extractor)

--- Matriz de Confusión ---
  True Negatives  (VN): {tn}
  False Positives (FP): {fp}
  False Negatives (FN): {fn}
  True Positives  (VP): {tp}

--- Métricas ---
  Accuracy:    {accuracy:.4f}  ({accuracy*100:.2f}%)
  Precision:   {precision:.4f}  ({precision*100:.2f}%)
  Recall:      {recall:.4f}  ({recall*100:.2f}%)
  F1-Score:    {f1:.4f}  ({f1*100:.2f}%)
  ROC-AUC:     {roc_auc:.4f}  ({roc_auc*100:.2f}%)

--- Interpretación Clínica ---
  Sensibilidad (Recall): {recall*100:.2f}% → Capacidad de detectar COVID positivos
  Especificidad: {specificity:.2f}% → Capacidad de detectar COVID negativos
  Precisión: {precision*100:.2f}% → De los predichos positivos, cuántos lo son realmente

{'='*60}
"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Reporte guardado en: {save_path}")
    print(report)


def main():
    print("="*60)
    print("  ENTRENAMIENTO: COVID-19 X-ray Classifier")
    print("  Regresión Logística + MobileNetV2 CNN")
    print("="*60)

    # 1. Cargar modelo CNN
    print("\n[1/6] Cargando MobileNetV2 pre-entrenado...")
    feature_extractor = get_feature_extractor()
    print("  Modelo cargado exitosamente.")

    # 2. Cargar imágenes de entrenamiento
    print("\n[2/6] Cargando imágenes de entrenamiento...")
    train_images, train_labels = load_images_from_folder(TRAIN_DIR)
    print(f"  Total imágenes de entrenamiento: {len(train_labels)}")
    print(f"  COVID Negativos: {np.sum(train_labels == 0)}")
    print(f"  COVID Positivos: {np.sum(train_labels == 1)}")

    # 3. Extraer features
    print("\n[3/6] Extrayendo features con CNN (esto puede tardar)...")
    train_features = extract_features(feature_extractor, train_images)
    print(f"  Features extraídas: {train_features.shape}")

    # 4. Entrenar Regresión Logística
    print("\n[4/6] Entrenando Regresión Logística...")
    log_reg = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42
    )
    log_reg.fit(train_features, train_labels)
    print("  Modelo entrenado exitosamente.")

    # Guardar modelo inmediatamente después de entrenar (resiliente a errores posteriores)
    modelo_data = {
        'log_reg': log_reg,
        'feature_extractor': feature_extractor.cpu(),
        'clases': CLASES,
        'img_size': IMG_SIZE,
        'roc_auc': 0.0,
        'accuracy': 0.0,
    }
    joblib.dump(modelo_data, MODEL_PATH)
    print(f"  Modelo (parcial) guardado en: {MODEL_PATH}")

    # Guardar coeficientes más relevantes
    coefs = log_reg.coef_[0]
    top_pos_idx = np.argsort(coefs)[-10:][::-1]
    top_neg_idx = np.argsort(coefs)[:10]
    print(f"\n  Top 10 features más influyentes para POSITIVO:")
    for idx in top_pos_idx:
        print(f"    Feature {idx}: peso = {coefs[idx]:.6f}")

    # 5. Evaluar en test
    print("\n[5/6] Cargando y evaluando en conjunto de test...")
    test_images, test_labels = load_images_from_folder(TEST_DIR)
    print(f"  Total imágenes de test: {len(test_labels)}")

    test_features = extract_features(feature_extractor, test_images)

    y_pred = log_reg.predict(test_features)
    y_prob = log_reg.predict_proba(test_features)[:, 1]

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    print(f"\n  --- Métricas en Test ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 6. Guardar gráficos y modelo
    print("\n[6/6] Guardando resultados...")
    cm = save_confusion_matrix(test_labels, y_pred,
                               os.path.join(RESULTS_DIR, 'matriz_confusion.png'))
    roc_auc = save_roc_curve(test_labels, y_prob,
                             os.path.join(RESULTS_DIR, 'curva_roc.png'))
    save_metrics_report(cm, accuracy, precision, recall, f1, roc_auc,
                        os.path.join(RESULTS_DIR, 'metricas_reporte.txt'))

    # Guardar modelo completo
    modelo_data = {
        'log_reg': log_reg,
        'feature_extractor': feature_extractor.cpu(),
        'clases': CLASES,
        'img_size': IMG_SIZE,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
    }
    joblib.dump(modelo_data, MODEL_PATH)
    print(f"\n  Modelo guardado en: {MODEL_PATH}")

    # Gráfico de distribución de clases
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    train_counts = [np.sum(train_labels == 0), np.sum(train_labels == 1)]
    test_counts = [np.sum(test_labels == 0), np.sum(test_labels == 1)]

    axes[0].bar(['Negativo', 'Positivo'], train_counts, color=['steelblue', 'salmon'])
    axes[0].set_title('Distribución - Entrenamiento')
    axes[0].set_ylabel('Cantidad de imágenes')
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

    axes[1].bar(['Negativo', 'Positivo'], test_counts, color=['steelblue', 'salmon'])
    axes[1].set_title('Distribución - Prueba')
    axes[1].set_ylabel('Cantidad de imágenes')
    for i, v in enumerate(test_counts):
        axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'distribucion_clases.png'), dpi=150)
    plt.close()

    print("\n" + "="*60)
    print("  ¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)


if __name__ == '__main__':
    main()
