"""================================================================================
    APP STREAMLIT: Comparativa de Modelos de Clasificación
    ------------------------------------------------------
    Aplicación web interactiva que replica el experimento del script
    occupancy_classification.py:

      - Carga y concatenación de los CSV dentro de OccupancyData.
      - Pipeline con StandardScaler.
      - Optimización con GridSearchCV (KFold 5, shuffle=True) para 4 modelos:
        SVC, GaussianNB, GradientBoostingClassifier y KNeighborsClassifier.
      - Métricas: F1-Score macro y tiempo computacional por modelo.

    CÓMO EJECUTAR LA APP:
        streamlit run app_occupancy.py
================================================================================
"""

import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score

# Dirección de la carpeta con los datos
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OccupancyData")


# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------------------------
def cargar_datos(carpeta=DATA_DIR):
    """
    Lee TODOS los archivos .csv/.txt dentro de 'carpeta' y los concatena
    en un solo DataFrame. Elimina la columna de fecha/hora y la columna de
    índice sin nombre si existen.
    """
    archivos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".csv") or f.endswith(".txt")
    ]
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .csv/.txt en la carpeta '{carpeta}'."
        )

    frames = []
    for a in archivos:
        try:
            df_parcial = pd.read_csv(a)
        except Exception:
            df_parcial = pd.read_csv(a, sep=r"\s+")
        frames.append(df_parcial)

    df = pd.concat(frames, ignore_index=True)

    # Eliminar columna de índice sin nombre (p.ej. 'Unnamed: 0')
    for col in df.columns:
        if str(col).startswith("Unnamed"):
            df.drop(columns=[col], inplace=True)

    # Eliminar columnas de fecha/hora
    cols_fecha = [c for c in df.columns if str(c).lower() in {
        "date", "datetime", "timestamp", "time", "fecha", "fecha/hora"
    }]
    if cols_fecha:
        df.drop(columns=cols_fecha, inplace=True)

    return df


# ------------------------------------------------------------------------------
# 2. DETECCIÓN DE LA COLUMNA OBJETIVO
# ------------------------------------------------------------------------------
def detectar_objetivo(df):
    """
    La columna objetivo se llama 'Occupancy'. Si no existe, se busca de forma
    automática la última columna binaria (0/1).
    """
    if "Occupancy" in df.columns:
        return "Occupancy"

    for col in df.columns[::-1]:
        if df[col].dropna().isin([0, 1]).all():
            return col
    raise ValueError("No se pudo detectar una columna objetivo binaria.")


# ------------------------------------------------------------------------------
# 3. MODELOS Y CUADRÍCULAS DE HIPERPARÁMETROS
# ------------------------------------------------------------------------------
def definir_modelos_y_grids():
    """
    Devuelve un diccionario: nombre del modelo -> (estimador, grid de hiperm.)
    Solo 2-3 hiperparámetros clave por modelo para no alargar la búsqueda.
    """
    return {
        # --- Support Vector Classifier (kernel RBF) ---
        "SVC": (
            SVC(random_state=42),
            {
                "clf__C": [0.1, 1, 10],            # penalización del error
                "clf__gamma": [0.01, 0.1, 1],      # radio de influencia del kernel
                "clf__kernel": ["rbf"],
            },
        ),
        # --- Naive Bayes Gaussiano ---
        "GaussianNB": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        # --- Gradient Boosting ---
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [2, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        # --- K-Nearest Neighbors ---
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7],
                "clf__weights": ["uniform", "distance"],
            },
        ),
    }


# ------------------------------------------------------------------------------
# 4. PIPELINE UNIFICADO + OPTIMIZACIÓN (GridSearchCV) Y EVALUACIÓN
# ------------------------------------------------------------------------------
def construir_pipeline(clasificador):
    """Pipeline unificado: StandardScaler + clasificador."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clasificador),
    ])


def optimizar_y_evaluar(X, y, nombre, clasificador, param_grid, n_splits=5):
    """
    - Validación cruzada KFold (n_splits, shuffle=True).
    - GridSearchCV buscando el mejor F1-Score macro.
    - Devuelve: (nombre, mejor_f1, tiempo_seg).
    """
    cv_interno = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipeline = construir_pipeline(clasificador)
    scorer = make_scorer(f1_score, average="macro")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv_interno,
        n_jobs=-1,
        verbose=0,  # desactivado para no saturar la consola de Streamlit
    )

    # Medir el tiempo exacto de la búsqueda + entrenamiento
    t_inicio = time.perf_counter()
    grid.fit(X, y)
    t_fin = time.perf_counter()
    tiempo_seg = round(t_fin - t_inicio, 4)

    mejor_f1 = round(grid.best_score_, 4)
    return nombre, mejor_f1, tiempo_seg, grid.best_params_


# ------------------------------------------------------------------------------
# 5. FUNCIÓN DE ENTRENAMIENTO COMPLETO (NO SE EJECUTA AL ABRIR LA APP)
# ------------------------------------------------------------------------------
def ejecutar_entrenamiento():
    """
    Orquesta todo el experimento:
     1. Carga + limpieza de datos.
     2. Detección del objetivo.
     3. Optimización con GridSearchCV para cada modelo.
     4. Tabla final ordenada por F1-Score y tiempo.

    Devuelve: (df_resultados, mejor_fila, parametros_por_modelo)
    """
    df = cargar_datos()
    objetivo = detectar_objetivo(df)

    cols_features = [c for c in df.columns if c != objetivo]
    X = df[cols_features].select_dtypes(include=[np.number])
    y = df[objetivo].astype(int)

    modelos = definir_modelos_y_grids()

    resultados = []
    parametros = {}
    for nombre, (clf, grid) in modelos.items():
        _, f1, tiempo, mejor_params = optimizar_y_evaluar(X, y, nombre, clf, grid)
        resultados.append({
            "Modelo": nombre,
            "Mejor F1-Score": f1,
            "Tiempo Computacional (segundos)": tiempo,
        })
        parametros[nombre] = mejor_params

    # Tabla final ordenada por F1-Score descendente (y menor tiempo)
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(
        by=["Mejor F1-Score", "Tiempo Computacional (segundos)"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df_resultados, df_resultados.iloc[0], parametros


# ------------------------------------------------------------------------------
# 6. INTERFAZ STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Comparativa de Modelos de Clasificación",
    page_icon="📊",
    layout="wide",
)

# Título principal
st.title("📊 Comparativa de Modelos de Clasificación - Sensores Ambientales")

# Breve descripción del experimento (cols para aprovechar el ancho)
st.markdown(
    """
    **Experimento:** Clasificación binaria de la ocupación de una sala usando datos de
    sensores ambientales (`Temperature`, `Humidity`, `Light`, `CO2`, `HumidityRatio`). \n
    Se entrenan y comparan **4 clasificadores** — `SVC`, `GaussianNB`,
    `GradientBoostingClassifier` y `KNeighborsClassifier` — bajo un pipeline unificado con
    `StandardScaler`. Los hiperparámetros de cada modelo se optimizan mediante
    `GridSearchCV` con validación cruzada `KFold(5, shuffle=True)` y la métrica objetivo es
    el **F1-Score macro**.
    """
)

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Carpeta de datos: `OccupancyData`")
    st.markdown("4 modelos · GridSearchCV · KFold(5) · F1-macro")
    # Botón que dispara el entrenamiento bajo demanda
    ejecutar = st.button("🚀 Ejecutar Entrenamiento y Validación")

# Solo se entrena si el usuario presiona el botón
if ejecutar:
    with st.spinner("⏳ Optimizando los 4 modelos con GridSearchCV... Esto puede tardar unos segundos."):
        df_resultados, mejor_fila, parametros = ejecutar_entrenamiento()

    st.success("✅ Entrenamiento y validación finalizados")

    # --- Tabla de resultados con st.dataframe ---
    st.subheader("Resultados de la comparativa")
    st.dataframe(df_resultados, use_container_width=True)

    # --- Mejores hiperparámetros por modelo (información extra) ---
    with st.expander("Ver mejores hiperparámetros por modelo"):
        for nombre, params in parametros.items():
            st.markdown(f"**{nombre}**: `{params}`")

    # --- Destacar el mejor modelo ---
    st.markdown(
        f"🏆 **Mejor modelo:** `{mejor_fila['Modelo']}` con "
        f"**F1-Score = {mejor_fila['Mejor F1-Score']}** en "
        f"**{mejor_fila['Tiempo Computacional (segundos)']} segundos**."
    )

    # --- Gráfico de barras comparando F1-Score de los 4 modelos ---
    st.subheader("Comparativa visual del F1-Score (macro)")
    chart_data = df_resultados.set_index("Modelo")[["Mejor F1-Score"]]
    st.bar_chart(chart_data, color="#ff7f0e")
else:
    st.info("👈 Presiona el botón **'Ejecutar Entrenamiento y Validación'** en la barra lateral para comenzar.")