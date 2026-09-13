"""================================================================================
    SCRIPT: occupancy_classification.py
    PROPOSITO: Clasificacion binaria de ocupacion de una sala a partir de
               datos de sensores ambientales (Temperatura, Humedad, Luz, CO2,
               HumidityRatio) usando scikit-learn y pandas.

    PIPELINE:
        1) Carga y concatenacion de todos los archivos CSV dentro de OccupancyData.
        2) Limpieza: eliminar columna de fecha/hora y columna de indice sin nombre.
        3) Deteccion automatica de la columna objetivo binaria.
        4) Pipeline unificado con StandardScaler.
        5) 4 modelos: SVC, GaussianNB, GradientBoostingClassifier,
           KNeighborsClassifier.
        6) Optimizacion de hiperparametros con GridSearchCV (KFold 5, shuffle).
        7) Validacion cruzada KFold con n_splits=5 y shuffle=True.
        8) Metricas: F1-Score macro y tiempo computacional de cada modelo.
        9) Tabla final ordenada por F1-Score y tiempo.
================================================================================
"""

import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score

# Direccion de la carpeta con los datos
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OccupancyData")


# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------------------------
def cargar_datos(carpeta=DATA_DIR):
    """
    Lee TODOS los archivos .csv/.txt dentro de 'carpeta' y los concatena
    en un solo DataFrame. Tambien elimina la columna de fecha/hora y la
    columna de indice sin nombre si existen.
    """
    # Buscar archivos CSV/TXT dentro de la carpeta de datos
    archivos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".csv") or f.endswith(".txt")
    ]

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .csv/.txt en la carpeta '{carpeta}'."
        )

    print("=" * 80)
    print("1) CARGA DE DATOS")
    print("=" * 80)
    print(f"Archivos encontrados en '{carpeta}':")
    for a in archivos:
        print(f"   -> {os.path.basename(a)}")

    # Leer cada archivo y concatenarlos en un solo DataFrame
    frames = []
    for a in archivos:
        # intentar leer como CSV; si falla lo leemos como texto plano
        try:
            df_parcial = pd.read_csv(a)
        except Exception:
            df_parcial = pd.read_csv(a, sep=r"\s+")
        frames.append(df_parcial)

    df = pd.concat(frames, ignore_index=True)
    print(f"Dimensiones concatenadas: {df.shape[0]} filas x {df.shape[1]} columnas")

    # --- LIMPIEZA DE COLUMNAS -----------------------------------------------
    # 1) Eliminar columna de indice sin nombre (p.ej. la primera columna 'Unnamed: 0')
    for col in df.columns:
        if str(col).startswith("Unnamed"):
            print(f"Columna de indice sin nombre eliminada: '{col}'")
            df.drop(columns=[col], inplace=True)

    # 2) Eliminar columnas de fecha/hora (date, timestamp, datetime, time, etc.)
    cols_fecha = [c for c in df.columns if str(c).lower() in {
        "date", "datetime", "timestamp", "time", "fecha", "fecha/hora"
    }]
    if cols_fecha:
        print(f"Columna(s) de fecha/hora eliminada(s): {cols_fecha}")
        df.drop(columns=cols_fecha, inplace=True)

    print(f"DataFrame final: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


# ------------------------------------------------------------------------------
# 2. DETECCION DE LA COLUMNA OBJETIVO
# ------------------------------------------------------------------------------
def detectar_objetivo(df):
    """
    La columna objetivo se llama 'Occupancy' en este dataset.
    Si no existe, buscamos automaticamente la columna binaria (0/1) final.
    """
    if "Occupancy" in df.columns:
        objetivo = "Occupancy"
    else:
        # Detectar columnas binarias; preferir la que este mas a la derecha
        objetivo = None
        for col in df.columns[::-1]:
            if df[col].dropna().isin([0, 1]).all():
                objetivo = col
                break
        if objetivo is None:
            raise ValueError("No se pudo detectar una columna objetivo binaria.")

    print("=" * 80)
    print("2) DETECCION DE LA COLUMNA OBJETIVO")
    print("=" * 80)
    print(f"Columna objetivo detectada: '{objetivo}'")
    print("Distribucion de clases:")
    print(df[objetivo].value_counts().to_string())
    return objetivo


# ------------------------------------------------------------------------------
# 3. DEFINICION DE LOS MODELOS Y SUS CUADRÍCULAS DE HIPERPARAMETROS
# ------------------------------------------------------------------------------
def definir_modelos_y_grids():
    """
    Devuelve un diccionario: nombre del modelo -> (estimador, grid de hiperm.)
    Solo se eligen 2-3 hiperparametros clave por modelo para no alargar
    demasiado la busqueda.
    """
    modelos = {
        # --- Support Vector Classifier (kernel RBF) ---
        "SVC": (
            SVC(random_state=42),
            {
                "clf__C": [0.1, 1, 10],            # penalizacion del error
                "clf__gamma": [0.01, 0.1, 1],      # radio de influencia del kernel
                "clf__kernel": ["rbf"],            # kernel por defecto
            },
        ),
        # --- Naive Bayes Gaussiano (sin hiperparametros relevantes) ---
        "GaussianNB": (
            GaussianNB(),
            {
                "clf__var_smoothing": [          # suavizado de varianza numerico
                    1e-9, 1e-8, 1e-7
                ],
            },
        ),
        # --- Gradient Boosting Classifier ---
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [50, 100],   # numero de arboles
                "clf__max_depth": [2, 4],         # profundidad de cada arbol
                "clf__learning_rate": [0.05, 0.1],# tasa de aprendizaje
            },
        ),
        # --- K-Nearest Neighbors ---
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7],            # numero de vecinos
                "clf__weights": ["uniform", "distance"],  # ponderacion
            },
        ),
    }
    return modelos


# ------------------------------------------------------------------------------
# 4. PIPELINE UNIFICADO + OPTIMIZACION CON GridSearchCV
# ------------------------------------------------------------------------------
def construir_pipeline(clasificador):
    """
    Pipeline unificado: estandariza caracteristicas (StandardScaler) y luego
    aplica el clasificador indicado.
    """
    return Pipeline([
        ("scaler", StandardScaler()),   # estandarizacion de caracteristicas
        ("clf", clasificador),
    ])


def optimizar_y_evaluar(X, y, nombre, clasificador, param_grid, n_splits=5):
    """
    - Valida con KFold (n_splits, shuffle=True).
    - Realiza GridSearchCV sobre la cuadricula de hiperparametros.
    - Devuelve: mejor F1-macro de validacion cruzada y tiempo en segundos.
    """
    print("=" * 80)
    print(f"3) OPTIMIZACION Y EVALUACION -> {nombre}")
    print("=" * 80)

    # KFold estratificado internamente por GridSearchCV con shuffle=True
    cv_interno = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipeline con estandarizacion + clasificador
    pipeline = construir_pipeline(clasificador)

    # Metricas objetivo: F1-Score macro
    scorer = make_scorer(f1_score, average="macro")

    # GridSearchCV: realiza la busqueda de hiperparametros de forma exhaustiva
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv_interno,          # validacion cruzada interna
        n_jobs=-1,              # usar todos los nucleos disponibles
        verbose=1,
    )

    # Medir el tiempo EXACTO de la busqueda + evaluacion
    t_inicio = time.perf_counter()
    grid.fit(X, y)
    t_fin = time.perf_counter()
    tiempo_seg = round(t_fin - t_inicio, 4)

    # Mejor F1-Score macro obtenido en la validacion cruzada interna
    mejor_f1 = round(grid.best_score_, 4)

    print(f"   Mejores hiperparametros: {grid.best_params_}")
    print(f"   Mejor F1-Score (macro)  : {mejor_f1}")
    print(f"   Tiempo computacional    : {tiempo_seg} segundos")

    return mejor_f1, tiempo_seg, grid.best_params_


# ------------------------------------------------------------------------------
# 5. FUNCION PRINCIPAL
# ------------------------------------------------------------------------------
def main():
    # --- Carga y preparacion de datos ---
    df = cargar_datos()
    objetivo = detectar_objetivo(df)

    # Separar caracteristicas (X) y objetivo (y)
    # Solo se usan columnas numericas como caracteristicas
    cols_features = [c for c in df.columns if c != objetivo]
    X = df[cols_features].select_dtypes(include=[np.number])
    y = df[objetivo].astype(int)

    print("=" * 80)
    print("CARACTERISTICAS (features):")
    print(list(X.columns))
    print("=" * 80)

    # --- Definir modelos ---
    modelos = definir_modelos_y_grids()

    # --- Optimizar y evaluar cada modelo ---
    resultados = []
    for nombre, (clf, grid) in modelos.items():
        f1, tiempo, params = optimizar_y_evaluar(X, y, nombre, clf, grid)
        resultados.append({
            "Modelo": nombre,
            "Mejor F1-Score": f1,
            "Tiempo Computacional (segundos)": tiempo,
        })

    # --- Tabla final ordenada por F1-Score descendente (y mejor tiempo) ---
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(
        by=["Mejor F1-Score", "Tiempo Computacional (segundos)"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print("=" * 80)
    print("RESULTADOS FINALES (ordenados por F1-Score macro descendente)")
    print("=" * 80)
    print(df_resultados.to_string(index=False))

    # Indicar el mejor modelo en general
    mejor = df_resultados.iloc[0]
    print("=" * 80)
    print(f"MEJOR MODELO: {mejor['Modelo']} "
          f"con F1-Score = {mejor['Mejor F1-Score']} "
          f"en {mejor['Tiempo Computacional (segundos)']} segundos")
    print("=" * 80)


# ------------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()