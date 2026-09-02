import numpy as np


# ============================================================
#  REGRESION LOGISTICA IMPLEMENTADA DESDE CERO
#  (sin usar librerias de machine learning como scikit-learn)
# ============================================================

def sigmoid(z):
    """Funcion sigmoide: comprime valores reales a [0, 1]."""
    z = np.asarray(z, dtype=np.float64)
    # Clip para evitar overflow/underflow numerico
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def normalize(X, mean, std):
    """Estandarizacion: (X - media) / desviacion estandar."""
    X = np.asarray(X, dtype=np.float64)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe


def compute_cost(y, p):
    """Costo de entropia cruzada binaria (log-loss)."""
    p = np.clip(p, 1e-15, 1 - 1e-15)  # evitar log(0)
    m = y.shape[0]
    cost = -(1.0 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return cost


class LogisticRegressionManual:
    """
    Clasificador de regresion logistica binaria implementado desde cero
    usando descenso de gradiente por mini-batches.

    El modelo aprende los pesos w y el sesgo b que minimizan la
    entropia cruzada binaria. Antes de entrenar, las features se
    estandarizan (media 0, desviacion 1) para mejorar la convergencia;
    los valores de media y desviacion se guardan y se reutilizan
    en la prediccion.
    """

    def __init__(self, lr=0.1, epochs=50, batch_size=64, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.mean_ = None
        self.std_ = None
        self.cost_history_ = []

    def fit(self, X, y):
        """Entrena el modelo con descenso de gradiente por mini-batches."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        # Estandarizar features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X_norm = normalize(X, self.mean_, self.std_)

        # Inicializar pesos y sesgo
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, size=n_features)
        self.bias_ = 0.0

        # Descenso de gradiente por mini-batches
        for epoch in range(self.epochs):
            # Barajar datos al inicio de cada epoch
            indices = rng.permutation(n_samples)
            X_shuffled = X_norm[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                m_batch = X_batch.shape[0]

                # Propagacion hacia adelante
                z = X_batch @ self.weights_ + self.bias_
                p = sigmoid(z)

                # Gradientes
                error = p - y_batch
                dw = (1.0 / m_batch) * (X_batch.T @ error)
                db = (1.0 / m_batch) * np.sum(error)

                # Actualizacion de pesos
                self.weights_ -= self.lr * dw
                self.bias_ -= self.lr * db

            # Registrar costo de la epoch completa
            z_all = X_norm @ self.weights_ + self.bias_
            p_all = sigmoid(z_all)
            cost = compute_cost(y, p_all)
            self.cost_history_.append(cost)

        return self

    def decision_function(self, X):
        """Retorna el logit z = w·x + b para las muestras."""
        if self.weights_ is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        X = np.asarray(X, dtype=np.float64)
        X_norm = normalize(X, self.mean_, self.std_)
        z = X_norm @ self.weights_ + self.bias_
        return z

    def predict_proba(self, X):
        """Probabilidad de pertenecer a la clase positiva P(y=1|x)."""
        z = self.decision_function(X)
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Clasifica: 1 si P(y=1) >= threshold, si no 0."""
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


# ============================================================
#  METRICAS DE EVALUACION IMPLEMENTADAS DESDE CERO
# ============================================================

def confusion_matrix(y_true, y_pred, num_classes=2):
    """Matriz de confusion calculada manualmente."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_score(y_true, y_pred):
    """Proporcion de predicciones correctas."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred):
    """De los predichos positivos, cuantos son realmente positivos."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score(y_true, y_pred):
    """De los positivos reales, cuantos se detectaron (sensibilidad)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(y_true, y_pred):
    """Media armonica entre precision y recall."""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def specificity_score(y_true, y_pred):
    """De los negativos reales, cuantos se detectaron correctamente."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tn + fp == 0:
        return 0.0
    return float(tn / (tn + fp))


def roc_curve(y_true, y_prob):
    """
    Calcula FPR (tasa de falsos positivos) y TPR (tasa de verdaderos
    positivos) para distintos umbrales de decision. Devuelve
    (fpr, tpr, umbrales).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Ordenar por probabilidad descendente
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    # Agregar punto (0,0) y (1,1)
    thresholds = np.unique(p_sorted)
    fpr = [0.0]
    tpr = [0.0]

    n_pos = np.sum(y_sorted == 1)
    n_neg = y_sorted.shape[0] - n_pos

    for thr in thresholds:
        pred = (p_sorted >= thr).astype(int)
        tp = np.sum((y_sorted == 1) & (pred == 1))
        fp = np.sum((y_sorted == 0) & (pred == 1))

        tpr_val = tp / n_pos if n_pos > 0 else 0.0
        fpr_val = fp / n_neg if n_neg > 0 else 0.0
        tpr.append(tpr_val)
        fpr.append(fpr_val)

    fpr.append(1.0)
    tpr.append(1.0)

    return np.array(fpr), np.array(tpr), thresholds


def auc(fpr, tpr):
    """Area bajo la curva ROC usando la regla del trapecio."""
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    # Ordenar por fpr
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    # Regla del trapecio (np.trapz fue renombrado a np.trapezoid en NumPy 2.0+)
    trapz_func = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(trapz_func(tpr, fpr))
