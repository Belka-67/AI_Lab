"""Данные, обучение и объяснения лаборатории. Не зависит от Streamlit."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CLASSES = ["Яблоко", "Груша"]
FEATURES = ["Размер", "Округлость"]
MODELS = ["Ближайшие соседи", "Дерево решений", "Логистическая регрессия"]
TEST_SIZE = 120


def generate_data(count, apple_percent, spread_percent, measurement_percent,
                  mistakes, outliers, seed):
    """Раздельные случайные потоки позволяют менять ровно один фактор опыта."""
    class_rng, shape_rng, outlier_rng, measurement_rng, label_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    apple_count = round(count * apple_percent / 100)
    real = np.array(["Яблоко"] * apple_count + ["Груша"] * (count - apple_count))
    class_rng.shuffle(real)
    spread = 0.4 + 1.4 * spread_percent / 100
    centers = {"Яблоко": (5.7, 7.7), "Груша": (7.0, 4.7)}
    values = np.array([shape_rng.normal(centers[c], spread) for c in real])
    unusual = np.zeros(count, dtype=bool)
    if (n := round(count * outliers / 100)):
        indices = outlier_rng.choice(count, size=n, replace=False)
        unusual[indices] = True
        other = np.where(real[indices] == "Яблоко", "Груша", "Яблоко")
        values[indices] = [outlier_rng.normal(centers[c], 0.35) for c in other]
    values += measurement_rng.normal(0, 1.5 * measurement_percent / 100,
                                     size=values.shape)
    values = np.round(np.clip(values, 0, 10), 1)
    labels = real.copy()
    if (n := round(count * mistakes / 100)):
        indices = label_rng.choice(count, size=n, replace=False)
        labels[indices] = np.where(labels[indices] == "Яблоко", "Груша", "Яблоко")
    return pd.DataFrame({"Размер": values[:, 0], "Округлость": values[:, 1],
                         "Метка": labels, "Верный класс": real, "Выброс": unusual})


def validate(data):
    if len(data) < 6:
        return "Нужно минимум 6 карточек."
    if data["Метка"].isna().any() or data["Верный класс"].isna().any():
        return "У каждой карточки должен быть указан класс."
    if not set(data["Метка"]).issubset(CLASSES) or not set(data["Верный класс"]).issubset(CLASSES):
        return "Выберите класс «Яблоко» или «Груша»."
    if data["Метка"].nunique() != 2:
        return "Для обучения нужны карточки обоих классов."
    for col in FEATURES:
        numeric = pd.to_numeric(data[col], errors="coerce")
        if numeric.isna().any() or (~numeric.between(0, 10)).any():
            return f"В столбце «{col}» нужны числа от 0 до 10."
    return None


def new_model(name, count, k=3, depth=5):
    if name == MODELS[0]:
        available_odd = count if count % 2 else count - 1
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=min(k, available_odd)))
    if name == MODELS[1]:
        return DecisionTreeClassifier(max_depth=depth, random_state=42)
    if name == MODELS[2]:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    raise ValueError(f"Неизвестная модель: {name}")


def class_scores(true, predicted):
    """Строка матрицы — настоящий класс, столбец — прогноз."""
    matrix = confusion_matrix(true, predicted, labels=CLASSES)
    return {"matrix": matrix,
            "by_class": {cls: (int(matrix[i, i]), int(matrix[i].sum()))
                         for i, cls in enumerate(CLASSES)},
            "correct": int(np.trace(matrix)), "total": int(matrix.sum())}


def train(data, test, k=3, depth=5):
    """Знакомые карточки сверяются с учебными метками; новые — с истиной."""
    x, y = data[FEATURES], data["Метка"]
    x_test, y_test = test[FEATURES], test["Верный класс"]
    results = {}
    for name in MODELS:
        model = new_model(name, len(data), k, depth)
        model.fit(x, y)
        seen = model.predict(x)
        unseen = model.predict(x_test)
        score = class_scores(y_test, unseen)
        results[name] = {
            "model": model, "train_correct": int((seen == y.to_numpy()).sum()),
            "test_correct": score["correct"], "train_predictions": seen,
            "test_predictions": unseen, "matrix": score["matrix"],
            "by_class": score["by_class"],
        }
    return results


def data_fingerprint(data):
    return hashlib.sha256(pd.util.hash_pandas_object(data.reset_index(drop=True),
                                                      index=False).values.tobytes()).hexdigest()


def neighbor_details(model, data, point):
    scaler, knn = model.steps[0][1], model.steps[1][1]
    distances, indices = knn.kneighbors(scaler.transform(point[FEATURES]))
    rows = data.iloc[indices[0]].copy()
    rows.insert(0, "Карточка №", indices[0] + 1)
    rows["Расстояние"] = np.round(distances[0], 2)
    return rows, indices[0]


def tree_steps(model, point):
    values = point[FEATURES].iloc[0]
    path = model.decision_path(point[FEATURES]).indices
    lines = []
    for node in path:
        feature = model.tree_.feature[node]
        if feature < 0:
            continue
        name = FEATURES[feature]
        threshold = model.tree_.threshold[node]
        answer = "да" if values[name] <= threshold else "нет"
        lines.append(f"{name} ≤ {threshold:.1f}? {answer} (у объекта {values[name]:.1f})")
    return lines


def logistic_boundary(model):
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    weights = logistic.coef_[0] / scaler.scale_
    offset = float(logistic.intercept_[0] - (weights * scaler.mean_).sum())
    return weights, offset, logistic.classes_


def logistic_explanation(model, point):
    weights, offset, classes = logistic_boundary(model)
    value = float(offset + np.dot(weights, point[FEATURES].iloc[0]))
    side = classes[1] if value >= 0 else classes[0]
    return f"Точка находится по сторону класса «{side}» от прямой границы."
