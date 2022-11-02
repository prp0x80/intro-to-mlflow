from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def load_csv(path: Path) -> DataFrame:
    try:
        dataframe = pd.read_csv(path)
        return dataframe
    except Exception as e:
        raise Exception(f"Failed to load csv from {path} - {e}")


def split_data(
    data: DataFrame,
    label_col: str,
    drop_cols: List[str] = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    if drop_cols:
        features = data.drop(columns=drop_cols)

    labels = data[label_col]
    train_X, test_X, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_X, train_y, test_X, test_y


def get_labels_encoder(labels: DataFrame) -> np.ndarray:
    labels_encoder = LabelEncoder()
    labels_encoder.fit(labels.values)
    return labels_encoder


def get_feature_transformers(
    numerical_cols: DataFrame,
    categorical_cols: DataFrame,
    numerical_scaler: Callable,
    categorical_encoder: Callable,
) -> List[Tuple[str, Callable, List[str]]]:
    transformers = [
        ("encoder", categorical_encoder(), categorical_cols),
        ("scaler", numerical_scaler(), numerical_cols),
    ]
    return transformers


def train(
    X: DataFrame,
    y: DataFrame,
    feature_encoder: Callable,
    feature_scaler: Callable,
    sklearn_classifier: Callable,
    category_threshold: int = 8,
    model_params: Dict[str, Any] = None,
) -> Tuple:
    if not is_classifier(sklearn_classifier):
        raise Exception(
            f"{sklearn_classifier.__name__} - is not a classification model"
        )
    if not model_params:
        model_params = {}
    cat_cols = [c for c in X.columns if X[c].nunique() <= category_threshold]
    num_cols = X.select_dtypes(include=["int", "float"]).columns
    feature_transformers = get_feature_transformers(
        num_cols, cat_cols, feature_scaler, feature_encoder
    )
    labels_encoder = get_labels_encoder(y)
    y = labels_encoder.transform(y)
    data_preprocessor = ColumnTransformer(transformers=feature_transformers)
    classifier = sklearn_classifier(**model_params)
    pipeline = Pipeline(
        steps=[
            ("data_preprocessor", data_preprocessor),
            ("classifier", classifier),
        ]
    )
    model = pipeline.fit(X, y)
    return model, labels_encoder, pipeline


def evaluate(
    model: Pipeline,
    labels_encoder: LabelEncoder,
    test_X: pd.DataFrame,
    test_y: pd.DataFrame,
) -> Tuple:
    predictions = model.predict(test_X)
    test_y = labels_encoder.transform(test_y)
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions)
    plot_confusion_matrix(
        model, test_X, test_y, display_labels=labels_encoder.classes_
    )
    return accuracy, precision, recall, f1, cm, plt
