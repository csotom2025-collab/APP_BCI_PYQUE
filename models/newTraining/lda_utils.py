# -*- coding: utf-8 -*-
"""
lda_utils.py — Clases LDA compartidas entre entrenamiento e inferencia.

IMPORTANTE: Este módulo debe ser importado en TODOS los scripts que guarden
o carguen pipelines que contengan SafeLDA, para que joblib/pickle pueda
resolver la clase correctamente al deserializar.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class ZeroThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier wrapper that applies a zero decision-score threshold."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError("ZeroThresholdClassifier requiere exactamente dos clases.")
        return self

    def decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            scores = np.asarray(self.estimator_.decision_function(X))
            if scores.ndim == 2:
                return scores[:, 1] - scores[:, 0]
            return scores

        if hasattr(self.estimator_, "predict_proba"):
            probabilities = self.estimator_.predict_proba(X)
            eps = np.finfo(float).eps
            return np.log(np.clip(probabilities[:, 1], eps, 1.0)) - np.log(
                np.clip(probabilities[:, 0], eps, 1.0)
            )

        raise AttributeError(
            "El clasificador base debe implementar decision_function o predict_proba."
        )

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)

        scores = self.decision_function(X)
        positive_probability = np.exp(-np.logaddexp(0.0, -scores))
        return np.column_stack((1.0 - positive_probability, positive_probability))


class SafeLDA(LinearDiscriminantAnalysis):
    """
    Wrapper de LDA que ajusta n_components en fit() para respetar el límite real
    min(n_classes - 1, n_features - 1). Necesario en CV donde cada fold puede
    tener menos clases que el dataset completo, o cuando el LDA intermedio
    trabaja sobre un espacio ya reducido con menos dimensiones.
    """
    def fit(self, X, y):
        max_comp = min(len(np.unique(y)) - 1, X.shape[1])
        if max_comp < 1:
            max_comp = 1
        if self.n_components is not None and self.n_components > max_comp:
            self.n_components = max_comp
        return super().fit(X, y)

class ZeroThresholdLDA(LinearDiscriminantAnalysis):
    """Binary LDA classifier whose decision boundary is exactly zero."""

    def predict(self, X):
        if len(self.classes_) != 2:
            raise ValueError("ZeroThresholdLDA requiere exactamente dos clases.")
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]
