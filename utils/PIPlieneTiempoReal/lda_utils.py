# -*- coding: utf-8 -*-
"""
lda_utils.py — Clases LDA compartidas entre entrenamiento e inferencia.

IMPORTANTE: Este módulo debe ser importado en TODOS los scripts que guarden
o carguen pipelines que contengan SafeLDA, para que joblib/pickle pueda
resolver la clase correctamente al deserializar.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
