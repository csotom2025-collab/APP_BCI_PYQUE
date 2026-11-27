# -*- coding: utf-8 -*-
"""
Ensamble soft voting por canal EEG.

Un nodo de la cascada (color, super-grupo o comando fino) se entrena una vez
por canal usando solo las features de ese canal. Este objeto agrupa esos
modelos y expone la misma interfaz que un ``Pipeline`` de sklearn
(``predict``, ``predict_proba``, ``classes_``, ``named_steps``), de modo que
``HierarchicalBCIPredictor._predict_with_bundle`` lo usa sin cambios.

Canales faltantes
-----------------
Solo votan los canales disponibles:
  * Explícito: ``set_active_channels(["F3", "F4", ...])``.
  * Automático (por defecto): un canal se considera ausente si TODAS sus
    columnas valen exactamente 0 en la entrada (el predictor rellena con 0 las
    columnas que no existen mediante ``reindex`` + ``nan_to_num``).

Este módulo se mantiene aparte (sin dependencias pesadas) para que joblib pueda
deserializar el bundle sin importar el script de entrenamiento.
"""

import numpy as np


class ChannelSoftVotingEnsemble:
    def __init__(self, channel_models, weights=None):
        """
        channel_models: dict canal -> {"pipeline": Pipeline ajustado,
                                       "feature_columns": [columnas del canal]}
        weights: dict opcional canal -> peso (por defecto, todos 1.0).
        """
        if not channel_models:
            raise ValueError("Se requiere al menos un modelo de canal.")
        self.channel_models = {
            str(ch): {
                "pipeline": m["pipeline"],
                "feature_columns": list(m["feature_columns"]),
            }
            for ch, m in channel_models.items()
        }
        self.weights = dict(weights) if weights else None
        self.active_channels = None
        self.last_active_channels_ = None

        # Columnas de entrada: unión ordenada de las de cada canal
        union = []
        seen = set()
        for model in self.channel_models.values():
            for col in model["feature_columns"]:
                if col not in seen:
                    seen.add(col)
                    union.append(col)
        self.feature_columns = union
        position = {col: i for i, col in enumerate(union)}
        self._index = {
            ch: np.array([position[c] for c in m["feature_columns"]], dtype=int)
            for ch, m in self.channel_models.items()
        }

        first = next(iter(self.channel_models.values()))["pipeline"]
        self.classes_ = np.asarray(first.classes_)
        for ch, m in self.channel_models.items():
            if not np.array_equal(np.asarray(m["pipeline"].classes_), self.classes_):
                raise ValueError(
                    f"El canal {ch} tiene clases distintas a las del resto: "
                    f"{m['pipeline'].classes_} vs {self.classes_}."
                )

    # -- interfaz tipo Pipeline ------------------------------------------------
    @property
    def named_steps(self):
        # El predictor consulta named_steps.get("clf") / get("lda_clf").
        return {}

    @property
    def channels(self):
        return list(self.channel_models)

    def set_active_channels(self, channels=None):
        """Fija los canales que votan. ``None`` vuelve a la detección automática."""
        self.active_channels = None if channels is None else [str(c) for c in channels]

    def _resolve_active_channels(self, X):
        if self.active_channels is not None:
            wanted = set(self.active_channels)
            active = [ch for ch in self.channel_models if ch in wanted]
        else:
            active = [
                ch for ch, idx in self._index.items()
                if np.any(np.nan_to_num(X[:, idx]) != 0)
            ]
        if not active:
            raise ValueError(
                "Ninguno de los canales modelados está disponible. "
                f"Canales del modelo: {self.channels}."
            )
        return active

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        active = self._resolve_active_channels(X)
        n_classes = len(self.classes_)
        total = np.zeros((X.shape[0], n_classes), dtype=float)
        weight_sum = 0.0
        for ch in active:
            model = self.channel_models[ch]
            Xc = np.nan_to_num(
                X[:, self._index[ch]], nan=0.0, posinf=0.0, neginf=0.0
            )
            proba = model["pipeline"].predict_proba(Xc)
            full = np.zeros((X.shape[0], n_classes), dtype=float)
            cols = np.searchsorted(
                self.classes_, np.asarray(model["pipeline"].classes_)
            )
            full[:, cols] = proba
            weight = 1.0 if self.weights is None else float(self.weights.get(ch, 1.0))
            total += weight * full
            weight_sum += weight
        self.last_active_channels_ = list(active)
        return total / weight_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def __repr__(self):
        return f"ChannelSoftVotingEnsemble(canales={self.channels})"
