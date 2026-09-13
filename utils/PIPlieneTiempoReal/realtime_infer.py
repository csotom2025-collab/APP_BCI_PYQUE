# -*- coding: utf-8 -*-
"""
Inferencia en tiempo real / offline sobre una grabacion nueva.

Uso por linea de comandos:
    python realtime_infer.py ruta/a/la_grabacion.csv

Uso programatico (ej. desde tu app en tiempo real, ya con la señal en memoria):
    from realtime_infer import BCIPredictor
    predictor = BCIPredictor()
    comando, proba = predictor.predict_from_signals(signals)   # signals: (n_channels, n_samples)
    comando, proba = predictor.predict_from_csv("trial_nuevo.csv")
"""

import sys
import joblib
import numpy as np
import pandas as pd

import config
from eeg_features import EEGFeatureExtractor, apply_baseline_correction, log_transform_power_columns


class BCIPredictor:
    def __init__(self, model_path=None):
        model_path = model_path or config.BEST_MODEL_PATH
        self.bundle = joblib.load(model_path)

        self.pipeline = self.bundle["pipeline"]
        self.label_encoder = self.bundle["label_encoder"]
        self.feature_columns = self.bundle["feature_columns"]
        self.channel_names = self.bundle["channel_names"]
        self.fs = self.bundle["fs"]
        self.use_p300_window_only = self.bundle.get("use_p300_window_only", False)
        self.p300_window_s = self.bundle.get("p300_window_s", (0.5, 1.2))
        self.apply_baseline = self.bundle.get("apply_baseline_correction", True)
        self.baseline_window_s = self.bundle.get("baseline_window_s", (0.0, 0.5))

        self.extractor = EEGFeatureExtractor(fs=self.fs)

        print(f"Modelo cargado: feature_set='{self.bundle['feature_set_name']}' "
              f"| clasificador='{self.bundle['classifier_name']}' "
              f"| n_clases={len(self.label_encoder.classes_)}")

    # -----------------------------------------------------------------
    def _crop_p300(self, signals):
        start = int(self.p300_window_s[0] * self.fs)
        end = int(self.p300_window_s[1] * self.fs)
        end = min(end, signals.shape[1])
        if start >= end:
            return signals
        return signals[:, start:end]

    def _signals_to_feature_vector(self, signals):
        """signals: (n_channels, n_samples) -> vector alineado con feature_columns del modelo."""
        # Misma correccion de linea base usada en entrenamiento (pre-estimulo)
        if self.apply_baseline:
            signals = apply_baseline_correction(signals, self.fs, self.baseline_window_s)

        if self.use_p300_window_only:
            signals = self._crop_p300(signals)

        feat_df = self.extractor.extract_features(
            signals,
            channel_names=self.channel_names,
            available_channel_names=self.channel_names,
            window_size=signals.shape[1],
            overlap=0.0,
        )
        feat_row = feat_df.mean(axis=0, numeric_only=True).to_frame().T

        # Mismo log-transform de columnas de potencia (_Abs/_energy) usado en
        # entrenamiento, si el bundle indica que se aplico.
        if self.bundle.get("log_transform_power", False):
            feat_row = log_transform_power_columns(feat_row, feat_row.columns)

        # Alinear exactamente con las columnas usadas en entrenamiento
        x = feat_row.reindex(columns=self.feature_columns).iloc[0].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.reshape(1, -1)

    # -----------------------------------------------------------------
    def predict_from_signals(self, signals):
        """
        signals: array (n_channels, n_samples), mismo orden que config.CHANNEL_NAMES.
        Retorna: (comando_predicho, dict {comando: probabilidad})
        """
        X = self._signals_to_feature_vector(signals)
        pred_idx = self.pipeline.predict(X)[0]
        comando = self.label_encoder.inverse_transform([pred_idx])[0]

        proba_dict = {}
        if hasattr(self.pipeline, "predict_proba"):
            probas = self.pipeline.predict_proba(X)[0]
            clases = self.label_encoder.inverse_transform(np.arange(len(probas)))
            proba_dict = dict(sorted(zip(clases, probas), key=lambda kv: -kv[1]))

        return comando, proba_dict

    def predict_from_csv(self, path):
        """Carga un CSV crudo de EEG (columnas = canales) y predice el comando."""
        df = pd.read_csv(path)
        cols_presentes = [c for c in self.channel_names if c in df.columns]
        if len(cols_presentes) == len(self.channel_names):
            signals = df[self.channel_names].to_numpy().T
        elif df.shape[1] >= len(self.channel_names):
            signals = df.iloc[:, :len(self.channel_names)].to_numpy().T
        else:
            raise ValueError(f"El CSV {path} no tiene las columnas de canal esperadas.")
        return self.predict_from_signals(signals)


def main():
    if len(sys.argv) < 2:
        print("Uso: python realtime_infer.py ruta/a/grabacion.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    predictor = BCIPredictor()
    comando, probas = predictor.predict_from_csv(csv_path)

    print(f"\n>>> Comando predicho: {comando}")
    print("\nTop 5 probabilidades:")
    for cmd, p in list(probas.items())[:5]:
        print(f"  {cmd}: {p:.3f}")


if __name__ == "__main__":
    main()
