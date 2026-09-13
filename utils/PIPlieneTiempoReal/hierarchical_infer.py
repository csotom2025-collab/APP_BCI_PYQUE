# -*- coding: utf-8 -*-
"""
Inferencia en tiempo real con la arquitectura jerarquica de 4 redes:
    1) super_clase predice el tipo de comando (Letters/Numbers/Controls)
    2) segun el resultado, se usa el modelo especializado de ese grupo
       para predecir el comando final.

Uso por linea de comandos:
    python hierarchical_infer.py ruta/a/grabacion.csv

Uso programatico:
    from hierarchical_infer import HierarchicalBCIPredictor
    predictor = HierarchicalBCIPredictor()
    comando, detalle = predictor.predict_from_signals(signals)  # signals: (n_channels, n_samples)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

import config
from eeg_features import EEGFeatureExtractor, apply_baseline_correction, log_transform_power_columns
from p300_segmentation import extract_with_flash_segmentation
from lda_utils import SafeLDA  # noqa: F401 — necesario para que joblib deserialice SafeLDA correctamente


class HierarchicalBCIPredictor:
    def __init__(self, usuario=None, model_path=None, use_flash_segmentation=False):
        """
        usuario: nombre del usuario cuyo modelo monousuario se debe cargar
        model_path: ruta explicita al .joblib
        use_flash_segmentation: si True (default), segmenta los 2 segundos en 5 flashes P300,
                                extrae características de cada uno, y promedia antes de
                                clasificar (mejora SNR y accuracy). Si False, usa el
                                archivo completo tal cual.
        """
        if model_path is None:
            if usuario is None:
                raise ValueError(
                    "Debes indicar 'usuario' (modelo monousuario) o 'model_path' explicito."
                )
            model_path = os.path.join(config.MODELS_DIR, usuario, "hierarchical_bundle_optimizado.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontro el modelo en: {model_path}\n"
                f"Verifica que ya corriste 'python hierarchical_train.py' para ese usuario."
            )

        self.bundle = joblib.load(model_path)
        self.usuario = self.bundle.get("usuario", usuario)
        self.use_flash_segmentation = use_flash_segmentation

        self.channel_names = self.bundle["channel_names"]
        self.fs = self.bundle["fs"]
        self.use_p300_window_only = self.bundle.get("use_p300_window_only", False)
        self.p300_window_s = self.bundle.get("p300_window_s", (0.5, 1.2))
        self.apply_baseline = self.bundle.get("apply_baseline_correction", True)
        self.baseline_window_s = self.bundle.get("baseline_window_s", (0.0, 0.5))
        self.log_transform_power = self.bundle.get("log_transform_power", True)

        self.super_bundle = self.bundle["super_clase"]
        self.group_bundles = self.bundle["por_grupo"]  # {"Letters": {...}, "Numbers": {...}, "Controls": {...}}

        # feature_columns: unión de todas las columnas de todos los sub-bundles
        # (no existe en el nivel raíz del bundle — se deriva de los sub-modelos)
        all_cols = list(self.super_bundle["feature_columns"])
        for b in self.group_bundles.values():
            for c in b["feature_columns"]:
                if c not in all_cols:
                    all_cols.append(c)
        self.feature_columns = all_cols

        self.extractor = EEGFeatureExtractor(fs=self.fs)

        print(f"Modelo jerarquico monousuario cargado: usuario='{self.usuario}'")
        print(f"  Super-clase: f1_cv={self.super_bundle.get('f1_macro_cv', 0.0):.3f}")
        for grupo, b in self.group_bundles.items():
            n_clases = len(b["label_encoder"].classes_)
            print(f"  {grupo:10s}: f1_cv={b.get('f1_macro_cv', 0.0):.3f} ({n_clases} clases)")


    # -----------------------------------------------------------------
    def _crop_p300(self, signals):
        start = int(self.p300_window_s[0] * self.fs)
        end = int(self.p300_window_s[1] * self.fs)
        end = min(end, signals.shape[1])
        if start >= end:
            return signals
        return signals[:, start:end]

    def _signals_to_feature_row(self, signals):
        """signals: (n_channels, n_samples) -> DataFrame de 1 fila con TODAS las columnas de features.
        
        Si use_flash_segmentation=True (default): segmenta en 5 flashes P300,
        extrae características de cada uno, promedia (máxima SNR, máxima accuracy).
        Si False: usa la señal completa (backward compatibility).
        """
        if self.apply_baseline:
            signals = apply_baseline_correction(signals, self.fs, self.baseline_window_s)

        if self.use_flash_segmentation:
            # Segmentación de 5 flashes + extracción + promediado automático
            feat_row = extract_with_flash_segmentation(
                signals, self.feature_columns, self.extractor,
                apply_baseline=False,  # ya se hizo arriba
                use_p300_window=self.use_p300_window_only,
                p300_window_s=self.p300_window_s,
                log_transform_power=self.log_transform_power,
                fs=self.fs,
                verbose=False
            )
        else:
            # Método anterior (sin segmentación)
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

            if self.log_transform_power:
                feat_row = log_transform_power_columns(feat_row, feat_row.columns)

        return feat_row

    def _predict_with_bundle(self, bundle, feat_row):
        x = feat_row.reindex(columns=bundle["feature_columns"]).iloc[0].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

        pipeline = bundle["pipeline"]
        le = bundle["label_encoder"]

        pred_idx = pipeline.predict(x)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        proba_dict = {}
        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(x)[0]
            clases = le.inverse_transform(np.arange(len(probas)))
            proba_dict = dict(sorted(zip(clases, probas), key=lambda kv: -kv[1]))

        return pred_label, proba_dict

    # -----------------------------------------------------------------
    def predict_from_signals(self, signals):
        """
        ATAJO para una sola repeticion (signals: array (n_channels, n_samples)).
        Solo da resultados consistentes con el entrenamiento si TODAS las
        etapas fueron entrenadas con n_rep=1. Si alguna etapa (ej. Letters o
        Numbers) fue entrenada promediando varias repeticiones, usa
        `predict_from_repetitions()` en su lugar -- ver esa funcion para el
        porque.
        """
        return self.predict_from_repetitions([signals])

    def predict_from_repetitions(self, lista_signals):
        """
        Version correcta para tiempo real cuando las etapas se entrenaron
        promediando repeticiones (ver STAGE_CONFIG["n_rep"] en
        hierarchical_train.py). En un speller P300 real, nunca se decide con
        un solo flash: se presenta el mismo estimulo candidato varias veces
        seguidas y se promedian las respuestas antes de clasificar (asi se
        entrenaron estos modelos, y asi hay que usarlos para que el
        rendimiento en vivo coincida con el medido en validacion cruzada).

        lista_signals: lista de arrays (n_channels, n_samples), cada uno una
                       repeticion/flash del MISMO estimulo candidato, en
                       cualquier cantidad >= max(n_rep de todas las etapas)
                       -- si tienes menos repeticiones que las que pide una
                       etapa, se usan todas las disponibles (con un aviso).

        Retorna: (comando_final, detalle)
        """
        feat_rows = [self._signals_to_feature_row(s) for s in lista_signals]
        all_feats = pd.concat(feat_rows, ignore_index=True)
        n_disponibles = len(all_feats)

        # Etapa 1: super-clase (usa las primeras n_rep_super repeticiones)
        n_rep_super = self.super_bundle.get("n_rep", 1)
        n_use = min(n_rep_super, n_disponibles)
        if n_use < n_rep_super:
            print(f"[AVISO] super_clase espera {n_rep_super} repeticiones, "
                  f"solo hay {n_disponibles}. Se usan todas las disponibles.")
        feat_super = all_feats.iloc[:n_use].mean(axis=0).to_frame().T
        grupo_pred, grupo_probas = self._predict_with_bundle(self.super_bundle, feat_super)

        # Etapa 2: modelo especializado del grupo predicho
        if grupo_pred not in self.group_bundles:
            raise ValueError(f"Grupo predicho '{grupo_pred}' no tiene modelo especializado asociado.")
        group_bundle = self.group_bundles[grupo_pred]
        n_rep_group = group_bundle.get("n_rep", 1)
        n_use_g = min(n_rep_group, n_disponibles)
        if n_use_g < n_rep_group:
            print(f"[AVISO] el modelo de '{grupo_pred}' espera {n_rep_group} repeticiones, "
                  f"solo hay {n_disponibles}. Se usan todas las disponibles (la accuracy "
                  f"esperada sera menor a la reportada en validacion cruzada).")
        feat_group = all_feats.iloc[:n_use_g].mean(axis=0).to_frame().T
        comando_final, comando_probas = self._predict_with_bundle(group_bundle, feat_group)

        detalle = {
            "grupo_predicho": grupo_pred,
            "grupo_probabilidades": grupo_probas,
            "comando_probabilidades": comando_probas,
            "n_repeticiones_usadas_super": n_use,
            "n_repeticiones_usadas_grupo": n_use_g,
        }
        return comando_final, detalle

    def predict_from_csv_list(self, paths):
        """Version de predict_from_repetitions() que recibe una lista de rutas CSV
        (una por repeticion del mismo estimulo candidato)."""
        signals_list = [self._load_csv_as_signals(p) for p in paths]
        return self.predict_from_repetitions(signals_list)

    def _load_csv_as_signals(self, path):
        df = pd.read_csv(path)
        cols_presentes = [c for c in self.channel_names if c in df.columns]
        if len(cols_presentes) == len(self.channel_names):
            return df[self.channel_names].to_numpy().T
        elif df.shape[1] >= len(self.channel_names):
            return df.iloc[:, :len(self.channel_names)].to_numpy().T
        raise ValueError(f"El CSV {path} no tiene las columnas de canal esperadas.")

    def predict_from_csv(self, path):
        """Atajo de 1 sola repeticion (ver advertencia en predict_from_signals)."""
        signals = self._load_csv_as_signals(path)
        return self.predict_from_signals(signals)


def main():
    if len(sys.argv) < 3:
        print("Uso: python hierarchical_infer.py <usuario> ruta/a/grabacion.csv")
        sys.exit(1)

    usuario = sys.argv[1]
    csv_path = sys.argv[2]
    predictor = HierarchicalBCIPredictor(usuario=usuario)
    comando, detalle = predictor.predict_from_csv(csv_path)

    print(f"\n>>> Grupo predicho: {detalle['grupo_predicho']}")
    print(f">>> Comando final: {comando}")
    print("\nTop 3 probabilidades de grupo:")
    for g, p in list(detalle["grupo_probabilidades"].items())[:3]:
        print(f"  {g}: {p:.3f}")
    print("\nTop 5 probabilidades de comando (dentro del grupo elegido):")
    for cmd, p in list(detalle["comando_probabilidades"].items())[:5]:
        print(f"  {cmd}: {p:.3f}")


if __name__ == "__main__":
    main()
