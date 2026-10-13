# -*- coding: utf-8 -*-
"""
Inferencia en tiempo real con clasificación jerárquica:
    1) color predice Green/Blue
    2) super-clase del color predice Letters/Numbers/Controls
    3) el modelo fino del color y grupo predice el comando final.

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

import models.newTraining.config as config
from models.newTraining.eeg_features import (
    EEGFeatureExtractor,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_baseline_correction,
    apply_zscore_normalization,
    log_transform_power_columns
)
from models.newTraining.p300_segmentation import extract_with_flash_segmentation
from models.newTraining.lda_utils import SafeLDA  # noqa: F401 — necesario para que joblib deserialice SafeLDA correctamente


class HierarchicalBCIPredictor:
    def __init__(self, usuario=None, model_path=None, use_flash_segmentation=False):
        """
        usuario: nombre del usuario cuyo modelo monousuario se debe cargar
        model_path: ruta explicita al .joblib
        use_flash_segmentation: si True, segmenta en flashes P300. Si False (default),
                                procesa la señal completa con las mismas ventanas y filtros
                                que data_loader.py.
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

        self.channel_names = self.bundle.get("channel_names", config.CHANNEL_NAMES)
        self.fs = self.bundle.get("fs", config.FS)
        self.use_p300_window_only = self.bundle.get("use_p300_window_only", getattr(config, "USE_P300_WINDOW_ONLY", False))
        self.p300_window_s = self.bundle.get("p300_window_s", getattr(config, "P300_WINDOW_S", (0.5, 1.2)))
        self.apply_baseline = self.bundle.get("apply_baseline_correction", getattr(config, "APPLY_BASELINE_CORRECTION", True))
        self.baseline_window_s = self.bundle.get("baseline_window_s", getattr(config, "BASELINE_WINDOW_S", (0.0, 0.5)))
        self.log_transform_power = self.bundle.get("log_transform_power", True)

        # Filtros y normalización idénticos a data_loader.py
        self.apply_notch = getattr(config, "APPLY_NOTCH_FILTER", True)
        self.notch_freq = getattr(config, "NOTCH_FREQ", 60.0)
        self.apply_bandpass = getattr(config, "APPLY_BANDPASS_FILTER", True)
        self.lowcut = getattr(config, "BANDPASS_LOWCUT", 0.5)
        self.highcut = getattr(config, "BANDPASS_HIGHCUT", 40.0)
        self.apply_zscore = getattr(config, "APPLY_ZSCORE_NORMALIZATION", True)
        self.max_peak_threshold = getattr(config, "MAX_PEAK_THRESHOLD", 150.0)

        # Configuración de ventaneo idéntica a data_loader.py
        self.use_win_size = getattr(config, "USE_WIN_SIZE", True)
        self.window_size = getattr(config, "WINDOW_SIZE", 192)
        self.window_overlap = getattr(config, "WINDOW_OVERLAP", 0.89)

        self.last_artifact_info = {}

        if "color" not in self.bundle or "por_color" not in self.bundle:
            raise ValueError(
                "El bundle no contiene los modelos jerárquicos separados por color. "
                "Vuelve a entrenarlo desde la interfaz de entrenamiento."
            )
        self.color_bundle = self.bundle["color"]
        self.color_hierarchies = self.bundle["por_color"]
        self.color_feature_cols = list(self.color_bundle["feature_columns"])
        self.super_feature_cols = {
            color: list(hierarchy["super_clase"]["feature_columns"])
            for color, hierarchy in self.color_hierarchies.items()
        }
        self.group_feature_cols = {
            color: {
                group: list(bundle["feature_columns"])
                for group, bundle in hierarchy["por_grupo"].items()
            }
            for color, hierarchy in self.color_hierarchies.items()
        }

        # La extracción de features se hace una sola vez para cubrir todos los niveles.
        all_cols = list(self.color_feature_cols)
        for color, cols in self.super_feature_cols.items():
            for col in cols:
                if col not in all_cols:
                    all_cols.append(col)
            for group_cols in self.group_feature_cols[color].values():
                for col in group_cols:
                    if col not in all_cols:
                        all_cols.append(col)
        self.feature_columns = all_cols

        self.extractor = EEGFeatureExtractor(fs=self.fs)

        print(f"Modelo jerarquico monousuario cargado: usuario='{self.usuario}'")
        print(f"  Color: f1_cv={self.color_bundle.get('f1_macro_cv', 0.0):.3f}")
        for color, hierarchy in self.color_hierarchies.items():
            super_bundle = hierarchy["super_clase"]
            print(f"  Super-clase {color}: f1_cv={super_bundle.get('f1_macro_cv', 0.0):.3f}")
            for grupo, bundle in hierarchy["por_grupo"].items():
                if "fixed_label" in bundle:
                    print(f"    {grupo:10s}: salida fija {ascii(bundle['fixed_label'])}")
                    continue
                n_clases = len(bundle["label_encoder"].classes_)
                print(f"    {grupo:10s}: f1_cv={bundle.get('f1_macro_cv', 0.0):.3f} "
                      f"({n_clases} clases)")


    # -----------------------------------------------------------------
    def _crop_p300(self, signals):
        start = int(self.p300_window_s[0] * self.fs)
        end = int(self.p300_window_s[1] * self.fs)
        end = min(end, signals.shape[1])
        if start >= end:
            return signals
        return signals[:, start:end]

    def preprocess_signals(self, signals):
        """
        Aplica exactamente la misma cadena de preprocesamiento que data_loader.py:
        0) Filtro Notch (60 Hz)
        1) Filtro Paso de Banda (0.5 a 40 Hz)
        2) Corrección de línea base (pre-estímulo 0.0 - 0.5s)
        3) Detección de picos (> 150 uV)
        4) Normalización Z-score
        5) Recorte a ventana P300 si use_p300_window_only=True
        """
        signals = np.array(signals, dtype=float, copy=True)

        # 0) Filtro Notch (60 Hz)
        if self.apply_notch:
            signals = apply_notch_filter(signals, self.fs, notch_freq=self.notch_freq)

        # 1) Filtro Paso de Banda (0.5 a 40 Hz)
        if self.apply_bandpass:
            signals = apply_bandpass_filter(signals, self.fs, lowcut=self.lowcut, highcut=self.highcut)

        # 2) Correccion de linea base
        if self.apply_baseline:
            signals = apply_baseline_correction(signals, self.fs, self.baseline_window_s)

        # 3) Deteccion de picos (> 150 uV)
        pico_maximo = float(np.max(np.abs(signals)))
        es_artefacto = pico_maximo > self.max_peak_threshold

        # 4) Normalizacion y Escalamiento (Z-score)
        if self.apply_zscore:
            signals = apply_zscore_normalization(signals)

        # 5) Recorte opcional a la ventana P300
        if self.use_p300_window_only:
            signals = self._crop_p300(signals)

        return signals, es_artefacto, pico_maximo

    def _signals_to_features(self, signals, return_artifact_info=False):
        """
        signals: (n_channels, n_samples) -> DataFrame con 1 fila por ventana (ej. 4 ventanas de 192 muestras).
        Aplica el mismo preprocesamiento y extracción que data_loader.py para que cada
        ventana se pase individualmente al clasificador tal como se entrenó.
        """
        signals, es_artefacto, pico_maximo = self.preprocess_signals(signals)

        if self.use_flash_segmentation:
            # Segmentación de flashes P300 (experimental)
            feat_df = extract_with_flash_segmentation(
                signals, self.feature_columns, self.extractor,
                apply_baseline=False,
                use_p300_window=False,
                p300_window_s=self.p300_window_s,
                log_transform_power=self.log_transform_power,
                fs=self.fs,
                verbose=False
            )
        else:
            n_samples = signals.shape[1]
            if self.use_win_size:
                window_size = min(self.window_size, n_samples)
                overlap = self.window_overlap
            else:
                window_size = n_samples
                overlap = 0.0

            feat_df = self.extractor.extract_features(
                signals,
                channel_names=self.channel_names,
                available_channel_names=self.channel_names,
                window_size=window_size,
                overlap=overlap,
            )

            if self.log_transform_power:
                feat_df = log_transform_power_columns(feat_df, feat_df.columns)

        self.last_artifact_info = {
            "es_artefacto": es_artefacto,
            "pico_maximo_uV": pico_maximo,
        }

        if return_artifact_info:
            return feat_df, es_artefacto, pico_maximo
        return feat_df

    def _signals_to_feature_row(self, signals, return_artifact_info=False):
        """Método de compatibilidad: promedia las ventanas en una sola fila."""
        feat_df, es_art, pico = self._signals_to_features(signals, return_artifact_info=True)
        feat_row = feat_df.mean(axis=0, numeric_only=True).to_frame().T
        if return_artifact_info:
            return feat_row, es_art, pico
        return feat_row

    def _predict_with_bundle(self, bundle, feat_df):
        """
        Pasa todas las ventanas (filas de feat_df) al clasificador.
        Evalúa cada ventana y combina las predicciones mediante soft voting (promedio de probabilidades).
        
        Retorna:
            pred_label: clase ganadora global
            proba_dict: diccionario de probabilidades promedio
            window_details: lista con predicción y probabilidades de cada ventana individual
        """
        if "fixed_label" in bundle:
            pred_label = bundle["fixed_label"]
            proba_dict = {pred_label: 1.0}
            window_details = [
                {
                    "ventana": i + 1,
                    "prediccion": pred_label,
                    "probabilidades": proba_dict,
                }
                for i in range(len(feat_df))
            ]
            return pred_label, proba_dict, window_details

        X = feat_df.reindex(columns=bundle["feature_columns"]).to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pipeline = bundle["pipeline"]
        le = bundle["label_encoder"]
        clases = le.classes_

        window_preds_idx = pipeline.predict(X)
        window_preds = le.inverse_transform(window_preds_idx)

        window_details = []
        if hasattr(pipeline, "predict_proba"):
            all_probas = pipeline.predict_proba(X)  # shape: (n_ventanas, n_clases)
            avg_probas = np.mean(all_probas, axis=0)  # Soft voting entre las 4 ventanas
            pred_idx = np.argmax(avg_probas)
            pred_label = clases[pred_idx]
            proba_dict = dict(sorted(zip(clases, avg_probas), key=lambda kv: -kv[1]))

            for i in range(len(X)):
                w_dict = dict(sorted(zip(clases, all_probas[i]), key=lambda kv: -kv[1]))
                window_details.append({
                    "ventana": i + 1,
                    "prediccion": window_preds[i],
                    "probabilidades": w_dict
                })
        else:
            vals, counts = np.unique(window_preds, return_counts=True)
            pred_label = vals[np.argmax(counts)]
            proba_dict = {pred_label: 1.0}
            for i in range(len(X)):
                window_details.append({
                    "ventana": i + 1,
                    "prediccion": window_preds[i],
                    "probabilidades": {}
                })

        return pred_label, proba_dict, window_details

    # -----------------------------------------------------------------
    def predict_from_signals(self, signals):
        """
        Predice a partir de un arreglo (n_channels, n_samples).
        Extrae y pasa las 4 ventanas directamente al clasificador jerárquico.
        """
        return self.predict_from_repetitions([signals])

    def predict_from_repetitions(self, lista_signals):
        """
        Pasa todas las ventanas de las señales de entrada al clasificador.
        Si hay 1 trial de 2s, extrae 4 ventanas y las clasifica individualmente,
        combinando sus probabilidades con soft voting.
        """
        feat_dfs = []
        artefactos = []
        picos = []
        for s in lista_signals:
            f_df, es_art, pico = self._signals_to_features(s, return_artifact_info=True)
            feat_dfs.append(f_df)
            artefactos.append(es_art)
            picos.append(pico)

        all_feats = pd.concat(feat_dfs, ignore_index=True)

        # Etapa 1: color
        color_feats = all_feats.reindex(columns=self.color_feature_cols)
        color_pred, color_probas, color_ventanas = self._predict_with_bundle(
            self.color_bundle, color_feats
        )

        hierarchy = self.color_hierarchies.get(str(color_pred))
        if hierarchy is None:
            raise ValueError(
                f"El color predicho '{color_pred}' no tiene modelos asociados."
            )
        super_bundle = hierarchy["super_clase"]
        group_bundles = hierarchy["por_grupo"]

        # Etapa 2: el super-grupo se predice solo con los datos de este color.
        super_feats = all_feats.reindex(columns=self.super_feature_cols[color_pred])
        grupo_pred, grupo_probas, grupo_ventanas = self._predict_with_bundle(
            super_bundle, super_feats
        )

        # Etapa 3: comando fino restringido al color y grupo elegidos.
        if grupo_pred not in group_bundles:
            raise ValueError(
                f"El grupo '{grupo_pred}' no tiene un modelo para el color '{color_pred}'."
            )
        group_bundle = group_bundles[grupo_pred]
        group_feats = all_feats.reindex(
            columns=self.group_feature_cols[color_pred][grupo_pred]
        )
        comando_final, comando_probas, comando_ventanas = self._predict_with_bundle(
            group_bundle, group_feats
        )

        detalle = {
            "color_predicho": color_pred,
            "color_probabilidades": color_probas,
            "grupo_predicho": grupo_pred,
            "grupo_probabilidades": grupo_probas,
            "comando_probabilidades": comando_probas,
            "n_repeticiones_usadas": len(lista_signals),
            "n_ventanas": len(all_feats),
            "ventanas": [
                {
                    "ventana": i + 1,
                    "color": color_ventanas[i]["prediccion"],
                    "grupo": grupo_ventanas[i]["prediccion"],
                    "comando": comando_ventanas[i]["prediccion"],
                    "confianza_grupo": float(grupo_ventanas[i]["probabilidades"].get(grupo_ventanas[i]["prediccion"], 1.0)),
                    "confianza_comando": float(comando_ventanas[i]["probabilidades"].get(comando_ventanas[i]["prediccion"], 1.0)),
                }
                for i in range(len(all_feats))
            ],
            "detalle_ventanas_color": color_ventanas,
            "detalle_ventanas_grupo": grupo_ventanas,
            "detalle_ventanas_comando": comando_ventanas,
            "artefactos_detectados": artefactos,
            "picos_maximos_uV": picos,
            "es_artefacto": any(artefactos) if artefactos else False,
            "pico_maximo_uV": max(picos) if picos else 0.0,
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

    print(f"\n>>> Color predicho: {detalle['color_predicho']}")
    print(f">>> Grupo predicho: {detalle['grupo_predicho']}")
    print(f">>> Comando final: {comando}")
    print("\nProbabilidades de color:")
    for color, p in detalle["color_probabilidades"].items():
        print(f"  {color}: {p:.3f}")
    print("\nTop 3 probabilidades de grupo:")
    for g, p in list(detalle["grupo_probabilidades"].items())[:3]:
        print(f"  {g}: {p:.3f}")
    print("\nTop 5 probabilidades de comando (dentro del grupo elegido):")
    for cmd, p in list(detalle["comando_probabilidades"].items())[:5]:
        print(f"  {cmd}: {p:.3f}")
