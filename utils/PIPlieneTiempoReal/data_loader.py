# -*- coding: utf-8 -*-
"""
Recorre la estructura:
    {BASE_DIR}/{usuario}/{tpComando}/{usuario}_{letra}_{trial}.csv
extrae caracteristicas por trial con EEGFeatureExtractor y arma un
DataFrame final con una fila por trial (label = letra/simbolo/numero).
"""

import os
import glob
import numpy as np
import pandas as pd

import config
from eeg_features import (
    EEGFeatureExtractor, 
    apply_baseline_correction,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_ica_artifact_removal,
    apply_car_rereference,
    apply_zscore_normalization
)


def _parse_filename(fname, usuario):
    """
    Extrae (letra, trial) del nombre de archivo:
    {usuario}_{letra}_{trial}.csv
    La letra puede contener guiones/simbolos unicode (ej. '───', '↩', '⟵'),
    por eso se usa rsplit para separar el trial (ultimo token numerico).
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    prefix = f"{usuario}_"
    if not stem.startswith(prefix):
        return None, None
    resto = stem[len(prefix):]
    if "_" not in resto:
        return None, None
    letra, trial = resto.rsplit("_", 1)
    return letra, trial


def _load_signal_csv(path, channel_names):
    """
    Carga un CSV de una grabacion y devuelve un array (n_channels, n_samples).
    Asume que el CSV tiene columnas con los nombres de canal (CHANNEL_NAMES).
    Si el CSV no tiene encabezado, ajusta este metodo para usar posiciones fijas.
    """
    df = pd.read_csv(path)

    # Caso 1: el CSV trae columnas con los nombres de canal esperados
    cols_presentes = [c for c in channel_names if c in df.columns]
    if len(cols_presentes) == len(channel_names):
        signals = df[channel_names].to_numpy().T  # (n_channels, n_samples)
        return signals

    # Caso 2: no hay encabezado util -> asumir que las primeras N columnas
    # son los canales en el mismo orden que channel_names
    if df.shape[1] >= len(channel_names):
        signals = df.iloc[:, :len(channel_names)].to_numpy().T
        return signals

    raise ValueError(f"No se pudo interpretar el CSV {path}: "
                      f"se esperaban {len(channel_names)} canales.")


def _crop_p300_window(signals, fs):
    """Recorta la señal a la ventana P300 (0.5 - 1.2s) si esta activado en config."""
    start = int(config.P300_WINDOW_S[0] * fs)
    end = int(config.P300_WINDOW_S[1] * fs)
    end = min(end, signals.shape[1])
    if start >= end:
        return signals
    return signals[:, start:end]


def build_dataset(base_dir=None, usuarios=None, tp_comandos=None, verbose=True):
    """
    Recorre todos los usuarios/tpComando/archivos y construye un DataFrame
    con una fila por trial: columnas de caracteristicas + metadatos
    (usuario, tpComando, letra, trial, label).
    """
    base_dir = base_dir or config.BASE_DIR
    usuarios = usuarios or config.USUARIOS
    tp_comandos = tp_comandos or config.TP_COMANDOS

    extractor = EEGFeatureExtractor(fs=config.FS)
    rows = []
    n_ok, n_fail = 0, 0

    for usuario in usuarios:
        for tp in tp_comandos:
            carpeta = os.path.join(base_dir, usuario, tp)
            archivos = sorted(glob.glob(os.path.join(carpeta, f"{usuario}_*.csv")))
            if not archivos and verbose:
                print(f"[AVISO] Sin archivos en: {carpeta}")

            for path in archivos:
                letra, trial = _parse_filename(path, usuario)
                if letra is None:
                    if verbose:
                        print(f"[AVISO] No se pudo parsear el nombre: {path}")
                    continue

                try:
                    signals = _load_signal_csv(path, config.CHANNEL_NAMES)

                    # --- PREPROCESAMIENTO AVANZADO ---
                    # 0) Filtro Notch (60 Hz)
                    signals = apply_notch_filter(signals, config.FS, notch_freq=60.0)
                    
                    # 1) Filtro Paso de Banda (0.5 a 40 Hz)
                    signals = apply_bandpass_filter(signals, config.FS, lowcut=0.5, highcut=40.0)
                    
                    # 2) Correccion de linea base (usando el pre-estimulo 0.0-0.5s)
                    if config.APPLY_BASELINE_CORRECTION:
                        signals = apply_baseline_correction(
                            signals, config.FS, config.BASELINE_WINDOW_S
                        )
                        
                    # 3) Eliminacion de Artefactos Biologicos (ICA)
                    signals = apply_ica_artifact_removal(signals)
                    
                    # 4) Re-referenciacion Espacial (CAR)
                    signals = apply_car_rereference(signals)
                    
                    # 5) Normalizacion y Escalamiento (Z-score)
                    signals = apply_zscore_normalization(signals)
                    # ---------------------------------

                    # 2) Recorte opcional a la ventana P300 (0.5-1.2s)
                    if config.USE_P300_WINDOW_ONLY:
                        signals = _crop_p300_window(signals, config.FS)

                    n_samples = signals.shape[1]
                    # Un solo "window" = el trial (o el recorte P300) completo
                    window_size = n_samples

                    feat_df = extractor.extract_features(
                        signals,
                        channel_names=config.CHANNEL_NAMES,
                        available_channel_names=config.CHANNEL_NAMES,
                        window_size=window_size,
                        overlap=config.WINDOW_OVERLAP,
                    )
                    # Si por algun motivo salieran varias ventanas, se promedian
                    # para dejar un solo vector de caracteristicas por trial.
                    feat_row = feat_df.mean(axis=0, numeric_only=True).to_dict()

                    feat_row["usuario"] = usuario
                    feat_row["tpComando"] = tp
                    feat_row["letra"] = letra
                    feat_row["trial"] = trial
                    feat_row["label"] = letra  # 40 clases posibles: A-Z, 0-9, simbolos control
                    rows.append(feat_row)
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    if verbose:
                        print(f"[ERROR] {path}: {e}")

    if verbose:
        print(f"\nTrials procesados OK: {n_ok} | fallidos: {n_fail}")

    if not rows:
        raise RuntimeError(
            "No se extrajo ningun trial. Revisa config.BASE_DIR, config.USUARIOS "
            "y la estructura de carpetas."
        )

    df = pd.DataFrame(rows)
    meta_cols = ["usuario", "tpComando", "letra", "trial", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]
    return df


def average_repetitions(df, n_rep=3, seed=42):
    """
    Promedia grupos de `n_rep` trials de la MISMA clase (misma letra/numero/
    simbolo) para simular la acumulacion de varias repeticiones del mismo
    estimulo, tal como funcionan los speller P300 reales (nunca deciden con
    un solo flash: promedian N repeticiones para subir la relacion señal-ruido
    antes de clasificar).

    IMPORTANTE - implica un trade-off velocidad vs precision: se necesitan
    n_rep presentaciones del mismo estimulo antes de poder predecir un
    comando en tiempo real (en vez de 1). A cambio, la accuracy sube mucho
    (medido empiricamente: n_rep=1 -> ~0.30 acc, n_rep=3 -> ~0.41 acc,
    n_rep=5 -> ~0.41 acc con este dataset).

    df: DataFrame de caracteristicas (salida de build_dataset), debe tener
        columna 'label' y todas las columnas de caracteristicas numericas.
    n_rep: cuantos trials se promedian por grupo.
    Retorna: nuevo DataFrame, mismas columnas, con menos filas
             (aprox len(df) // n_rep).
    """
    rng = np.random.default_rng(seed)
    meta_cols = ["usuario", "tpComando", "letra", "trial", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    rows = []
    for label, group in df.groupby("label"):
        idxs = group.index.to_numpy().copy()
        rng.shuffle(idxs)
        n_groups = len(idxs) // n_rep
        for g in range(n_groups):
            sel = idxs[g * n_rep:(g + 1) * n_rep]
            chunk = df.loc[sel]
            avg_feats = chunk[feature_cols].mean(axis=0).to_dict()
            avg_feats["label"] = label
            avg_feats["letra"] = chunk["letra"].iloc[0]
            avg_feats["tpComando"] = chunk["tpComando"].iloc[0]
            avg_feats["usuario"] = chunk["usuario"].iloc[0]
            avg_feats["trial"] = f"avg_{g}"
            rows.append(avg_feats)

    out = pd.DataFrame(rows)
    return out[meta_cols + feature_cols]


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv(config.FEATURES_CSV, index=False)
    print(f"\nDataset guardado en: {config.FEATURES_CSV}")
    print(f"Shape: {dataset.shape}")
    print(f"Clases (labels) encontradas: {sorted(dataset['label'].unique())}")
