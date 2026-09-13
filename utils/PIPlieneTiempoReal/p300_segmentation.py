# -*- coding: utf-8 -*-
"""
Segmentación de grabaciones P300 speller.

Protocolo estándar:
    - Archivo de 2 segundos (256 samples @ 128 Hz)
    - 5 flashes intercalados durante esos 2 segundos
    - Cada flash: 125 ms encendido + 75 ms apagado = 200 ms (25.6 samples @ 128 Hz)
    - Antes/después de los flashes: basal/recuperación

Estrategia: dividir en 5 segmentos, extraer features de cada uno, promediar antes
de clasificar (es lo que mejora la SNR en P300 y lo que más sube accuracy).
"""

import numpy as np
import pandas as pd
from eeg_features import EEGFeatureExtractor


def calculate_flash_times(fs=128, num_flashes=5, flash_duration=125, isi_duration=75,
                          total_duration=2.0, start_offset=0.2):
    """
    Calcula los tiempos (inicio, fin en muestras) de cada flash dentro de la
    grabación de 2 segundos.

    fs: frecuencia de muestreo (Hz)
    num_flashes: número de flashes intercalados (5 en tu protocolo)
    flash_duration: 125 ms (encendido)
    isi_duration: 75 ms (apagado)
    total_duration: 2.0 segundos de grabación
    start_offset: 0.2 segundos antes del primer flash (pre-estímulo/basal)

    Retorna: lista de tuplas (inicio_sample, fin_sample) por flash
    """
    flash_period_ms = flash_duration + isi_duration  # 200 ms total
    flash_period_s = flash_period_ms / 1000.0
    start_s = start_offset

    flash_times = []
    for i in range(num_flashes):
        flash_start_s = start_s + i * flash_period_s
        flash_end_s = flash_start_s + (flash_duration / 1000.0)

        flash_start_sample = int(np.round(flash_start_s * fs))
        flash_end_sample = int(np.round(flash_end_s * fs))

        flash_start_sample = max(0, min(flash_start_sample, int(total_duration * fs)))
        flash_end_sample = max(0, min(flash_end_sample, int(total_duration * fs)))

        if flash_start_sample < flash_end_sample:
            flash_times.append((flash_start_sample, flash_end_sample))

    return flash_times


def segment_flashes(signals, fs=128, num_flashes=5, flash_duration=125,
                    isi_duration=75, total_duration=2.0, start_offset=0.2,
                    window_before=50, window_after=50):
    """
    Divide la grabación de 2 segundos en 5 segmentos (flashes) y expande
    cada uno con ventanas de contexto pre/post para capturar la respuesta P300
    completa (que típicamente ocurre 200-600 ms post-flash).

    signals: array (n_channels, n_samples) donde n_samples = 256 @ 128 Hz
    window_before: muestras para incluir ANTES del flash (pre-estímulo)
    window_after: muestras para incluir DESPUÉS del flash (post-estímulo / respuesta P300)

    Retorna: lista de arrays (n_channels, n_muestras_segmento) por flash
    """
    flash_times = calculate_flash_times(fs, num_flashes, flash_duration,
                                         isi_duration, total_duration, start_offset)

    flash_segments = []
    for flash_start, flash_end in flash_times:
        seg_start = max(0, flash_start - window_before)
        seg_end = min(signals.shape[1], flash_end + window_after)
        seg = signals[:, seg_start:seg_end]
        if seg.shape[1] > 0:
            flash_segments.append(seg)

    return flash_segments


def extract_and_average_flashes(signals, feature_cols, extractor, fs=128,
                                 num_flashes=5, flash_duration=125, isi_duration=75,
                                 total_duration=2.0, start_offset=0.2,
                                 window_before=50, window_after=50, verbose=False):
    """
    Segmenta en 5 flashes, extrae características de cada uno, y retorna
    el promedio (la estrategia que maximiza SNR en P300 spellers).

    signals: (n_channels, n_samples)
    feature_cols: columnas de características esperadas (del modelo)
    extractor: EEGFeatureExtractor(fs=...)
    ... resto de parámetros igual a segment_flashes

    Retorna: DataFrame de 1 fila con características promediadas de los 5 flashes
    """
    flash_segments = segment_flashes(signals, fs, num_flashes, flash_duration,
                                     isi_duration, total_duration, start_offset,
                                     window_before, window_after)

    if verbose:
        print(f"Segmentados {len(flash_segments)} flashes")

    feat_dfs = []
    for i, seg in enumerate(flash_segments):
        feat_df = extractor.extract_features(
            seg,
            channel_names=None,  # ya está segmentado
            available_channel_names=None,
            window_size=seg.shape[1],
            overlap=0.0,
        )
        feat_dfs.append(feat_df.mean(axis=0, numeric_only=True))
        if verbose:
            print(f"  Flash {i+1}: shape {seg.shape}, features extraídas")

    # Promediar características de los 5 flashes
    avg_feats = pd.concat(feat_dfs, axis=1).mean(axis=1).to_frame().T
    if verbose:
        print(f"Características promediadas de {len(feat_dfs)} flashes")

    return avg_feats


def extract_with_flash_segmentation(signals, feature_cols, extractor,
                                    apply_baseline=True, baseline_window_s=(0.0, 0.5),
                                    use_p300_window=False, p300_window_s=(0.5, 1.2),
                                    log_transform_power=True, fs=128, verbose=False):
    """
    Flujo completo similar a _signals_to_feature_row() pero con segmentación
    de los 5 flashes internamente (el usuario no lo ve).

    1. Correccion de linea base (opcional)
    2. Segmentar en 5 flashes
    3. Extraer características de cada uno
    4. Promediar
    5. Log-transform (opcional)

    Retorna: DataFrame de 1 fila, listo para el pipeline
    """
    from eeg_features import apply_baseline_correction, log_transform_power_columns

    if apply_baseline:
        signals = apply_baseline_correction(signals, fs, baseline_window_s)

    if use_p300_window:
        start = int(p300_window_s[0] * fs)
        end = int(p300_window_s[1] * fs)
        end = min(end, signals.shape[1])
        if start < end:
            signals = signals[:, start:end]

    feat_row = extract_and_average_flashes(
        signals, feature_cols, extractor, fs=fs,
        num_flashes=1, flash_duration=125, isi_duration=75,
        total_duration=2.0, start_offset=0.2,
        window_before=50, window_after=50,
        verbose=verbose
    )

    if log_transform_power:
        feat_row = log_transform_power_columns(feat_row, feat_row.columns)

    return feat_row
