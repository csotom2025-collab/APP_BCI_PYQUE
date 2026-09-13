# -*- coding: utf-8 -*-
"""
Extractor de caracteristicas EEG.
Contiene la funcion extract_features original (provista por el usuario),
envuelta en una clase con fs y un metodo `safe` para sanitizar nombres de canal.
"""

from pandas.core import window
from joblib import compressor
import re
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as signal
from sklearn.decomposition import FastICA

try:
    import pywt
except ImportError:
    pywt = None


class EEGFeatureExtractor:
    """
    Extractor de caracteristicas estadisticas, espectrales (FFT por banda) y
    wavelet (DWT db4) para señales EEG multicanal.
    """

    def __init__(self, fs):
        """
        fs: frecuencia de muestreo en Hz (necesaria para las bandas de frecuencia).
        """
        self.fs = fs

    @staticmethod
    def safe(name):
        """Sanitiza un nombre de canal para usarlo como prefijo de columna."""
        return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")

    def extract_features(self, signals, channel_names=None, available_channel_names=None,
                          window_size=60, overlap=0.5):
        """
        Extrae caracteristicas para canales seleccionados.
        - signals: array (n_channels, n_samples)
        - channel_names: list de nombres de canales a procesar OR list de indices
        - available_channel_names: lista completa de nombres de canales (si pasas nombres en channel_names)
        - window_size, overlap: parametros de ventana (window_size en muestras)
        Retorna DataFrame con columnas: <safe_channel_name>_<feature>
        """
        # decidir indices a procesar
        if channel_names is None:
            sel_idxs = list(range(signals.shape[0]))
        else:
            if available_channel_names is not None:
                sel_idxs = []
                for ch in channel_names:
                    if ch in available_channel_names:
                        sel_idxs.append(available_channel_names.index(ch))
                    else:
                        try:
                            sel_idxs.append(int(ch))
                        except Exception:
                            print(f'Warning: canal "{ch}" no encontrado y sera ignorado.')
                sel_idxs = [i for i in dict.fromkeys(sel_idxs) if 0 <= i < signals.shape[0]]
            else:
                sel_idxs = []
                for ch in channel_names:
                    try:
                        idx = int(ch)
                        if 0 <= idx < signals.shape[0]:
                            sel_idxs.append(idx)
                    except Exception:
                        print(f'Warning: canal "{ch}" invalido y sera ignorado.')
                sel_idxs = list(dict.fromkeys(sel_idxs))
        if not sel_idxs:
            raise ValueError("No se encontraron indices de canal validos para extraer caracteristicas.")

        step_size = int(window_size * (1 - overlap))
        if step_size <= 0:
            step_size = max(1, window_size // 2)
        n_windows = max(0, (signals.shape[1] - window_size) // step_size + 1)
        if n_windows <= 0:
            raise ValueError("Segmento demasiado corto para el window_size y overlap especificados.")

        features = []
        for window_idx in range(n_windows):
            start = window_idx * step_size
            end = start + window_size
            window_features = {}
            for channel_idx in sel_idxs:
                channel_data = signals[channel_idx, start:end]
                if available_channel_names is not None and channel_idx < len(available_channel_names):
                    cname = self.safe(available_channel_names[channel_idx])
                else:
                    cname = f'ch{channel_idx}'

                window_features[f'{cname}_mean'] = float(np.mean(channel_data))
                window_features[f'{cname}_std'] = float(np.std(channel_data))
                window_features[f'{cname}_var'] = float(np.var(channel_data))
                window_features[f'{cname}_rms'] = float(np.sqrt(np.mean(channel_data**2)))
                window_features[f'{cname}_skewness'] = float(stats.skew(channel_data))
                window_features[f'{cname}_kurtosis'] = float(stats.kurtosis(channel_data))

                # FFT y potencias de banda
                fft_vals = np.abs(np.fft.rfft(channel_data))
                freqs = np.fft.rfftfreq(len(channel_data), 1.0 / self.fs)
                delta_mask = (freqs >= 0.5) & (freqs <= 4)
                theta_mask = (freqs > 4) & (freqs <= 8)
                alpha_mask = (freqs > 8) & (freqs <= 14)
                beta_mask = (freqs > 14) & (freqs <= 30)
                gamma_mask = (freqs > 30)

                deltavallfft = fft_vals[delta_mask]
                thetavallfft = fft_vals[theta_mask]
                alphavallfft = fft_vals[alpha_mask]
                betavallfft = fft_vals[beta_mask]
                gammavallfft = fft_vals[gamma_mask]

                power_vals = fft_vals**2
                total_power = float(np.sum(power_vals))

                window_features[f'{cname}_delta_Abs'] = float(np.sum(power_vals[delta_mask])) if delta_mask.any() else 0.0
                window_features[f'{cname}_theta_Abs'] = float(np.sum(power_vals[theta_mask])) if theta_mask.any() else 0.0
                window_features[f'{cname}_alpha_Abs'] = float(np.sum(power_vals[alpha_mask])) if alpha_mask.any() else 0.0
                window_features[f'{cname}_beta_Abs'] = float(np.sum(power_vals[beta_mask])) if beta_mask.any() else 0.0
                window_features[f'{cname}_gamma_Abs'] = float(np.sum(power_vals[gamma_mask])) if gamma_mask.any() else 0.0

                window_features[f'{cname}_delta_mean'] = float(np.mean(deltavallfft)) if delta_mask.any() else np.nan
                window_features[f'{cname}_theta_mean'] = float(np.mean(thetavallfft)) if theta_mask.any() else np.nan
                window_features[f'{cname}_alpha_mean'] = float(np.mean(alphavallfft)) if alpha_mask.any() else np.nan
                window_features[f'{cname}_beta_mean'] = float(np.mean(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_mean'] = float(np.mean(gammavallfft)) if gamma_mask.any() else np.nan

                window_features[f'{cname}_beta_std'] = float(np.std(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_std'] = float(np.std(gammavallfft)) if gamma_mask.any() else np.nan

                window_features[f'{cname}_beta_var'] = float(np.var(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_var'] = float(np.var(gammavallfft)) if gamma_mask.any() else np.nan

                window_features[f'{cname}_beta_rms'] = float(np.sqrt(np.mean(betavallfft**2))) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_rms'] = float(np.sqrt(np.mean(gammavallfft**2))) if gamma_mask.any() else np.nan

                window_features[f'{cname}_beta_skewness'] = float(stats.skew(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_skewness'] = float(stats.skew(gammavallfft)) if gamma_mask.any() else np.nan

                window_features[f'{cname}_beta_kurtosis'] = float(stats.kurtosis(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_kurtosis'] = float(stats.kurtosis(gammavallfft)) if gamma_mask.any() else np.nan

                if total_power > 0:
                    window_features[f'{cname}_delta_rel'] = window_features[f'{cname}_delta_Abs'] / total_power
                    window_features[f'{cname}_theta_rel'] = window_features[f'{cname}_theta_Abs'] / total_power
                    window_features[f'{cname}_alpha_rel'] = window_features[f'{cname}_alpha_Abs'] / total_power
                    window_features[f'{cname}_beta_rel'] = window_features[f'{cname}_beta_Abs'] / total_power
                    window_features[f'{cname}_gamma_rel'] = window_features[f'{cname}_gamma_Abs'] / total_power
                else:
                    window_features[f'{cname}_delta_rel'] = 0.0
                    window_features[f'{cname}_theta_rel'] = 0.0
                    window_features[f'{cname}_alpha_rel'] = 0.0
                    window_features[f'{cname}_beta_rel'] = 0.0
                    window_features[f'{cname}_gamma_rel'] = 0.0

                # --- Caracteristicas Wavelet Discreta (DWT) usando db4 nivel 5 ---
                if pywt is None:
                    for lvl in range(1, 6):
                        window_features[f'{cname}_wD{lvl}_energy'] = 0.0
                        window_features[f'{cname}_wD{lvl}_rel'] = 0.0
                    window_features[f'{cname}_wA5_energy'] = 0.0
                    window_features[f'{cname}_wA5_rel'] = 0.0
                    window_features[f'{cname}_wA5_std'] = 0.0
                    window_features[f'{cname}_wA5_mean'] = 0.0
                    window_features[f'{cname}_wA5_var'] = 0.0
                    window_features[f'{cname}_wA5_rms'] = 0.0
                    window_features[f'{cname}_wA5_skewness'] = 0.0
                    window_features[f'{cname}_wA5_kurtosis'] = 0.0
                else:
                    try:
                        max_level = pywt.dwt_max_level(len(channel_data), 'db4')
                        level = min(5, max_level)
                        coeffs = pywt.wavedec(channel_data, 'db4', level=level)
                        energies = [float(np.sum(np.asarray(c)**2)) for c in coeffs]
                        total_w_energy = sum(energies) if sum(energies) > 0 else 1.0

                        window_features[f'{cname}_wA5_energy'] = energies[0] if level >= 5 else 0.0
                        window_features[f'{cname}_wA5_rel'] = (energies[0] / total_w_energy) if level >= 5 else 0.0
                        # estadisticas con la Walveltes (Aproximacion nivel 5 o mayor)
                        if level >= 5:
                            wA5 = np.asarray(coeffs[0])
                            window_features[f'{cname}_wA5_std'] = float(np.std(wA5))
                            window_features[f'{cname}_wA5_mean'] = float(np.mean(wA5))
                            window_features[f'{cname}_wA5_var'] = float(np.var(wA5))
                            window_features[f'{cname}_wA5_rms'] = float(np.sqrt(np.mean(wA5**2)))
                            window_features[f'{cname}_wA5_skewness'] = float(stats.skew(wA5))
                            window_features[f'{cname}_wA5_kurtosis'] = float(stats.kurtosis(wA5))
                        else:
                            window_features[f'{cname}_wA5_std'] = 0.0
                            window_features[f'{cname}_wA5_mean'] = 0.0
                            window_features[f'{cname}_wA5_var'] = 0.0
                            window_features[f'{cname}_wA5_rms'] = 0.0
                            window_features[f'{cname}_wA5_skewness'] = 0.0
                            window_features[f'{cname}_wA5_kurtosis'] = 0.0
                        
                        for lvl in range(1, 6):
                            if lvl <= level:
                                idx = level - lvl + 1
                                e = energies[idx]
                                rel = e / total_w_energy
                            else:
                                e = 0.0
                                rel = 0.0
                            window_features[f'{cname}_wD{lvl}_energy'] = e
                            window_features[f'{cname}_wD{lvl}_rel'] = rel
                    except Exception:
                        for lvl in range(1, 6):
                            window_features[f'{cname}_wD{lvl}_energy'] = 0.0
                            window_features[f'{cname}_wD{lvl}_rel'] = 0.0
                        window_features[f'{cname}_wA5_energy'] = 0.0
                        window_features[f'{cname}_wA5_rel'] = 0.0
                        window_features[f'{cname}_wA5_std'] = 0.0
                        window_features[f'{cname}_wA5_mean'] = 0.0
                        window_features[f'{cname}_wA5_var'] = 0.0
                        window_features[f'{cname}_wA5_rms'] = 0.0
                        window_features[f'{cname}_wA5_skewness'] = 0.0
                        window_features[f'{cname}_wA5_kurtosis'] = 0.0

            features.append(window_features)
        return pd.DataFrame(features)


def apply_baseline_correction(signals, fs, baseline_window_s=(0.0, 0.5)):
    """
    Corrige la linea base de cada canal restando la media de la ventana de
    pre-estimulo (por defecto 0.0 - 0.5s, segun el protocolo de grabacion).

    signals: array (n_channels, n_samples)
    fs: frecuencia de muestreo (Hz)
    baseline_window_s: (inicio, fin) en segundos de la ventana de pre-estimulo

    Retorna: signals corregidas, misma forma (n_channels, n_samples)
    """
    start = int(baseline_window_s[0] * fs)
    end = int(baseline_window_s[1] * fs)
    end = min(end, signals.shape[1])

    if end <= start:
        # Ventana de baseline invalida (trial demasiado corto): no se corrige
        return signals

    baseline_mean = np.mean(signals[:, start:end], axis=1, keepdims=True)  # (n_channels, 1)
    return signals - baseline_mean


def log_transform_power_columns(X_df, feature_cols):
    """
    Aplica log1p a las columnas de potencia absoluta (_Abs) y energia wavelet
    (_energy), que tienen escalas enormes y muy asimetricas (siguen una
    especie de ley de potencia). Sin este transform, dominan la varianza y
    perjudican a LDA/SVM. Mejora medida empiricamente sobre datos reales:
    accuracy ~0.24 -> ~0.27 solo con este cambio.
    """
    power_cols = [c for c in feature_cols if c in X_df.columns and
                  ('_Abs' in c or c.endswith('_energy'))]
    X_df = X_df.copy()
    for c in power_cols:
        X_df[c] = np.log1p(np.abs(X_df[c]))
    return X_df


def get_feature_sets(all_cols):
    """
    Agrupa las columnas de caracteristicas en subconjuntos tematicos
    (estadisticas, frecuencia absoluta/relativa, wavelets, etc.)
    """
    wA_keys = ["wA5","wA5_std","wA5_rms","wA5_skewness","wA5_kurtosis"]
    wD_keys = ["wD1", "wD2", "wD3", "wD4", "wD5"]
    wav_keys = wA_keys + wD_keys

    sets = {
        "Estadisticas": [c for c in all_cols if any(k in c for k in
                         ["mean", "std", "var", "rms", "skewness", "kurtosis"])
                         and not any(k in c for k in
                         ["delta_", "theta_", "alpha_", "beta_", "gamma_", "wA", "wD"])],

        "Frecuencias_Abs": [c for c in all_cols if any(k in c for k in
                            ["delta_Abs", "theta_Abs", "alpha_Abs", "beta_Abs", "gamma_Abs"])],

        "Frecuencias_Rel": [c for c in all_cols if any(k in c for k in
                            ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"])],

        "Frecuencias_Est": [c for c in all_cols if any(k in c for k in
                            ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                             "beta_std", "gamma_std", "beta_var", "gamma_var",
                             "beta_rms", "gamma_rms", "beta_skewness", "gamma_skewness",
                             "beta_kurtosis", "gamma_kurtosis"])],

        "Wavelets": [c for c in all_cols if any(k in c for k in wav_keys)],

        "Frecuencias_Todas": [c for c in all_cols if any(k in c for k in
                              ["delta_", "theta_", "alpha_", "beta_", "gamma_"])],

        "TODAS": list(all_cols),
    }
    return {k: v for k, v in sets.items() if v}


def apply_notch_filter(signals, fs, notch_freq=60.0, Q=30.0):
    """
    Filtro Notch (Rechazo de banda): Se aplica especificamente a 60 Hz 
    para eliminar la interferencia de la red electrica.
    """
    b, a = signal.iirnotch(w0=notch_freq, Q=Q, fs=fs)
    return signal.filtfilt(b, a, signals, axis=1)


def apply_bandpass_filter(signals, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Filtro Paso de Banda (Band-pass): Limita la senal a las frecuencias de interes
    (generalmente entre 0.5 Hz y 40 Hz). Elimina frecuencias muy bajas (movimiento) 
    y muy altas (ruido muscular).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, signals, axis=1)


def apply_ica_artifact_removal(signals, n_components=None, random_state=42):
    """
    Analisis de Componentes Independientes (ICA):
    Algoritmo estandar para eliminar artefactos biologicos (parpadeos, latidos).
    Descompone la senal de EEG en "fuentes independientes".
    """
    if n_components is None:
        n_components = signals.shape[0]
        
    X = signals.T
    try:
        ica = FastICA(n_components=n_components, random_state=random_state, max_iter=200, tol=1e-3)
        S_ = ica.fit_transform(X)
        
        # Eliminacion automatica y simplificada de artefactos (ej. mayor curtosis = parpadeo/ruido)
        kurt = stats.kurtosis(S_, axis=0)
        idx_max_kurt = np.argmax(np.abs(kurt))
        S_[:, idx_max_kurt] = 0  # Cero al componente mas ruidoso
        
        X_reconstructed = ica.inverse_transform(S_)
        return X_reconstructed.T
    except Exception as e:
        print("[Aviso] ICA fallo (posible corto de muestras), devolviendo senal original.", e)
        return signals


def apply_car_rereference(signals):
    """
    Re-referenciacion Espacial (Common Average Reference - CAR):
    Se resta el promedio de todos los electrodos a cada canal individual.
    Ayuda a reducir el ruido que afecta a toda la cabeza por igual.
    """
    avg = np.mean(signals, axis=0, keepdims=True)
    return signals - avg


def apply_laplacian_rereference(signals, laplacian_matrix=None):
    """
    Laplaciano Superficial (simplificado):
    Se requiere la matriz de distancias o configuracion de montaje (laplacian_matrix).
    Como fallback sin matriz, funciona como un CAR o se deja la senal original.
    """
    if laplacian_matrix is not None:
        return signals - np.dot(laplacian_matrix, signals)
    else:
        # Fallback a original o CAR si no hay matriz de montaje definida
        return apply_car_rereference(signals)


def apply_zscore_normalization(signals):
    """
    Normalizacion y Escalamiento (Z-score):
    Normalizacion (media=0, std=1) para estabilizar cambios de impedancia.
    Estabiliza los datos antes de introducirlos al modelo.
    """
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    return (signals - mean) / std
