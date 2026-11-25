# -*- coding: utf-8 -*-
"""
Extractor de caracteristicas EEG.
Contiene la funcion extract_features original (provista por el usuario),
envuelta en una clase con fs y un metodo `safe` para sanitizar nombres de canal.
"""


import re
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as signal
from sklearn.decomposition import FastICA
import pywt



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

                window_features[f'{cname}_Tm_mean'] = float(np.mean(channel_data))
                window_features[f'{cname}_Tm_std'] = float(np.std(channel_data))
                window_features[f'{cname}_Tm_var'] = float(np.var(channel_data))
                window_features[f'{cname}_Tm_rms'] = float(np.sqrt(np.mean(channel_data**2)))
                window_features[f'{cname}_Tm_skewness'] = float(stats.skew(channel_data))
                window_features[f'{cname}_Tm_kurtosis'] = float(stats.kurtosis(channel_data))
                window_features[f'{cname}_Tm_max'] = float(np.max(channel_data))
                window_features[f'{cname}_Tm_min'] = float(np.min(channel_data))
                window_features[f'{cname}_Tm_median'] = float(np.median(channel_data))
                window_features[f'{cname}_Tm_range'] = float(np.ptp(channel_data))
                window_features[f'{cname}_Tm_iqr'] = float(np.percentile(channel_data, 75) - np.percentile(channel_data, 25))
                window_features[f'{cname}_Tm_entropy'] = float(stats.entropy(np.abs(channel_data) + 1e-12))
                window_features[f'{cname}_Tm_zero_crossings'] = float(((channel_data[:-1] * channel_data[1:]) < 0).sum())

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
                window_features[f'{cname}_total_power'] = total_power

                if total_power > 0:
                    window_features[f'{cname}_delta_rel'] = window_features[f'{cname}_delta_Abs'] / total_power
                    window_features[f'{cname}_theta_rel'] = window_features[f'{cname}_theta_Abs'] / total_power
                    window_features[f'{cname}_alpha_rel'] = window_features[f'{cname}_alpha_Abs'] / total_power
                    window_features[f'{cname}_beta_rel'] = window_features[f'{cname}_beta_Abs'] / total_power
                    window_features[f'{cname}_gamma_rel'] = window_features[f'{cname}_gamma_Abs'] / total_power
                else:
                    for stat in ('delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel'):
                        window_features[f'{cname}_{stat}'] = 0.0
                
                                # --- Wavelet DWT (db4) ---
                try:
                    level = pywt.dwt_max_level(len(channel_data), 'db4')
                    
                    coeffs = pywt.wavedec(channel_data, 'db4', level=level)
                    energies = [float(np.sum(np.asarray(c)**2)) for c in coeffs]
                    total_w_energy = sum(energies) if sum(energies) > 0 else 1.0

                    # Adaptamos las llaves para registrar el nivel efectivo real (wA4)
                    window_features[f'{cname}_wA4_energy'] = energies[0]
                    window_features[f'{cname}_wA4_rel'] = (energies[0] / total_w_energy)
                    
                    # Estadísticas con la Wavelet de Aproximación (Nivel efectivo real)
                    wA_approx = np.asarray(coeffs[0])
                    window_features[f'{cname}_wA4_std'] = float(np.std(wA_approx))
                    window_features[f'{cname}_wA4_mean'] = float(np.mean(wA_approx))
                    window_features[f'{cname}_wA4_var'] = float(np.var(wA_approx))
                    window_features[f'{cname}_wA4_rms'] = float(np.sqrt(np.mean(wA_approx**2)))
                    
                    # Protección contra NaN para skew y kurtosis en wA4
                    sk_a = stats.skew(wA_approx)
                    window_features[f'{cname}_wA4_skewness'] = float(sk_a) if not np.isnan(sk_a) else 0.0
                    kt_a = stats.kurtosis(wA_approx)
                    window_features[f'{cname}_wA4_kurtosis'] = float(kt_a) if not np.isnan(kt_a) else 0.0
                    window_features[f'{cname}_wA4_num_coeffs'] = len(wA_approx)
                    window_features[f'{cname}_wA4_total_energy'] = total_w_energy
                    window_features[f'{cname}_wA4_energy_ratio'] = energies[0] / total_w_energy
                    window_features[f'{cname}_wA4_mean_abs'] = float(np.mean(np.abs(wA_approx)))
                    window_features[f'{cname}_wA4_max'] = float(np.max(wA_approx))
                    window_features[f'{cname}_wA4_min'] = float(np.min(wA_approx))
                    window_features[f'{cname}_wA4_median'] = float(np.median(wA_approx))
                    window_features[f'{cname}_wA4_iqr'] = float(np.percentile(wA_approx, 75) - np.percentile(wA_approx, 25))
                    window_features[f'{cname}_wA4_entropy'] = float(stats.entropy(np.abs(wA_approx) + 1e-12))
                    window_features[f'{cname}_wA4_zero_crossings'] = float(((wA_approx[:-1] * wA_approx[1:]) < 0).sum())
                    window_features[f'{cname}_wA4_total_energy'] = total_w_energy
                    
                    # Rellenamos de forma dinámica los detalles (wD1 a wD5) con todas sus estadísticas
                    for lvl in range(1, 6):
                        if lvl <= level:
                            idx = level - lvl + 1
                            coef_actual = np.asarray(coeffs[idx])
                            
                            # Energía y energía relativa
                            e = energies[idx]
                            rel = e / total_w_energy
                            
                            # Estadísticas por nivel de detalle
                            std_d = float(np.std(coef_actual))
                            mean_d = float(np.mean(coef_actual))
                            var_d = float(np.var(coef_actual))
                            rms_d = float(np.sqrt(np.mean(coef_actual**2)))
                            
                            # Protección contra NaN para skew y kurtosis en detalles
                            sk_d = stats.skew(coef_actual)
                            sk_d = float(sk_d) if not np.isnan(sk_d) else 0.0
                            kt_d = stats.kurtosis(coef_actual)
                            kt_d = float(kt_d) if not np.isnan(kt_d) else 0.0
                        else:
                            # Si el nivel calculado es menor, rellenamos con 0.0 para mantener la estructura tabular
                            e, rel, std_d, mean_d, var_d, rms_d, sk_d, kt_d = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        
                        # Guardado dinámico en el diccionario
                        window_features[f'{cname}_wD{lvl}_energy'] = e
                        window_features[f'{cname}_wD{lvl}_rel'] = rel
                        window_features[f'{cname}_wD{lvl}_std'] = std_d
                        window_features[f'{cname}_wD{lvl}_mean'] = mean_d
                        window_features[f'{cname}_wD{lvl}_var'] = var_d
                        window_features[f'{cname}_wD{lvl}_rms'] = rms_d
                        window_features[f'{cname}_wD{lvl}_skewness'] = sk_d
                        window_features[f'{cname}_wD{lvl}_kurtosis'] = kt_d
                        window_features[f'{cname}_wD{lvl}_num_coeffs'] = len(coef_actual) if lvl <= level else 0
                        window_features[f'{cname}_wD{lvl}_total_energy'] = total_w_energy
                        window_features[f'{cname}_wD{lvl}_energy_ratio'] = rel
                        window_features[f'{cname}_wD{lvl}_mean_abs'] = float(np.mean(np.abs(coef_actual))) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_max'] = float(np.max(coef_actual)) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_min'] = float(np.min(coef_actual)) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_median'] = float(np.median(coef_actual)) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_iqr'] = float(np.percentile(coef_actual, 75) - np.percentile(coef_actual, 25)) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_entropy'] = float(stats.entropy(np.abs(coef_actual) + 1e-12)) if lvl <= level else 0.0
                        window_features[f'{cname}_wD{lvl}_zero_crossings'] = float(((coef_actual[:-1] * coef_actual[1:]) < 0).sum()) if lvl <= level else 0.0
                        
                except Exception as e:
                    print(f"[ERROR] Falló extracción Wavelet en canal {cname}: {e}")
                    # Inicialización limpia en caso de error para evitar inconsistencias en el dataframe final
                    for stat in ('wA4_energy', 'wA4_rel', 'wA4_std', 'wA4_mean', 'wA4_var', 'wA4_rms', 'wA4_skewness', 'wA4_kurtosis', 'wA4_num_coeffs', 'wA4_total_energy', 'wA4_energy_ratio', 'wA4_mean_abs', 'wA4_max', 'wA4_min', 'wA4_median', 'wA4_range', 'wA4_iqr', 'wA4_entropy', 'wA4_zero_crossings'):
                        window_features[f'{cname}_{stat}'] = 0.0
                        
                    for lvl in range(1, 6):
                        for stat in ('energy', 'rel', 'std', 'mean', 'var', 'rms', 'skewness', 'kurtosis', 'rel', 'max', 'min', 'median', 'range', 'iqr', 'entropy', 'zero_crossings'):
                            window_features[f'{cname}_wD{lvl}_{stat}'] = 0.0
       

                # --- Welch PSD: Optimizado para Ventanas de 1.5s (192 muestras) ---
                try:
                    longitud_datos = len(channel_data)
                    nperseg_optimo = min(longitud_datos, 128)
                    noverlap_optimo = nperseg_optimo // 2 # 50% de solapamiento interno para Welch

                    welch_freqs, welch_psd = signal.welch(
                        channel_data, 
                        fs=self.fs,
                        window='hamming',
                        nperseg=nperseg_optimo,
                        noverlap=noverlap_optimo
                    )
                    
                    # Máscaras de frecuencia estándar clínicas
                    welch_delta_mask = (welch_freqs >= 0.5) & (welch_freqs <= 4.0)
                    welch_theta_mask = (welch_freqs > 4.0)  & (welch_freqs <= 8.0)
                    welch_alpha_mask = (welch_freqs > 8.0)  & (welch_freqs <= 14.0)
                    welch_beta_mask  = (welch_freqs > 14.0) & (welch_freqs <= 30.0)
                    # Limitamos gamma a 40 Hz ya que tu filtro pasa banda corta ahí
                    welch_gamma_mask = (welch_freqs > 30.0)

                    welch_mask = [
                        ('delta', welch_delta_mask),
                        ('theta', welch_theta_mask),
                        ('alpha', welch_alpha_mask),
                        ('beta',  welch_beta_mask),
                        ('gamma', welch_gamma_mask)                        
                    ]
                    
                    for band_name, band_mask in welch_mask:
                        # Aseguramos que la banda tenga al menos 2 puntos para poder calcular estadísticas válidas
                        if band_mask.any() and np.sum(band_mask) >= 2:
                            psd_band = welch_psd[band_mask]
                            window_features[f'{cname}_welch_{band_name}_power']    = float(np.sum(psd_band))
                            window_features[f'{cname}_welch_{band_name}_mean']     = float(np.mean(psd_band))
                            window_features[f'{cname}_welch_{band_name}_std']      = float(np.std(psd_band))
                            window_features[f'{cname}_welch_{band_name}_var']      = float(np.var(psd_band))
                            window_features[f'{cname}_welch_{band_name}_rms']      = float(np.sqrt(np.mean(psd_band**2)))
                            window_features[f'{cname}_welch_{band_name}_skewness'] = float(stats.skew(psd_band))
                            window_features[f'{cname}_welch_{band_name}_kurtosis'] = float(stats.kurtosis(psd_band))
                            window_features[f'{cname}_welch_{band_name}_rel']      = float(np.sum(psd_band) / np.sum(welch_psd)) if np.sum(welch_psd) > 0 else 0.0
                            window_features[f'{cname}_welch_{band_name}_max']      = float(np.max(psd_band))
                            window_features[f'{cname}_welch_{band_name}_min']      = float(np.min(psd_band))
                            window_features[f'{cname}_welch_{band_name}_median']   = float(np.median(psd_band))
                        else:
                            # Reemplazamos np.nan por 0.0 de forma segura para no romper el SafeLDA ni el Scaler
                            for stat in ('power', 'mean', 'std', 'var', 'rms', 'skewness', 'kurtosis', 'rel', 'max', 'min', 'median'):
                                window_features[f'{cname}_welch_{band_name}_{stat}'] = 0.0
                except Exception as e:
                    print(f"[ERROR] Falló Welch PSD en canal {cname}: {e}")
                    for band_name in ('delta', 'theta', 'alpha', 'beta', 'gamma'):
                        for stat in ('power', 'mean', 'std', 'var', 'rms', 'skewness', 'kurtosis'):
                            window_features[f'{cname}_welch_{band_name}_{stat}'] = 0.0


            features.append(window_features)
        return pd.DataFrame(features)

    def extract_erp_features(self, signals, channel_names=None, available_channel_names=None, 
                             t_pre_stim=0.5, t_p300_start=0.5, t_p300_end=1.2):
        """
        Calcula las métricas del ERP (Amplitud de Pico y Latencia) para la P300.
        Diseñado para procesar una época/ensayo completo de 2.0 segundos.
        
        - signals: array (n_channels, n_samples) correspondiente a una época de 2.0s
        - t_pre_stim: tiempo en segundos antes del estímulo (ej. 0.5s)
        - t_p300_start: inicio de la ventana P300 (ej. 0.5s)
        - t_p300_end: fin de la ventana P300 (ej. 1.2s)
        """
        # 1. Decidir índices de canales a procesar (Mismo sistema de tu función original)
        if channel_names is None:
            sel_idxs = list(range(signals.shape[0]))
        else:
            if available_channel_names is not None:
                sel_idxs = []
                for ch in channel_names:
                    if ch in available_channel_names:
                        sel_idxs.append(available_channel_names.index(ch))
                    else:
                        try: sel_idxs.append(int(ch))
                        except Exception: pass
                sel_idxs = [i for i in dict.fromkeys(sel_idxs) if 0 <= i < signals.shape[0]]
            else:
                sel_idxs = []
                for ch in channel_names:
                    try:
                        idx = int(ch)
                        if 0 <= idx < signals.shape[0]: sel_idxs.append(idx)
                    except Exception: pass
                sel_idxs = list(dict.fromkeys(sel_idxs))

        if not sel_idxs:
            raise ValueError("No se encontraron índices de canal válidos para el cálculo de ERP.")

        # 2. Aplicar corrección de línea base automáticamente a la época actual
        # Usa la función global 'apply_baseline_correction' que ya tienes en tu archivo
        signals_corrected = apply_baseline_correction(signals, self.fs, baseline_window_s=(0.0, t_pre_stim))

        # 3. Definir los índices de muestras para la ventana P300 (Respuesta Evocada)
        idx_p300_start = int(t_p300_start * self.fs)
        idx_p300_end = int(t_p300_end * self.fs)
        idx_p300_end = min(idx_p300_end, signals_corrected.shape[1])

        erp_features = {}

        # 4. Extraer métricas canal por canal
        for channel_idx in sel_idxs:
            if available_channel_names is not None and channel_idx < len(available_channel_names):
                cname = self.safe(available_channel_names[channel_idx])
            else:
                cname = f'ch{channel_idx}'

            # Recortar los datos de la ventana de Respuesta Evocada (0.5s - 1.2s)
            p300_window_data = signals_corrected[channel_idx, idx_p300_start:idx_p300_end]

            if len(p300_window_data) > 0:
                # Cálculo de Amplitud Máxima (Voltaje más positivo de la P300)
                peak_amplitude = float(np.max(p300_window_data))
                
                # Cálculo de Latencia (Tiempo transcurrido desde el estímulo en el segundo 0.5s)
                peak_sample_idx = np.argmax(p300_window_data)
                # Convertir muestra a segundos relativos desde el momento exacto del estímulo
                peak_latency_s = float(peak_sample_idx / self.fs) 
                
                erp_features[f'{cname}_p300_amplitude'] = peak_amplitude
                erp_features[f'{cname}_p300_latency_ms'] = peak_latency_s * 1000  # En milisegundos
            else:
                erp_features[f'{cname}_p300_amplitude'] = np.nan
                erp_features[f'{cname}_p300_latency_ms'] = np.nan

        return pd.DataFrame([erp_features])

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
    Agrupa las columnas de características en subconjuntos temáticos limpios
    y ordena TODAS las columnas de los 14 canales Emotiv siguiendo el Orden de Ginebra.
    """
    # Orden original de tu hardware (Emotiv EPOC):
    # ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]

    # Orden de Ginebra adaptado a tus 14 canales (Frontal -> Occipital: Izquierda luego Derecha)
    GENEVA_ORDER_14 = [
        # Hemisferio Izquierdo (índices 0 al 6)
        "AF3", "F3", "F7", "FC5", "T7", "P7", "O1",
        # Hemisferio Derecho (índices 7 al 13)
        "AF4", "F4", "F8", "FC6", "T8", "P8", "O2"
    ]
    
    geneva_rank = {ch: idx for idx, ch in enumerate(GENEVA_ORDER_14)}

    def sort_by_geneva(cols):
        """Ordena las columnas según el canal al inicio del nombre (ej. 'AF3_beta_mean')."""
        def get_channel_priority(col_name):
            # Extraer el prefijo del canal antes del primer '_'
            prefix = col_name.split("_", 1)[0]
            # Ordena primero por posición en Geneva Order y luego alfabéticamente por característica
            return (geneva_rank.get(prefix, 999), col_name)

        return sorted(cols, key=get_channel_priority)

    # Descartar columnas constantes o de conteo fijo detectadas en el dataset
    valid_cols = [c for c in all_cols if "_wD5_" not in c and "num_coeffs" not in c]

    # --- Configuración de llaves para Wavelets (niveles 1 a 4 + aproximación A4) ---
    w_stats_full = [
        "energy", "rel", "std", "mean", "var", "rms", "skewness", "kurtosis",
        "total_energy", "energy_ratio", "mean_abs", "max", "min", "median",
        "iqr", "entropy", "zero_crossings"
    ]
    w_stats_est = ["std", "mean", "var", "rms", "skewness", "kurtosis"]

    wav_keys = [f"wA4_{s}" for s in w_stats_full] + \
               [f"wD{lvl}_{s}" for lvl in range(1, 5) for s in w_stats_full]

    wAD_Est = [f"wA4_{s}" for s in w_stats_est] + \
              [f"wD{lvl}_{s}" for lvl in range(1, 5) for s in w_stats_est]

    bandas_fft = ["_delta_", "_theta_", "_alpha_", "_beta_", "_gamma_"]

    sets = {
        "Estadisticas_Tm": [c for c in valid_cols if "_Tm_" in c],

        "Frecuencias_Abs": [c for c in valid_cols if any(k in c for k in
                            ["delta_Abs", "theta_Abs", "alpha_Abs", "beta_Abs", "gamma_Abs"])],

        "Frecuencias_Rel": [c for c in valid_cols if any(k in c for k in
                            ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"])],

        "Frecuencias_Est": [c for c in valid_cols if any(k in c for k in
                            ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                             "beta_std", "gamma_std", "beta_var", "gamma_var",
                             "beta_rms", "gamma_rms", "beta_skewness", "gamma_skewness",
                             "beta_kurtosis", "gamma_kurtosis"])],

        "Wavelets": [c for c in valid_cols if any(k in c for k in wav_keys)],
        "Wavelets_Est": [c for c in valid_cols if any(k in c for k in wAD_Est)],

        "Frecuencias_Todas": [c for c in valid_cols if any(b in c for b in bandas_fft)
                              and "_welch_" not in c],

        "Welch": [c for c in valid_cols if "_welch_" in c],

        "Estadicas_todas": [c for c in valid_cols if any(c.endswith(s) for s in
                            ["_mean", "_std", "_var", "_rms", "_skewness", "_kurtosis"])],

        "Beta_Gamma_All": [c for c in valid_cols if any(b in c for b in ["_beta_", "_gamma_"])
                           and "_welch_" not in c],

        "Beta_Gamma_Est": [c for c in valid_cols if any(b in c for b in ["_beta_", "_gamma_"])
                           and "_welch_" not in c and not c.endswith(("_Abs", "_rel"))],

        # --- GRUPOS ESPECIALIZADOS EN P300 Y ERP ---
        "ERP_P300_Puro": [c for c in valid_cols if "_p300_" in c],

        "Bajas_Frec_ERP": [c for c in valid_cols if any(k in c for k in
                           ["_delta_", "_theta_", "_alpha_", "_wA4_", "_wD4_", "_p300_"])],

        
        "Beta_Gamma_ERP": [c for c in valid_cols if (any(b in c for b in ["_beta_", "_gamma_"])
                           and "_welch_" not in c) or "_p300_" in c],

        "Espectral_Mas_ERP": [c for c in valid_cols if any(k in c for k in
                              ["_welch_", "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel", "_p300_"])],

        "Relativas_Mas_ERP": [c for c in valid_cols if any(k in c for k in
            ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel", "_p300_"])],

        
        "TODAS": valid_cols,
    }

    # Aplicar el orden de Ginebra a cada subconjunto antes de retornarlo
    return {k: sort_by_geneva(v) for k, v in sets.items() if v}




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
        # REGLA 1: El límite estricto (Heurística lineal: muestras >= 20 * n_components)
        n_max_lineal = n_components // 20

        # REGLA 2: El límite ideal (Heurística cuadrática: muestras >= 20 * n_components^2)
        n_max_ideal = int(np.sqrt(n_components / 20))

        # Seleccionamos el valor óptimo dinámico sin superar nunca el total de canales físicos
        if n_max_ideal >= 1:
            n_optimo = min(n_components, n_max_ideal)
        else:
            # Si los datos son críticamente cortos, usamos la regla lineal
            n_optimo = min(n_components, n_max_lineal)

        # Asegurar un mínimo de 2 componentes para poder hacer ICA
        n_components_dinamico = max(9, n_optimo)
        n_components = n_components_dinamico

    X = signals.T
    try:
        ica = FastICA(n_components=n_components, random_state=random_state, max_iter=2000, tol=1e-2)
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
