import numpy as np
import pandas as pd
from scipy import stats, signal
from pathlib import Path
import matplotlib.pyplot as plt
import math
import pywt


class FeatureExtractor:
    def __init__(self, fs=160):
        self.fs = fs

    def safe(self, name):
        """Sanitiza nombres de canales para usar en columnas."""
        return name.replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_')

    def extract_features(self, signals, channel_names=None, available_channel_names=None,
                        window_size=256, overlap=0.5):
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
                            print(f'Warning: canal "{ch}" no encontrado y ser� ignorado.')
                sel_idxs = [i for i in dict.fromkeys(sel_idxs) if 0 <= i < signals.shape[0]]
            else:
                sel_idxs = []
                for ch in channel_names:
                    try:
                        idx = int(ch)
                        if 0 <= idx < signals.shape[0]:
                            sel_idxs.append(idx)
                    except Exception:
                        print(f'Warning: canal "{ch}" inv�lido y ser� ignorado.')
                sel_idxs = list(dict.fromkeys(sel_idxs))
        if not sel_idxs:
            raise ValueError("No se encontraron �ndices de canal v�lidos para extraer caracter�sticas.")
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
                # 1. Calcula los valores de potencia (amplitud al cuadrado)
                power_vals = fft_vals**2
                # 2. Calcula la potencia total (Suma de toda la potencia)
                total_power = float(np.sum(power_vals))
                # 3. Potencia Absoluta de bandas (Suma de la potencia *dentro* de cada banda)
                window_features[f'{cname}_delta_Abs'] = float(np.sum(power_vals[delta_mask])) if delta_mask.any() else 0.0
                window_features[f'{cname}_theta_Abs'] = float(np.sum(power_vals[theta_mask])) if theta_mask.any() else 0.0
                window_features[f'{cname}_alpha_Abs'] = float(np.sum(power_vals[alpha_mask])) if alpha_mask.any() else 0.0
                window_features[f'{cname}_beta_Abs'] = float(np.sum(power_vals[beta_mask])) if beta_mask.any() else 0.0
                window_features[f'{cname}_gamma_Abs'] = float(np.sum(power_vals[gamma_mask])) if gamma_mask.any() else 0.0
                # 4. Potencia Relativa de bandas (Potencia de la banda / Potencia total)
                # valores estadisticos en la frecuancias de las bandas
                #medias
                window_features[f'{cname}_delta_mean'] = float(np.mean(deltavallfft)) if delta_mask.any() else np.nan
                window_features[f'{cname}_theta_mean'] = float(np.mean(thetavallfft)) if theta_mask.any() else np.nan
                window_features[f'{cname}_alpha_mean'] = float(np.mean(alphavallfft)) if alpha_mask.any() else np.nan
                window_features[f'{cname}_beta_mean'] = float(np.mean(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_mean'] = float(np.mean(gammavallfft)) if gamma_mask.any() else np.nan
                #desviasion estandar
                window_features[f'{cname}_beta_std'] = float(np.std(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_std'] = float(np.std(gammavallfft)) if gamma_mask.any() else np.nan
                # Varianza
                window_features[f'{cname}_beta_var'] = float(np.var(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_var'] = float(np.var(gammavallfft)) if gamma_mask.any() else np.nan
                # RMS
                window_features[f'{cname}_beta_rms'] = float(np.sqrt(np.mean(betavallfft**2))) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_rms'] = float(np.sqrt(np.mean(gammavallfft**2))) if gamma_mask.any() else np.nan
                # Skewness
                window_features[f'{cname}_beta_skewness'] = float(stats.skew(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_skewness'] = float(stats.skew(gammavallfft)) if gamma_mask.any() else np.nan
                # Kurtosis
                window_features[f'{cname}_beta_kurtosis'] = float(stats.kurtosis(betavallfft)) if beta_mask.any() else np.nan
                window_features[f'{cname}_gamma_kurtosis'] = float(stats.kurtosis(gammavallfft)) if gamma_mask.any() else np.nan
                if total_power > 0:
                    window_features[f'{cname}_delta_rel'] = window_features[f'{cname}_delta_Abs'] / total_power
                    window_features[f'{cname}_theta_rel'] = window_features[f'{cname}_theta_Abs'] / total_power
                    window_features[f'{cname}_alpha_rel'] = window_features[f'{cname}_alpha_Abs'] / total_power
                    window_features[f'{cname}_beta_rel'] = window_features[f'{cname}_beta_Abs'] / total_power
                    window_features[f'{cname}_gamma_rel'] = window_features[f'{cname}_gamma_Abs'] / total_power
                else:
                    # Manejar el caso de divisi�n por cero
                    window_features[f'{cname}_delta_rel'] = 0.0
                    window_features[f'{cname}_theta_rel'] = 0.0
                    window_features[f'{cname}_alpha_rel'] = 0.0
                    window_features[f'{cname}_beta_rel'] = 0.0
                    window_features[f'{cname}_gamma_rel'] = 0.0
                # --- Caracter�sticas Wavelet Discreta (DWT) usando db4 nivel 5 ---
                if pywt is None:
                    # No disponible, rellenar con ceros
                    for lvl in range(1, 6):
                        window_features[f'{cname}_wD{lvl}_energy'] = 0.0
                        window_features[f'{cname}_wD{lvl}_rel'] = 0.0
                    window_features[f'{cname}_wA5_energy'] = 0.0
                    window_features[f'{cname}_wA5_rel'] = 0.0
                else:
                    try:
                        max_level = pywt.dwt_max_level(len(channel_data), 'db4')
                        level = min(5, max_level)  # Usar el m�nimo entre 5 y el m�ximo posible
                        coeffs = pywt.wavedec(channel_data, 'db4', level=level)
                        # coeffs: [cA_level, cD_level, cD_{level-1}, ..., cD1]
                        energies = []
                        for c in coeffs:
                            energies.append(float(np.sum(np.asarray(c)**2)))
                        total_w_energy = sum(energies) if sum(energies) > 0 else 1.0
                        # approximation cA_level
                        window_features[f'{cname}_wA5_energy'] = energies[0] if level >= 5 else 0.0
                        window_features[f'{cname}_wA5_rel'] = (energies[0] / total_w_energy) if level >= 5 else 0.0
                        # details cD_level..cD1
                        for lvl in range(1, 6):
                            if lvl <= level:
                                idx = level - lvl + 1  # Para level=5, cD5 es index 1, cD1 es index 5
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
            features.append(window_features)
        return pd.DataFrame(features)



# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Procesar un archivos
    print("="*70)
    print("Obtercion de caracteristicas")
    print("="*70 + "\n")
    
    usuario= 'UserArcane'
    tpComando = 'Digit'
    #letras=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","Ñ","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    letras=["0","1","2","3","4","5","6","7","8","9"]
    path=f'results/{usuario}/{tpComando}/'
    # Ejemplo: procesar archivo User27_A_0.csv
    for trial in range(30):  # Procesar trials 0, 1 y 6
        for letra in letras:
            filename = f'Separados/{usuario}_{letra}_{trial}_post_estimulo.csv'
            print(f"Procesando archivo: {filename}")
            extractor = FeatureExtractor(fs=250)
            try:
                data = pd.read_csv(path+filename)
                channel_names = ["Oz", "Po7", "Po4", "Po3", "P4", "P3", "Po8", "Pz", "Fz", "F2", "F3", "F4", "AF3", "Cz", "AF4", "F1" ]
                signals = data.values.T  # Transponer para tener shape (n_channels, n_samples)
                features_df = extractor.extract_features(signals, channel_names=channel_names, available_channel_names=channel_names,window_size=200,overlap=0)
                print(f"Caracteristicas extraidas para {filename}:")
                print(features_df.head())
                # Guardar las caracteristicas en un nuevo archivo CSV
                #checa si existe la carpeta de salida, si no existe la crea
                output_dir = Path(path) / 'features'
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = Path(path) / f'features/{usuario}_{letra}_{trial}_features.csv'
                features_df.to_csv(output_path, index=False)
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
            print("-"*70)