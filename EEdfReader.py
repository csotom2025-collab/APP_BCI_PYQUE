import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal,stats
from pathlib import Path
import re
import re
try:
    import pywt
except Exception:
    pywt = None

class EEGEDFReader:
    def __init__(self, data_path):
        """
        Inicializa el lector de archivos EDF para EEG
        Args:
            data_path (str or Path): Ruta donde se encuentran los archivos EDF
        """
        self.data_path = str(data_path)
        self.sampling_rate = 160  # valor por defecto; se reemplaza si el archivo contiene info
    
    def list_edf_files(self, data_path=None, num=None,user=None):
        """Devuelve lista de archivos .edf en data_path (subdirectorios incluidos).
        Si num is None busca '*.edf', si num='0'..'9' busca '*_<num>.edf'."""
        if data_path is None:
            data_path = self.data_path
        data_path = Path(data_path)
        if num is None:
            pattern = '*.edf'
        else:
            if user is not None:
                pattern = f'{user}_{num}.edf'
            else:
                pattern = f'*_{num}.edf'
        return sorted([p for p in data_path.rglob(pattern) if p.is_file()])

    def read_edf_file(self, filename):
        """
        Lee un archivo EDF usando mne y extrae la información
        Args:
            filename (str or Path): Ruta o nombre del archivo EDF
        Returns:
            dict: Diccionario con datos y metadatos del EEG
        """
        # construir ruta completa si se pasa solo nombre
        if isinstance(filename, (str, Path)):
            fp = Path(filename)
            if not fp.is_absolute():
                file_path = Path(self.data_path) / fp
            else:
                file_path = fp
        else:
            raise ValueError("filename debe ser str o Path")
        try:
            print(f"Intentando leer el archivo EDF: {file_path}")
            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
            data = raw.get_data()  # shape (n_channels, n_samples)
            signal_labels = raw.ch_names
            sampling_rate = raw.info.get('sfreq', self.sampling_rate)
            file_duration = data.shape[1] / sampling_rate if sampling_rate and data.shape[1] else 0.0
            # actualizar sampling_rate por si viene del archivo
            self.sampling_rate = sampling_rate
            metadata = {
                'filename': str(file_path.name),
                'filepath': str(file_path),
                'n_channels': data.shape[0],
                'signal_labels': signal_labels,
                'duration': file_duration,
                'sampling_rate': sampling_rate
            }
            return {
                'metadata': metadata,
                'signals': data,
                'channel_names': signal_labels
            }
        except Exception as e:
            print(f"Error leyendo el archivo {filename}: {str(e)}")
            return None
    
    def preprocess_eeg(self, signals, lowcut=1.0, highcut=40.0, fs=None):
        """
        Preprocesa las señales EEG aplicando filtro bandpass
        Args:
            signals (numpy.array): Array de señales EEG (n_channels, n_samples)
            lowcut (float): Frecuencia de corte baja (Hz)
            highcut (float): Frecuencia de corte alta (Hz)
            fs (float): frecuencia de muestreo; si None usa self.sampling_rate
        Returns:
            numpy.array: Señales filtradas
        """
        if fs is None:
            fs = self.sampling_rate
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signals = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            filtered_signals[i, :] = signal.filtfilt(b, a, signals[i, :])
        # Notch filter de 40 Hz  remover ruido de línea
        F_notch = 40.0
        Q = 30.0  # Factor de calidad
        b_notch, a_notch = signal.iirnotch(F_notch, Q, fs)
        for i in range(signals.shape[0]):
            filtered_signals[i, :] = signal.filtfilt(b_notch, a_notch, filtered_signals[i, :])
        return filtered_signals
    
    def plot_eeg_channels(self, data, channels_to_plot=None, time_window=None):
        """
        Grafica múltiples canales de EEG
        Args:
            data (dict): Datos del EEG devueltos por read_edf_file
            channels_to_plot (list): Lista de índices de canales a graficar
            time_window (tuple): Tupla (start, end) para el tiempo en segundos
        """
        if channels_to_plot is None:
            channels_to_plot = list(range(min(8, data['signals'].shape[0])))
        signals = data['signals']
        metadata = data['metadata']
        fs = metadata.get('sampling_rate', self.sampling_rate)
        time = np.arange(signals.shape[1]) / fs
        if time_window:
            start_idx = int(time_window[0] * fs)
            end_idx = int(time_window[1] * fs)
            start_idx = max(0, start_idx)
            end_idx = min(signals.shape[1], end_idx)
            signals = signals[:, start_idx:end_idx]
            time = time[start_idx:end_idx]
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 2.5 * len(channels_to_plot)))
        if len(channels_to_plot) == 1:
            axes = [axes]
        for i, channel_idx in enumerate(channels_to_plot):
            axes[i].plot(time, signals[channel_idx, :])
            name = data['channel_names'][channel_idx] if 'channel_names' in data else f'Ch{channel_idx}'
            axes[i].set_ylabel(f'Channel {channel_idx}\n{name}')
            axes[i].grid(True)
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(f'EEG Signals - {metadata.get("filename","")}')
        plt.tight_layout()
        plt.show()
        
    def safe(self,name):
            return re.sub(r'[^0-9A-Za-z_]', '_', str(name)).strip('_')
    
    def extract_features(self, signals, channel_names=None, available_channel_names=None,
                        window_size=256, overlap=0.5):
        """
        Extrae características para canales seleccionados.
        - signals: array (n_channels, n_samples)
        - channel_names: list de nombres de canales a procesar OR list de índices
        - available_channel_names: lista completa de nombres de canales (si pasas nombres en channel_names)
        - window_size, overlap: parámetros de ventana (window_size en muestras)
        Retorna DataFrame con columnas: <safe_channel_name>_<feature>
        """
        
        
        # decidir índices a procesar
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
                            print(f'Warning: canal "{ch}" no encontrado y será ignorado.')
                sel_idxs = [i for i in dict.fromkeys(sel_idxs) if 0 <= i < signals.shape[0]]
            else:
                sel_idxs = []
                for ch in channel_names:
                    try:
                        idx = int(ch)
                        if 0 <= idx < signals.shape[0]:
                            sel_idxs.append(idx)
                    except Exception:
                        print(f'Warning: canal "{ch}" inválido y será ignorado.')
                sel_idxs = list(dict.fromkeys(sel_idxs))
        if not sel_idxs:
            raise ValueError("No se encontraron índices de canal válidos para extraer características.")
        step_size = int(window_size * (1 - overlap))
        if step_size <= 0:
            step_size = max(1, window_size // 2)
        n_windows = max(0, (signals.shape[1] - window_size) // step_size + 1)
        if n_windows <= 0:
            raise ValueError("Segmento demasiado corto para el window_size y overlap especificados.")
        features = []
        fs = self.sampling_rate
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
                freqs = np.fft.rfftfreq(len(channel_data), 1.0 / fs)
                delta_mask = (freqs >= 0.5) & (freqs <= 4)
                theta_mask = (freqs > 4) & (freqs <= 8)
                alpha_mask = (freqs > 8) & (freqs <= 14)
                beta_mask = (freqs > 14) & (freqs <= 30)
                gamma_mask = (freqs > 30)
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
                if total_power > 0:
                    window_features[f'{cname}_delta_rel'] = window_features[f'{cname}_delta_Abs'] / total_power
                    window_features[f'{cname}_theta_rel'] = window_features[f'{cname}_theta_Abs'] / total_power
                    window_features[f'{cname}_alpha_rel'] = window_features[f'{cname}_alpha_Abs'] / total_power
                    window_features[f'{cname}_beta_rel'] = window_features[f'{cname}_beta_Abs'] / total_power
                    window_features[f'{cname}_gamma_rel'] = window_features[f'{cname}_gamma_Abs'] / total_power
                else:
                    # Manejar el caso de división por cero
                    window_features[f'{cname}_delta_rel'] = 0.0
                    window_features[f'{cname}_theta_rel'] = 0.0
                    window_features[f'{cname}_alpha_rel'] = 0.0
                    window_features[f'{cname}_beta_rel'] = 0.0
                    window_features[f'{cname}_gamma_rel'] = 0.0
                # --- Características Wavelet Discreta (DWT) usando db4 nivel 5 ---
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
                        level = min(5, max_level)  # Usar el mínimo entre 5 y el máximo posible
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
    
    def plot_feactureres(self, features_df, channel_names, filename,
                        window_size=256, overlap=0.5,
                        out_dir=Path(r"D:/SeñaelesEEGpy/results"), save=True, show=True):
        """
        Grafica las características extraídas para los canales indicados
        en una CUADRÍCULA DE 3 FILAS x 2 COLUMNAS por cada canal.
        
        Busca columnas del tipo <safe_channel_name>_<feature>.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Asumimos que self.sampling_rate existe en la clase
        try:
            fs = self.sampling_rate
        except AttributeError:
            print("Error: El objeto no tiene el atributo 'sampling_rate'.")
            print("Por favor, asegúrate de que 'self.sampling_rate' esté definido.")
            fs = 1 # Valor por defecto para evitar que falle, ajusta si es necesario

        step = int(window_size * (1 - overlap))
        if step <= 0:
            step = max(1, window_size // 2)
        
        times = (np.arange(len(features_df)) * step + window_size / 2) / fs
        
        # --- CAMBIO 1: Definir el layout y las características ---
        # Lista fija de 6 características para la cuadrícula 3x2
        features = ['mean', 'std', 'var', 'rms', 'skewness', 'kurtosis']
        n_rows = 3
        n_cols = 2
        
        last_out_path = None # Para guardar la ruta del último archivo

        # Iterar sobre cada canal solicitado
        for ch in channel_names:
            
            # Asumimos que self.safe() existe en la clase
            try:
                cname = self.safe(ch)
            except AttributeError:
                print("Error: El objeto no tiene el método 'safe()'.")
                cname = ch.replace(" ", "_").replace(".", "-") # Implementación 'safe' simple

            # --- CAMBIO 2: Crear la Figura 3x2 ---
            # Creamos la figura con 3 filas y 2 columnas.
            # 'sharex=True' comparte el eje X (Tiempo) entre todos los subplots.
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharex=True)
            
            # 'axes' es un array 2D (3,2). Lo aplanamos (flatten) para
            # poder iterar fácilmente con un solo índice (de 0 a 5).
            axes_flat = axes.flatten()
            
            features_found_count = 0

            # --- CAMBIO 3: Iterar por las 6 posiciones de la cuadrícula ---
            for i, feat in enumerate(features):
                ax = axes_flat[i] # El subplot actual (0 a 5)
                col_name = f'{cname}_{feat}'
                
                if col_name in features_df.columns:
                    # Si la característica SÍ existe en el DataFrame, graficarla
                    ax.plot(times, features_df[col_name], label=feat.capitalize())
                    ax.set_ylabel(feat.capitalize())
                    ax.grid(True)
                    features_found_count += 1
                else:
                    # Si la característica NO existe, ocultamos este subplot
                    ax.axis('off')
                    # Opcional: imprimir un mensaje
                    # print(f"No se encontró {col_name}, ocultando gráfico.")

            # Si no se encontró NINGUNA característica, no guardar/mostrar
            if features_found_count == 0:
                print(f"No hay columnas de características para {ch} en features_df")
                plt.close(fig) # Cerramos la figura vacía
                continue
            
            # --- CAMBIO 4: Configurar etiquetas para la cuadrícula ---
            
            # Ponemos la etiqueta 'Tiempo (s)' solo en los subplots de la fila inferior
            axes[n_rows - 1, 0].set_xlabel('Tiempo (s)')
            axes[n_rows - 1, 1].set_xlabel('Tiempo (s)')
            
            # Ponemos un título general a toda la figura
            fig.suptitle(f'Características - Canal {ch}', fontsize=16)
            
            # Ajustamos el layout para que el título no se solape
            fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            
            # --- Lógica de guardar y mostrar (sin cambios) ---
            if save:
                out_path = out_dir / (f'{filename}_{cname}_features.png')
                fig.savefig(str(out_path), dpi=150)
                last_out_path = out_path
            
            if show:
                plt.show()
                
            plt.close(fig) # Cerramos la figura para liberar memoria
        
        return last_out_path
    
    def plot_channels_and_spectra(self, data, channel_names, time_window=None,
                                nperseg=512, out_dir=Path(r"D:/SeñaelesEEGpy/results"), save=True, show=True,
                                features_df=None, window_size=256, overlap=0.5):
        """
        Grafica señales y PSD. Sombrea bandas, dibuja líneas verticales en límites de banda,
        calcula potencia por banda integrando PSD y (si features_df) compara potencias.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = data.get('metadata', {})
        fs = meta.get('sampling_rate', self.sampling_rate)
        filename = meta.get('filename', 'edf')
        signals = data.get('filtered_signals', data.get('signals'))
        chan_list = list(channel_names)
        available = data.get('channel_names', [])
        idxs = []
        missing = []
        for ch in chan_list:
            if ch in available:
                idxs.append(available.index(ch))
            else:
                missing.append(ch)
        if missing:
            print(f"Canales no encontrados y serán ignorados: {missing}")
        if not idxs:
            print("No se encontraron canales válidos para graficar.")
            return (None, None, None)
        sig_sel = signals[idxs, :].copy()
        time = np.arange(signals.shape[1]) / fs
        if time_window:
            start_idx = int(time_window[0] * fs)
            end_idx = int(time_window[1] * fs)
            start_idx = max(0, start_idx)
            end_idx = min(signals.shape[1], end_idx)
            sig_sel = sig_sel[:, start_idx:end_idx]
            time = time[start_idx:end_idx]
        seg_len = sig_sel.shape[1]
        if seg_len < 1:
            raise ValueError("Segmento vacío seleccionado para análisis.")
        nperseg = min(nperseg, seg_len)
        # PLOT Tiempo
        n_ch = len(idxs)
        ncols = 2
        nrows = math.ceil(n_ch / ncols)
        fig_t, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), sharex=True)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, ch_idx in enumerate(idxs):
            ax = axes[i]
            ax.plot(time, sig_sel[i, :], color=f'C{i%10}', lw=0.7)
            ax.set_ylabel(f'{available[ch_idx]}')
            ax.grid(True)
        for j in range(n_ch, len(axes)):
            axes[j].axis('off')
        axes[-1].set_xlabel('Tiempo (s)')
        fig_t.suptitle(f'Dominio del Tiempo - {filename}')
        fig_t.tight_layout(rect=[0, 0, 1, 0.97])
        time_out = None
        if save:
            time_out = out_dir / (Path(filename).stem + '_time.png')
            fig_t.savefig(str(time_out), dpi=150)
        if show:
            plt.show()
        plt.close(fig_t)
        # PSDs y sombreado/lineas de bandas
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        # beta Gamma Mejor datos para clasificación
        bands = {
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        band_colors = {
            'delta': 'C0', 'theta': 'C1', 'alpha': 'C2', 'beta': 'C3', 'gamma': 'C4'
        }
        fig_p, axp = plt.subplots(1, 1, figsize=(11, 7))
        psd_band_powers = {}
        for i, ch_idx in enumerate(idxs):
            ch_data = sig_sel[i, :]
            freqs, psd = signal.welch(ch_data, fs=fs, nperseg=nperseg)
            axp.semilogy(freqs, psd, label=available[ch_idx], lw=0.9, alpha=0.9)
            ch_powers = {}
            for bname, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                power = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
                ch_powers[bname] = power
            psd_band_powers[available[ch_idx]] = ch_powers
        # sombreado y lineas verticales
        for bname, (lo, hi) in bands.items():
            axp.axvspan(lo, hi, color=band_colors.get(bname, 'gray'), alpha=0.06, linewidth=0)
        boundaries = sorted({b for pair in bands.values() for b in pair})
        # obtener y limits antes de texto
        axp.set_xlim(0, min(60, fs / 2))
        for b in boundaries:
            axp.axvline(x=b, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
        # anotar una vez (arriba)
        ylim = axp.get_ylim()
        y_annot = ylim[1] / 1.5
        for b in boundaries:
            axp.text(b, y_annot, f'{b}Hz', rotation=90, va='bottom', ha='center', fontsize=8, color='k', alpha=0.8)
        axp.set_xlabel('Frequencia (Hz)')
        axp.set_ylabel('PSD (V**2/Hz)')
        axp.grid(True, which='both', ls=':')
        axp.legend(loc='upper right', fontsize='small', ncol=2)
        fig_p.suptitle(f'PSD (Welch) - {filename}')
        fig_p.tight_layout(rect=[0, 0, 1, 0.96])
        psd_out = None
        if save:
            psd_out = out_dir / (Path(filename).stem + '_welch_with_bands.png')
            fig_p.savefig(str(psd_out), dpi=150)
        if show:
            plt.show()
        plt.close(fig_p)
        # Comparación con features_df si se proporciona
        compare_out = None
        if features_df is not None and not features_df.empty:
            
            rows = []
            for ch_idx in idxs:
                ch_name = available[ch_idx]
                cname = self.safe(ch_name)
                for b in bands.keys():
                    col = f'{cname}_{b}'
                    feat_mean = float(features_df[col].mean()) if col in features_df.columns else np.nan
                    psd_val = psd_band_powers.get(ch_name, {}).get(b, 0.0)
                    rows.append({'channel': ch_name, 'band': b, 'psd_power': psd_val, 'features_mean': feat_mean})
            compare_df = pd.DataFrame(rows)
            compare_out = out_dir / (Path(filename).stem + '_band_power_comparison.csv')
            compare_df.to_csv(compare_out, index=False)
            print(f'Comparación PSD vs features guardada en: {compare_out}')
        # Resumen PSD por canal
        summary_rows = []
        for ch, powers in psd_band_powers.items():
            row = {'channel': ch}
            row.update(powers)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_out = out_dir / (Path(filename).stem + '_psd_band_summary.csv')
        summary_df.to_csv(summary_out, index=False)
        print(f'Resumen de potencias por banda (desde PSD) guardado en: {summary_out}')
        return (time_out, psd_out, summary_out, compare_out)

    def plot_band_powers(self, features_df, data, channel_names,
                        window_size=256, overlap=0.5,
                        out_dir=Path(r"D:/SeñaelesEEGpy/results"), save=True, show=True):
        """
        Grafica las potencias de banda para los canales indicados.
        Busca columnas del tipo <safe_channel_name>_<band>.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = data.get('metadata', {})
        filename = meta.get('filename', 'edf')
        fs = meta.get('sampling_rate', self.sampling_rate)
        step = int(window_size * (1 - overlap))
        if step <= 0:
            step = max(1, window_size // 2)
        times = (np.arange(len(features_df)) * step + window_size / 2) / fs
        bands = ['delta_Abs', 'theta_Abs', 'alpha_Abs', 'beta_Abs', 'gamma_Abs']
        
        for ch in channel_names:
            if ch not in data.get('channel_names', []):
                print(f"Canal {ch} no encontrado, se omite.")
                continue
            cname = self.safe(ch)
            fig, ax = plt.subplots(figsize=(10, 4))
            found = False
            for band in bands:
                col_name = f'{cname}_{band}'
                if col_name in features_df.columns:
                    ax.plot(times, features_df[col_name], label=band.capitalize())
                    found = True
            if not found:
                print(f"No hay columnas de banda para {ch} en features_df")
                plt.close(fig)
                continue
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Bandas Poder')
            ax.set_title(f'Banda Poder - Canal {ch} - {filename}')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            if save:
                out_path = out_dir / (Path(filename).stem + f'_{cname}_band_powers.png')
                fig.savefig(str(out_path), dpi=150)
            if show:
                plt.show()
            plt.close(fig)
    
    def plot_power_relative_bandas(self, features_df, canales, filename,
                                window_size=256, overlap=0.5,
                                out_dir=Path(r"D:/SeñaelesEEGpy/results"), 
                                save=True, show=True):
        """
        Grafica las potencias relativas de banda (como áreas apiladas) para canales específicos.
        
        Args:
            features_df (pd.DataFrame): DataFrame con características extraídas
            canales (list): Lista de nombres de canales a graficar
            filename (str): Nombre base para los archivos guardados
            window_size (int): Tamaño de ventana en muestras
            overlap (float): Solapamiento entre ventanas (0-1)
            out_dir (Path): Directorio de salida
            save (bool): Guardar la figura
            show (bool): Mostrar la figura
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fs = self.sampling_rate
        
        # Cálculo del eje de tiempo
        step = int(window_size * (1 - overlap))
        if step <= 0:
            step = max(1, window_size // 2)
        times = (np.arange(len(features_df)) * step + window_size / 2) / fs
        
        bands = ['delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel']
        band_labels = [b.replace('_rel', '').capitalize() for b in bands]
        
        # Lista para guardar las rutas de los archivos generados
        saved_paths = [] 

        for channel_name in canales:
            cname = self.safe(channel_name)
            
            # Recolectar los datos para este canal
            band_data = []
            valid_bands = []
            for i, band in enumerate(bands):
                col_name = f'{cname}_{band}'
                if col_name in features_df.columns:
                    band_data.append(features_df[col_name])
                    valid_bands.append(band_labels[i])
                
            # Si no se encontraron datos para este canal, saltar al siguiente
            if not band_data:
                print(f"No hay columnas de potencia relativa para {channel_name} en features_df. Saltando canal.")
                continue # <-- CORRECCIÓN 2: Usar 'continue' en lugar de 'return'

            fig, ax = plt.subplots(figsize=(12, 5))
            
            # --- MEJORA: Usar Stackplot ---
            # Stackplot es ideal para potencia relativa
            try:
                ax.stackplot(times, band_data, labels=valid_bands, alpha=0.8)
            except Exception as e:
                print(f"Error al crear stackplot para {channel_name}: {e}. Volviendo a plot.")
                # Fallback a plot de líneas si stackplot falla (ej. datos con NaN)
                for i, data in enumerate(band_data):
                    ax.plot(times, data, label=valid_bands[i])

            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Potencia Relativa de Banda')
            ax.set_title(f'Potencia Relativa de Banda (Apilada) - Canal {channel_name}')
            ax.grid(True)
            ax.legend(loc='upper left')
            ax.set_ylim(0, max(1.0, np.max(np.sum(band_data, axis=0)))) # Asegurar que el eje Y empiece en 0
            ax.set_xlim(times[0], times[-1])
            
            fig.tight_layout()
            
            if save:
                # --- CORRECCIÓN 1: Usar 'cname' para el nombre del archivo ---
                out_path = out_dir / (f'{filename}_{cname}_relative_band_powers.png')
                fig.savefig(str(out_path), dpi=150)
                saved_paths.append(out_path) # Añadir la ruta a la lista
                
            if show:
                plt.show()
                
            plt.close(fig) # Cerrar la figura para liberar memoria
            
        # --- CORRECCIÓN 3: Retornar la lista de rutas ---
        return saved_paths
    
    def plot_espectrograma_banda(self, signal_data, fs, band,bandName, channel_name,filename,
                                nperseg=256, noverlap=128,
                                out_dir=Path(r"D:/SeñaelesEEGpy/results"), save=True, show=True):
        """
        Grafica el espectrograma para una banda específica de un canal.
        Args:
            signal_data (numpy.array): Señal del canal (1D array)
            fs (float): Frecuencia de muestreo
            band (tuple): Tupla (low, high) de la banda en Hz
            bandName (str): Nombre de la banda (ejemplo 'theta')
            channel_name (str): Nombre del canal
            filename (str): Nombre base del archivo para guardar
            nperseg (int): Longitud de segmento para espectrograma
            noverlap (int): Solapamiento entre segmentos
        """
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        f, t, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        band_mask = (f >= band[0]) & (f <= band[1])
        Sxx_band = Sxx[band_mask, :]
        f_band = f[band_mask]
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f_band, 10 * np.log10(Sxx_band), shading='gouraud')
        plt.colorbar(label='PSD (dB/Hz)')
        plt.ylabel('Frecuencia (Hz)')
        plt.xlabel('Tiempo (s)')
        plt.title(f'Espectrograma Banda {band[0]} -- {band[1]} Hz - Canal {channel_name} en {bandName}')
        plt.ylim(band)
        plt.tight_layout()
        out_path = None
        if save:
            safe_ch_name = re.sub(r'[^0-9A-Za-z_]', '_', str(channel_name)).strip('_')
            out_path = out_dir / f'{filename}_spectrogram_{safe_ch_name}_{bandName}.png'
            plt.savefig(str(out_path), dpi=150)
        if show:
            plt.show()
        plt.close()
        return out_path
    
    def aplicar_car(self, signal_data):
        """
        Aplica Common Average Reference (CAR) a las señales EEG.
        Resta el promedio de todos los canales a cada canal individual.
        
        Args:
            signal_data (numpy.array): Array de señales EEG
                - shape (n_canales, n_muestras) o
                - shape (n_canales, n_muestras, n_ensayos)
        
        Returns:
            numpy.array: Señales con CAR aplicado (misma forma que entrada)
        """
        signal_data = np.asarray(signal_data)
        
        if signal_data.ndim == 2:
            # (n_canales, n_muestras)
            # promedio sobre canales (axis=0)
            promedio_comun = np.mean(signal_data, axis=0, keepdims=True)
            data_car = signal_data - promedio_comun
        elif signal_data.ndim == 3:
            # (n_canales, n_muestras, n_ensayos)
            # promedio sobre canales (axis=0), manteniendo dimensiones de tiempo y ensayos
            promedio_comun = np.mean(signal_data, axis=0, keepdims=True)
            data_car = signal_data - promedio_comun
        else:
            raise ValueError("signal_data debe tener 2 o 3 dimensiones: (canales, muestras) o (canales, muestras, ensayos)")
        
        return data_car
    
    def obtener_muestras_por_ensayo(self, signals, window_size=32, overlap=8, 
                                    channel_names=None, available_channel_names=None):
        """
        Divide las señales en ventanas (ensayos) sin solapamiento progresivo.
        Equivalente a MuestrasDev.m de MATLAB.
        
        Args:
            signals (np.ndarray): Array de señales EEG
                - shape (n_canales, n_muestras)
            window_size (int): Tamaño de la ventana en muestras (default 32)
            overlap (int): Solapamiento en muestras (default 8)
            channel_names (list): Nombres de canales a extraer. Si None, extrae todos.
            available_channel_names (list): Lista completa de nombres de canales
        
        Returns:
            np.ndarray: shape (n_canales_seleccionados, window_size, num_windows)
                - Cada "ventana" es un ensayo independiente
            list: Nombres de canales seleccionados
        """
        signals = np.asarray(signals, dtype=float)
        
        if signals.ndim != 2:
            raise ValueError("signals debe tener shape (n_canales, n_muestras)")
        
        n_canales_total, n_muestras = signals.shape
        
        # Decidir qué canales procesar
        if channel_names is None:
            sel_idxs = list(range(n_canales_total))
            sel_names = [f'ch{i}' for i in range(n_canales_total)]
        else:
            sel_idxs = []
            sel_names = []
            if available_channel_names is not None:
                for ch in channel_names:
                    if ch in available_channel_names:
                        idx = available_channel_names.index(ch)
                        sel_idxs.append(idx)
                        sel_names.append(ch)
                    else:
                        try:
                            idx = int(ch)
                            if 0 <= idx < n_canales_total:
                                sel_idxs.append(idx)
                                sel_names.append(f'ch{idx}')
                        except Exception:
                            print(f'Warning: canal "{ch}" no encontrado y será ignorado.')
            else:
                for ch in channel_names:
                    try:
                        idx = int(ch)
                        if 0 <= idx < n_canales_total:
                            sel_idxs.append(idx)
                            sel_names.append(f'ch{idx}')
                    except Exception:
                        print(f'Warning: canal "{ch}" inválido y será ignorado.')
        
        if not sel_idxs:
            raise ValueError("No se encontraron índices de canal válidos.")
        
        stride = window_size - overlap  # paso entre ventanas
        
        # Calcular número de ventanas
        num_windows = (n_muestras - window_size) // stride + 1
        
        if num_windows <= 0:
            raise ValueError(f"Señal demasiado corta ({n_muestras} muestras) para window_size={window_size}, overlap={overlap}")
        
        # Crear matriz de salida: (n_canales_seleccionados, window_size, num_windows)
        n_sel = len(sel_idxs)
        windows = np.zeros((n_sel, window_size, num_windows), dtype=float)
        
        # Extraer ventanas solo para canales seleccionados
        for sel_idx, canal_idx in enumerate(sel_idxs):
            x = signals[canal_idx, :]
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                windows[sel_idx, :, i] = x[start_idx:end_idx]
        
        return windows, sel_names
    
    def extract_features_por_ensayo(self, windows, channel_names=None, fs=None):
        """
        Extrae características por ENSAYO (ventana).
        Cada ventana (ensayo) produce un conjunto de características.
        
        Args:
            windows (np.ndarray): shape (n_canales, window_size, num_windows)
                - Salida de obtener_muestras_por_ensayo()
            channel_names (list): Nombres de los canales (coincide con order de windows)
            fs (float): Frecuencia de muestreo (para bandas de frecuencia)
        
        Returns:
            pd.DataFrame: Filas = ensayos; Columnas = canal_caracteristica
                Ejemplo: F7_mean, F7_delta_Abs, F7_beta_rel, ...
        """
        if fs is None:
            fs = self.sampling_rate
        
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 3:
            raise ValueError("windows debe tener shape (n_canales, window_size, num_windows)")
        
        n_canales, window_size, num_windows = windows.shape
        
        features_list = []
        
        # Iterar sobre cada ensayo (ventana)
        for ensayo_idx in range(num_windows):
            ensayo_features = {}
            
            # Iterar sobre cada canal
            for canal_idx in range(n_canales):
                # Extraer señal de este canal en este ensayo
                signal_ensayo = windows[canal_idx, :, ensayo_idx]
                
                # Nombre seguro del canal
                if channel_names and canal_idx < len(channel_names):
                    cname = self.safe(channel_names[canal_idx])
                else:
                    cname = f'ch{canal_idx}'
                
                # --- Características de TIEMPO ---
                ensayo_features[f'{cname}_mean'] = float(np.mean(signal_ensayo))
                ensayo_features[f'{cname}_std'] = float(np.std(signal_ensayo))
                ensayo_features[f'{cname}_var'] = float(np.var(signal_ensayo))
                ensayo_features[f'{cname}_rms'] = float(np.sqrt(np.mean(signal_ensayo**2)))
                ensayo_features[f'{cname}_skewness'] = float(stats.skew(signal_ensayo))
                ensayo_features[f'{cname}_kurtosis'] = float(stats.kurtosis(signal_ensayo))
                
                # --- Características de FRECUENCIA (bandas) ---
                fft_vals = np.abs(np.fft.rfft(signal_ensayo))
                freqs = np.fft.rfftfreq(len(signal_ensayo), 1.0 / fs)
                power_vals = fft_vals ** 2
                total_power = float(np.sum(power_vals))
                
                # Máscaras de bandas
                delta_mask = (freqs >= 0.5) & (freqs <= 4)
                theta_mask = (freqs > 4) & (freqs <= 8)
                alpha_mask = (freqs > 8) & (freqs <= 14)
                beta_mask = (freqs > 14) & (freqs <= 30)
                gamma_mask = (freqs > 30)
                
                # Potencia absoluta por banda
                ensayo_features[f'{cname}_delta_Abs'] = float(np.sum(power_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_Abs'] = float(np.sum(power_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_Abs'] = float(np.sum(power_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_Abs'] = float(np.sum(power_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_Abs'] = float(np.sum(power_vals[gamma_mask])) if gamma_mask.any() else np.nan
                # valores estadisticos en la frecuancias de las bandas
                #medias
                ensayo_features[f'{cname}_delta_mean'] = float(np.mean(fft_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_mean'] = float(np.mean(fft_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_mean'] = float(np.mean(fft_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_mean'] = float(np.mean(fft_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_mean'] = float(np.mean(fft_vals[gamma_mask])) if gamma_mask.any() else np.nan
                #desviasion estandar
                ensayo_features[f'{cname}_delta_std']=float(np.std(fft_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_std']=float(np.std(fft_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_std']=float(np.std(fft_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_std']=float(np.std(fft_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_std']=float(np.std(fft_vals[gamma_mask])) if gamma_mask.any() else np.nan
                # Varianza
                ensayo_features[f'{cname}_delta_var']=float(np.var(fft_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_var']=float(np.var(fft_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_var']=float(np.var(fft_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_var']=float(np.var(fft_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_var']=float(np.var(fft_vals[gamma_mask])) if gamma_mask.any() else np.nan
                # RMS
                ensayo_features[f'{cname}_delta_rms']=float(np.sqrt(np.mean(fft_vals[delta_mask]**2))) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_rms']=float(np.sqrt(np.mean(fft_vals[theta_mask]**2))) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_rms']=float(np.sqrt(np.mean(fft_vals[alpha_mask]**2))) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_rms']=float(np.sqrt(np.mean(fft_vals[beta_mask]**2))) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_rms']=float(np.sqrt(np.mean(fft_vals[gamma_mask]**2))) if gamma_mask.any() else np.nan
                # Skewness
                ensayo_features[f'{cname}_delta_skewness']=float(stats.skew(fft_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_skewness']=float(stats.skew(fft_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_skewness']=float(stats.skew(fft_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_skewness']=float(stats.skew(fft_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_skewness']=float(stats.skew(fft_vals[gamma_mask])) if gamma_mask.any() else np.nan
                # Kurtosis
                ensayo_features[f'{cname}_delta_kurtosis']=float(stats.kurtosis(fft_vals[delta_mask])) if delta_mask.any() else np.nan
                ensayo_features[f'{cname}_theta_kurtosis']=float(stats.kurtosis(fft_vals[theta_mask])) if theta_mask.any() else np.nan
                ensayo_features[f'{cname}_alpha_kurtosis']=float(stats.kurtosis(fft_vals[alpha_mask])) if alpha_mask.any() else np.nan
                ensayo_features[f'{cname}_beta_kurtosis']=float(stats.kurtosis(fft_vals[beta_mask])) if beta_mask.any() else np.nan
                ensayo_features[f'{cname}_gamma_kurtosis']=float(stats.kurtosis(fft_vals[gamma_mask])) if gamma_mask.any() else np.nan
                # Potencia relativa por banda
                if total_power > 0:
                    ensayo_features[f'{cname}_delta_rel'] = ensayo_features[f'{cname}_delta_Abs'] / total_power
                    ensayo_features[f'{cname}_theta_rel'] = ensayo_features[f'{cname}_theta_Abs'] / total_power
                    ensayo_features[f'{cname}_alpha_rel'] = ensayo_features[f'{cname}_alpha_Abs'] / total_power
                    ensayo_features[f'{cname}_beta_rel'] = ensayo_features[f'{cname}_beta_Abs'] / total_power
                    ensayo_features[f'{cname}_gamma_rel'] = ensayo_features[f'{cname}_gamma_Abs'] / total_power
                else:
                    ensayo_features[f'{cname}_delta_rel'] = np.nan
                    ensayo_features[f'{cname}_theta_rel'] = np.nan
                    ensayo_features[f'{cname}_alpha_rel'] = np.nan
                    ensayo_features[f'{cname}_beta_rel'] = np.nan
                    ensayo_features[f'{cname}_gamma_rel'] = np.nan
                # --- Características Wavelet Discreta (DWT) usando db4 nivel 5 ---
                if pywt is None:
                    for lvl in range(1, 6):
                        ensayo_features[f'{cname}_wD{lvl}_energy'] = np.nan
                        ensayo_features[f'{cname}_wD{lvl}_rel'] = np.nan
                    ensayo_features[f'{cname}_wA5_energy'] = np.nan
                    ensayo_features[f'{cname}_wA5_rel'] = np.nan
                else:
                    try:
                        coeffs = pywt.wavedec(signal_ensayo, 'db4', level=5)
                        energies = [float(np.sum(np.asarray(c)**2)) for c in coeffs]
                        total_w_energy = sum(energies) if sum(energies) > 0 else 1.0
                        ensayo_features[f'{cname}_wA5_energy'] = energies[0]
                        ensayo_features[f'{cname}_wA5_rel'] = energies[0] / total_w_energy    
                    except Exception:
                        ensayo_features[f'{cname}_wA5_energy'] = np.nan
                        ensayo_features[f'{cname}_wA5_rel'] = np.nan
                # Calculo Welch de la señal para potencia media
                try:
                    #solo en Beta y Gamma
                    f_welch, Pxx = signal.welch(signal_ensayo, fs=fs, nperseg=min(256, len(signal_ensayo)))
                    band_mask_welch = (f_welch >= 8) & (f_welch <= 40)
                    Pxx = Pxx[band_mask_welch]
                    mean_power = float(np.mean(Pxx))
                    ensayo_features[f'{cname}_welch_mean_power'] = mean_power
                except Exception:
                    ensayo_features[f'{cname}_welch_mean_power'] = np.nan

            features_list.append(ensayo_features)
        
        return pd.DataFrame(features_list)
    
def main():
    # Configuración de rutas
    data_path = r"D:/EEG_Python/Imagined_speech_EEG_edf/Image"  # Cambia si hace falta
    # Ruta de resultados
    results_dir = Path(r"D:/EEG_Python/results/Image")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Inicializar lector
    eeg_reader = EEGEDFReader(data_path)
    # Listar archivos (ejemplo buscando sufijo '0')
    #'Digit':['0','1','2','3','4','5','6','7','8','9'],
    #'Char':['A','C','F','H','J','M','P','S','T','Y']
    #'Image':['Apple','Car','Dog','Gold','Mobile','Rose','Scooter','Tiger','Wallet','Watch']         
    sufijo=['Apple','Car','Dog','Gold','Mobile','Rose','Scooter','Tiger','Wallet','Watch']
    
    for rep in sufijo:  # Procesar varias veces para pruebas
        digit_files = eeg_reader.list_edf_files(data_path, num=rep,user='name0')
        print("Archivos EDF encontrados:")
        canales = ['F7','F3','O1','P8','P7']
        if digit_files: 
            for file in digit_files:
                print(f"  - {file}")
                #first_digit_file = digit_files[0]
                print(f"\nLeyendo archivo: {file}")
                eeg_data = eeg_reader.read_edf_file(file)
                if eeg_data is not None:
                    metadata = eeg_data['metadata']
                    print(f"\nInformación del archivo:")
                    print(f"  - Canales: {metadata['n_channels']}")
                    print(f"  - Duración: {metadata['duration']:.2f} segundos")
                    print(f"  - Frecuencia de muestreo: {metadata['sampling_rate']} Hz")
                    print(f"  - Etiquetas de canales: {metadata['signal_labels']}")
                    # Preprocesar señales
                    filtered_signals = eeg_reader.preprocess_eeg(eeg_data['signals'], fs=metadata.get('sampling_rate'))
                    eeg_data['filtered_signals'] = filtered_signals
                    # Aplicar CAR (Common Average Reference)
                    print("\nAplicando CAR (Common Average Reference)...")
                    car_signals = eeg_reader.aplicar_car(filtered_signals)
                    eeg_data['car_signals'] = car_signals
                    print("CAR aplicado exitosamente")
                    """
                    print(f"Extrayendo características...")                
                    features_df = eeg_reader.extract_features(car_signals,
                                                            channel_names=canales,
                                                            available_channel_names=eeg_data['channel_names'],
                                                            window_size=256, overlap=0.5)
                    print(f"Características extraídas: {features_df.shape}")
                    print(features_df.head())
                    eeg_data['filtered_signals'] = car_signals  # actualizar para plots
                    print("Generando gráficas (tiempo + PSD + bandas)...")
                    eeg_reader.plot_channels_and_spectra(eeg_data, channel_names=canales,
                                                        time_window=(0, 5), save=True, show=False,
                                                        features_df=features_df, window_size=256, overlap=0.5)
                    print("Generando gráficas de características por canal")
                    eeg_reader.plot_feactureres(features_df, canales,filename=Path(str(file)).stem,
                                                    window_size=256, overlap=0.5,
                                                    out_dir=results_dir, save=True, show=False)
                    print("Generando gráficas de espectrogramas por banda")
                    bands = {
                        'Rango interes': (13, 40)
                    }
                    for ch in canales:
                        if ch not in eeg_data['channel_names']:
                            print(f"Canal {ch} no encontrado, se omite espectrograma.")
                            continue
                        ch_idx = eeg_data['channel_names'].index(ch)
                        ch_signal = filtered_signals[ch_idx, :]
                        for bname, band in bands.items():
                            print(f"  - Canal {ch}, en {bname}")
                            eeg_reader.plot_espectrograma_banda(ch_signal, fs=metadata.get('sampling_rate'),
                                                                band=band,bandName=bname, channel_name=ch,filename=Path(str(file)).stem,
                                                                nperseg=256, noverlap=128, 
                                                                out_dir=results_dir, save=True, show=False)
                    print("Generando gráficas de potencias de banda")
                    eeg_reader.plot_band_powers(features_df, eeg_data, canales,window_size=256, overlap=0.5,
                                                out_dir=results_dir, save=True, show=False)
                    print("podencias relativas de banda para el canal F3")
                    eeg_reader.plot_power_relative_bandas(features_df, canales,filename=Path(str(file)).stem,
                                                        window_size=256, overlap=0.5,
                                                        out_dir=results_dir, save=True, show=False)
                    output_filename = Path(str(file)).stem + '_features.csv'
                    output_path = results_dir / output_filename
                    features_df.to_csv(output_path, index=False)
                    print(f"Características guardadas en: {output_path}") 
                    """
                    print("Obteniendo muestras por ensayo (ventanas)...")
                    windows, sel_canal_names = eeg_reader.obtener_muestras_por_ensayo(
                        car_signals, 
                        window_size=32,    # Tamaño de ventana
                        overlap=8,         # Solapamiento
                        channel_names=canales,
                        available_channel_names=eeg_data['channel_names']
                    )
                    print(f"Windows shape: {windows.shape}")  # (n_canales_seleccionados, 32, num_ensayos)
                    print(f"Canales seleccionados: {sel_canal_names}")
                    
                    print("Extrayendo características por ensayo...")
                    features_ensayos_df = eeg_reader.extract_features_por_ensayo(
                        windows,
                        channel_names=sel_canal_names,
                        fs=metadata.get('sampling_rate')
                    )
                    print(f"Features por ensayo: {features_ensayos_df.shape}")
                    print(features_ensayos_df.head())
                    
                    # Guardar características por ensayo
                    ensayos_output = results_dir / (Path(str(file)).stem + '_features_ensayos.csv')
                    features_ensayos_df.to_csv(ensayos_output, index=False)
                    print(f"Características por ensayo guardadas en: {ensayos_output}")
                else:
                    print("Error: No se pudo leer el archivo EDF")
        else:
            print("No se encontraron archivos EDF con el sufijo especificado.")

def main_2():
    data_path = "captures"  # Cambia si hace falta
    eeg_reader = EEGEDFReader(data_path)
    info =  eeg_reader.read_edf_file("User9/Letters/User9_A_4.edf")
    print(info)

if __name__ == "__main__":
    main_2()
