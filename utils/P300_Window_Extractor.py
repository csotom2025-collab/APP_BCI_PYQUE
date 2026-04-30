"""
Extractor de ventanas P300 de grabaciones EEG de 2 segundos
Extrae y visualiza:
- Respuesta Evocada P300 (0.5 - 1.2s): Componente de respuesta evocada
- Post-estímulo (1.2 - 2.0s): Período de recuperación
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from EEdfReader import EEGEDFReader as eeg_reader
import pandas as pd


class P300WindowExtractor:
    def __init__(self, data_path='.', sampling_rate=250):
        """
        Inicializa el extractor de ventanas P300
        Args:
            data_path: Ruta donde están los archivos EDF
            sampling_rate: Frecuencia de muestreo (Hz)
        """
        self.reader = eeg_reader(data_path=data_path)
        self.sampling_rate = sampling_rate
        
    def extract_windows_from_recording(self, eeg_data):
        """
        Extrae ventanas de tiempo específicas de una grabación de 2 segundos
        
        Args:
            eeg_data: Diccionario con datos y metadatos del EEG
            
        Returns:
            dict: Diccionario con ventanas extraídas y sus índices
        """
        signals = eeg_data['signals']
        fs = eeg_data['metadata']['sampling_rate']
        
        # Calcular índices para cada ventana (basado en tiempo)
        # Ventana P300: 0.5s - 1.2s (respuesta evocada)
        p300_start_idx = int(0.5 * fs)
        p300_end_idx = int(1.2 * fs)
        
        # Ventana Post-estímulo: 1.2s - 2.0s (recuperación)
        post_start_idx = int(1.2 * fs)
        post_end_idx = int(2.0 * fs)
        
        windows = {
            'p300': {
                'signals': signals[:, p300_start_idx:p300_end_idx],
                'time_range': (0.5, 1.2),
                'start_idx': p300_start_idx,
                'end_idx': p300_end_idx,
                'duration': 0.7  # segundos
            },
            'post_stimulus': {
                'signals': signals[:, post_start_idx:post_end_idx],
                'time_range': (1.2, 2.0),
                'start_idx': post_start_idx,
                'end_idx': post_end_idx,
                'duration': 0.8  # segundos
            }
        }
        
        return windows
    
    def calculate_p300_features(self, p300_signals, fs, channel_names=None):
        """
        Calcula características del P300 por canal
        
        Args:
            p300_signals: Array de señales en la ventana P300
            fs: Frecuencia de muestreo
            channel_names: Nombres de los canales
            
        Returns:
            DataFrame con características del P300
        """
        n_channels = p300_signals.shape[0]
        
        features = {
            'Channel': [],
            'P300_Amplitude_Max': [],
            'P300_Amplitude_Min': [],
            'P300_Latency_Max': [],  # Latencia del pico máximo
            'P300_Mean_Amplitude': [],
            'P300_RMS': []
        }
        
        time_vector = np.arange(p300_signals.shape[1]) / fs
        
        for ch_idx in range(n_channels):
            signal = p300_signals[ch_idx, :]
            ch_name = channel_names[ch_idx] if channel_names else f'Ch{ch_idx}'
            
            # Características
            max_amp = np.max(signal)
            min_amp = np.min(signal)
            max_latency = np.argmax(np.abs(signal)) / fs  # en segundos
            mean_amp = np.mean(signal)
            rms = np.sqrt(np.mean(signal**2))
            
            features['Channel'].append(ch_name)
            features['P300_Amplitude_Max'].append(max_amp)
            features['P300_Amplitude_Min'].append(min_amp)
            features['P300_Latency_Max'].append(max_latency)
            features['P300_Mean_Amplitude'].append(mean_amp)
            features['P300_RMS'].append(rms)
        
        return pd.DataFrame(features)
    
    def plot_p300_windows(self, eeg_data, windows, channel_names=None, 
                         save=False, out_dir=None, filename='p300_analysis'):
        """
        Visualiza las ventanas de P300 y post-estímulo
        
        Args:
            eeg_data: Datos EEG con filtrado y CAR aplicado
            windows: Diccionario con ventanas extraídas
            channel_names: Nombres de canales a visualizar
            save: Guardar figura
            out_dir: Directorio de salida
            filename: Nombre del archivo de salida
        """
        if channel_names is None:
            channel_names = eeg_data['channel_names'][:8]  # Primeros 8 canales
        
        fs = eeg_data['metadata']['sampling_rate']
        n_channels = len(channel_names)
        
        fig, axes = plt.subplots(n_channels, 2, figsize=(15, n_channels*2))
        fig.suptitle(f"P300 Analysis - {filename}", fontsize=14, fontweight='bold')
        
        if n_channels == 1:
            axes = axes.reshape(1, -1)
        
        p300_signals = windows['p300']['signals']
        post_signals = windows['post_stimulus']['signals']
        
        # Vector de tiempo para P300 (0.5 - 1.2s)
        p300_time = np.arange(p300_signals.shape[1]) / fs
        
        # Vector de tiempo para post-estímulo (1.2 - 2.0s)
        post_time = np.arange(post_signals.shape[1]) / fs
        
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name not in eeg_data['channel_names']:
                continue
            
            actual_idx = eeg_data['channel_names'].index(ch_name)
            
            # Columna 1: Ventana P300
            ax1 = axes[ch_idx, 0]
            p300_signal = p300_signals[actual_idx, :]
            ax1.plot(p300_time + 0.5, p300_signal, 'b-', linewidth=1.5, label='P300')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Estimulo')
            ax1.fill_between(p300_time + 0.5, p300_signal, alpha=0.2)
            ax1.set_title(f'{ch_name} - Respuesta Evocada P300 (0.5-1.2s)', fontweight='bold')
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud (µV)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Anotación del pico máximo
            max_idx = np.argmax(np.abs(p300_signal))
            max_val = p300_signal[max_idx]
            ax1.plot(p300_time[max_idx] + 0.5, max_val, 'ro', markersize=8)
            ax1.annotate(f'Pico: {max_val:.1f}µV\n@{p300_time[max_idx]:.3f}s',
                        xy=(p300_time[max_idx] + 0.5, max_val),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=8)
            
            # Columna 2: Ventana Post-estímulo
            ax2 = axes[ch_idx, 1]
            post_signal = post_signals[actual_idx, :]
            ax2.plot(post_time + 1.2, post_signal, 'g-', linewidth=1.5, label='Post-estímulo')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.axvline(x=1.2, color='r', linestyle='--', alpha=0.5, label='Inicio recuperación')
            ax2.fill_between(post_time + 1.2, post_signal, alpha=0.2, color='green')
            ax2.set_title(f'{ch_name} - Post-estímulo/Recuperación (1.2-2.0s)', fontweight='bold')
            ax2.set_xlabel('Tiempo (s)')
            ax2.set_ylabel('Amplitud (µV)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        if save and out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f'{filename}_p300_windows.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        
        plt.show()
    
    def plot_p300_comparison(self, eeg_data, windows, channel_names=None, 
                            save=False, out_dir=None, filename='p300_comparison'):
        """
        Grafica comparación superpuesta de P300 vs Post-estímulo
        """
        if channel_names is None:
            channel_names = eeg_data['channel_names'][:8]
        
        fs = eeg_data['metadata']['sampling_rate']
        n_channels = len(channel_names)
        
        fig, axes = plt.subplots(1, n_channels, figsize=(18, 4))
        if n_channels == 1:
            axes = [axes]
        
        fig.suptitle(f"P300 vs Post-Estímulo Comparison - {filename}", 
                    fontsize=14, fontweight='bold')
        
        p300_signals = windows['p300']['signals']
        post_signals = windows['post_stimulus']['signals']
        
        p300_time = np.arange(p300_signals.shape[1]) / fs
        post_time = np.arange(post_signals.shape[1]) / fs
        
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name not in eeg_data['channel_names']:
                continue
            
            actual_idx = eeg_data['channel_names'].index(ch_name)
            ax = axes[ch_idx]
            
            # Normalizar para comparación
            p300_signal = p300_signals[actual_idx, :]
            post_signal = post_signals[actual_idx, :]
            
            # Graficar ambas ventanas
            ax.plot(p300_time, p300_signal, 'b-', linewidth=2, label='P300 (0.5-1.2s)', alpha=0.7)
            ax.plot(post_time, post_signal, 'g-', linewidth=2, label='Post-estímulo (1.2-2.0s)', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_title(f'{ch_name}', fontweight='bold')
            ax.set_xlabel('Tiempo Relativo (s)')
            ax.set_ylabel('Amplitud (µV)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save and out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f'{filename}_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica de comparación guardada en: {save_path}")
        
        plt.show()
    
    def analyze_single_trial(self, file_path, channel_names=None, 
                           save_results=False, out_dir=None):
        """
        Analiza un único ensayo/grabación de 2 segundos
        
        Args:
            file_path: Ruta del archivo EDF
            channel_names: Lista de canales a analizar
            save_results: Guardar resultados
            out_dir: Directorio de salida
            
        Returns:
            dict: Resultados del análisis
        """
        # Leer archivo
        print(f"\n📁 Leyendo archivo: {file_path}")
        eeg_data = self.reader.read_edf_file(filename=file_path)
        
        if eeg_data is None:
            print("Error al leer el archivo")
            return None
        
        metadata = eeg_data['metadata']
        print(f"   - Canales: {metadata['n_channels']}")
        print(f"   - Duración: {metadata['duration']:.2f} segundos")
        print(f"   - Frecuencia de muestreo: {metadata['sampling_rate']} Hz")
        
        # Canales por defecto
        if channel_names is None:
            channel_names = eeg_data['channel_names']
        
        # Preprocesar
        print("\n🔧 Preprocesando señales...")
        filtered_signals = self.reader.preprocess_eeg(
            signals=eeg_data['signals'], 
            fs=metadata.get('sampling_rate')
        )
        eeg_data['filtered_signals'] = filtered_signals
        
        # Aplicar CAR
        print("📌 Aplicando CAR (Common Average Reference)...")
        car_signals = self.reader.aplicar_car(filtered_signals)
        eeg_data['car_signals'] = car_signals
        
        # Extraer ventanas
        print("\n📊 Extrayendo ventanas P300 y post-estímulo...")
        windows = self.extract_windows_from_recording(eeg_data)
        
        print(f"   - P300 (0.5-1.2s): {windows['p300']['signals'].shape}")
        print(f"   - Post-estímulo (1.2-2.0s): {windows['post_stimulus']['signals'].shape}")
        
        # Calcular características del P300
        print("\n📈 Calculando características del P300...")
        p300_features = self.calculate_p300_features(
            windows['p300']['signals'],
            fs=metadata['sampling_rate'],
            channel_names=channel_names
        )
        print(p300_features)
        
        # Visualizar
        print("\n🎨 Generando gráficas...")
        filename = Path(file_path).stem
        results_dir = Path(out_dir) if out_dir else Path('results')
        
        self.plot_p300_windows(
            eeg_data, windows, 
            channel_names=channel_names[:8],
            save=save_results, 
            out_dir=results_dir,
            filename=filename
        )
        
        self.plot_p300_comparison(
            eeg_data, windows,
            channel_names=channel_names[:8],
            save=save_results,
            out_dir=results_dir,
            filename=filename
        )
        
        # Guardar resultados
        if save_results and out_dir:
            results_dir = Path(out_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar características en CSV
            features_path = results_dir / f'{filename}_p300_features.csv'
            p300_features.to_csv(features_path, index=False)
            print(f"\n💾 Características guardadas en: {features_path}")
        
        return {
            'eeg_data': eeg_data,
            'windows': windows,
            'p300_features': p300_features,
            'filename': filename
        }


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Configuración
    user = "User1"
    letter = "A"
    trial = 0
    type_category = "Letters"
    
    file_path = f'captures/{user}/{type_category}/{user}_{letter}_{trial}.edf'
    
    # Crear extractor
    extractor = P300WindowExtractor(data_path='.')
    
    # Canales a analizar (puedes seleccionar un subconjunto)
    canales_interes = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    
    # Analizar archivo
    results = extractor.analyze_single_trial(
        file_path=file_path,
        channel_names=canales_interes,
        save_results=True,
        out_dir=f'results/{user}/{type_category}'
    )
    
    if results:
        print("\n✅ Análisis completado exitosamente")
        print(f"\nResumen de P300:")
        print(results['p300_features'].to_string())
