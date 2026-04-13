import matplotlib.pyplot as plt
import numpy as np
from utils.EEdfReader import EEGEDFReader

def main():
    data_path = "captures"
    eeg_reader = EEGEDFReader(data_path)
    
    user = "User9"
    letter = "A"
    info = eeg_reader.read_edf_file(f'{user}/Letters/{user}_{letter}_5.edf')
    
    # 2. Extraer datos y etiquetas
    signals = info['signals']  # Es una matriz (17, 250)
    labels = info['channel_names']
    
    # Identificar el índice del tiempo 'Tm' y los canales de señal
    idx_tm = labels.index('Tm')
    # Filtramos para quedarnos solo con los canales que no sean 'Tm'
    signal_indices = [i for i, label in enumerate(labels) if label != 'Tm']
    
    # 3. Configurar la figura (8 filas, 2 columnas para 16 canales)
    fig, axes = plt.subplots(8, 2, figsize=(14, 12))
    fig.suptitle(f"Señales EEG - {info['metadata']['filename']}", fontsize=16)
    axes = axes.flatten()

    # 4. Iterar solo sobre los canales de datos (saltando 'Tm')
    for plot_idx, sig_idx in enumerate(signal_indices):
        channel_name = labels[sig_idx]
        
        # Eje X: usamos los valores del canal 'Tm'
        # Eje Y: los valores del canal actual
        axes[plot_idx].plot(signals[idx_tm], signals[sig_idx], linewidth=0.8, color='blue')
        
        axes[plot_idx].set_title(f'Canal: {channel_name}', fontsize=10)
        axes[plot_idx].set_xlabel('Tiempo')
        axes[plot_idx].set_ylabel('Amplitud')
        axes[plot_idx].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el título superior
    plt.show()

if __name__ == "__main__":
    main()
