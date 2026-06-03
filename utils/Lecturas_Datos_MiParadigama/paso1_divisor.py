"""
PASO 1 — Divisor de tiempos con Corrección de Línea Base (Baseline)
===================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
USUARIO       = 'UserPony'
TIPO_COMANDO  = 'Letters'          
N_TRIALS      = 30                 
N_AVG         = 2                 
AVG_STEP      = 1               
SAMPLING_RATE = 250                

COMANDOS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","Ñ","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "0","1","2","3","4","5","6","7","8","9",
    "DEL","HOME","ENTER"
]

VENTANAS = {
    'p300': (0.5, 1.2),    
    'post': (1.2, 2.0),    
}

CARPETA_ENTRADA = Path(f'captures/{USUARIO}/{TIPO_COMANDO}')
CARPETA_SALIDA  = Path(f'results/{USUARIO}/Separados_Avg')
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

def apply_baseline_correction(signals: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    signals = np.asarray(signals, dtype=float)
    n_muestras_baseline = int(0.5 * fs) 
    baseline_mean = signals[:, :n_muestras_baseline].mean(axis=1, keepdims=True)
    return signals - baseline_mean

def preprocess_eeg(signals: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    signals  = np.asarray(signals, dtype=float)
    nyquist  = 0.5 * fs
    b, a     = scipy_signal.butter(4, [1.0 / nyquist, 40.0 / nyquist], btype='band')
    filtered = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered[i] = scipy_signal.filtfilt(b, a, signals[i])

    b_n, a_n = scipy_signal.iirnotch(60.0, Q=30.0, fs=fs)
    for i in range(signals.shape[0]):
        filtered[i] = scipy_signal.filtfilt(b_n, a_n, filtered[i])

    return filtered

def ventana_muestras(nombre: str) -> tuple:
    t_ini, t_fin = VENTANAS[nombre]
    return int(t_ini * SAMPLING_RATE), int(t_fin * SAMPLING_RATE)

def cargar_trial(usuario: str, cmd: str, trial: int) -> np.ndarray | None:
    ruta = CARPETA_ENTRADA / f'{usuario}_{cmd}_{trial}.csv'
    if not ruta.exists():
        return None
    try:
        df      = pd.read_csv(ruta)
        
        # LÍNEA ANTIVIRUS DE RAÍZ: Rellena pérdidas de paquetes
        df      = df.ffill().bfill().fillna(0)
        
        cols_ch = list(df.columns[1:])          
        arr     = df[cols_ch].to_numpy().T       
        
        arr     = apply_baseline_correction(arr)
        arr     = preprocess_eeg(arr)
        return arr
    except Exception as e:
        print(f'    ⚠ Error leyendo {ruta.name}: {e}')
        return None

def procesar_comando(cmd: str) -> dict:
    trials = []
    for t in range(N_TRIALS):
        arr = cargar_trial(USUARIO, cmd, t)
        if arr is not None:
            trials.append(arr)

    if len(trials) == 0:
        return {'cmd': cmd, 'trials_ok': 0, 'epocas': 0}

    n_cols = min(a.shape[1] for a in trials)
    trials = [a[:, :n_cols] for a in trials]

    epocas_guardadas = 0
    idx_avg = 0

    for inicio in range(0, len(trials) - N_AVG + 1, AVG_STEP):
        bloque   = trials[inicio : inicio + N_AVG]
        promedio = np.mean(bloque, axis=0)           

        ruta_ref = CARPETA_ENTRADA / f'{USUARIO}_{cmd}_0.csv'
        if ruta_ref.exists():
            df_ref   = pd.read_csv(ruta_ref)
            col_names = list(df_ref.columns[1:])
        else:
            col_names = [f'ch{i}' for i in range(promedio.shape[0])]

        for ventana in ('p300', 'post'):
            ini_m, fin_m = ventana_muestras(ventana)
            segmento     = promedio[:, ini_m:fin_m].T   
            df_out       = pd.DataFrame(segmento, columns=col_names)
            nombre       = f'{USUARIO}_{cmd}_{idx_avg}_{ventana}.csv'
            df_out.to_csv(CARPETA_SALIDA / nombre, index=False)

        epocas_guardadas += 1
        idx_avg          += 1

    return {'cmd': cmd, 'trials_ok': len(trials), 'epocas': epocas_guardadas}

if __name__ == '__main__':
    epocas_esperadas = max(0, (N_TRIALS - N_AVG) // AVG_STEP + 1)
    print('=' * 65)
    print('PASO 1 — Divisor Limpio (Baseline Corrected + NaN Handling)')
    print('=' * 65 + '\n')

    resumen = []
    for cmd in COMANDOS:
        r = procesar_comando(cmd)
        resumen.append(r)

    print(f'\n  ✓ Listo → {CARPETA_SALIDA}\n')