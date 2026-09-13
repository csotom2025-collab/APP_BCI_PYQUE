# -*- coding: utf-8 -*-
"""
Configuracion central del proyecto BCI.
Ajusta estos valores segun tu instalacion / dataset real.
"""

import os

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = "D:/APP_BCI_PYQUE/captures"

# Carpetas de "tpComando" tal cual aparecen en el filesystem.
TP_COMANDOS = ["Letters", "Numbers", "Controls"]

# Usuarios a incluir en el dataset (agrega todos los que tengas grabados)
USUARIOS = ["UserCMSM"]  # <-- EDITA esta lista con tus usuarios reales

# ---------------------------------------------------------------------------
# Vocabulario de "letra" por cada tpComando (nombre usado en el archivo)
# ---------------------------------------------------------------------------
LETRAS_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # A..Z (27)
LETRAS_NUMBERS = [str(d) for d in range(0, 10)]                   # 0..9 (10)
LETRAS_CONTROLS = ["───", "↩", "⟵"]                             # (3)

LETRAS_POR_TIPO = {
    "Letters": LETRAS_LETTERS,
    "Numbers": LETRAS_NUMBERS,
    "Controls": LETRAS_CONTROLS,
}

# ---------------------------------------------------------------------------
# Canales EEG disponibles (orden esperado en las columnas del CSV)
# ---------------------------------------------------------------------------
CHANNEL_NAMES = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2","P8", "T8", "F8", "AF4", "FC6", "F4"]
#CHANNEL_NAMES = ["P7","P8","O1","O2","F3","F4","F7","F8","AF3","AF4"]
# ---------------------------------------------------------------------------
# Parametros de la señal / grabacion
# ---------------------------------------------------------------------------
# Frecuencia de muestreo (Hz). AJUSTA al valor real de tu dispositivo (Emotiv=128).
FS = 128

# Duracion del trial en segundos, segun tu protocolo (0.0 - 2.0s)
TRIAL_DURATION_S = 2.0

# Ventana de interes P300 dentro del trial (0.5 - 1.2s). Se usa opcionalmente
# para recortar la señal antes de extraer caracteristicas.
P300_WINDOW_S = (0.5, 1.2)

# Ventana de pre-estimulo usada para la correccion de linea base (0.0 - 0.5s),
# tal como describe el protocolo de grabacion.
BASELINE_WINDOW_S = (0.0, 0.5)

# Si True, a cada trial se le resta la media de su ventana de pre-estimulo
# (por canal) antes de calcular cualquier caracteristica. Muy recomendado:
# elimina el offset/deriva de linea base y hace comparables los trials entre si.
APPLY_BASELINE_CORRECTION = True

# Si True, se recorta cada trial a la ventana P300_WINDOW_S antes de extraer
# caracteristicas (recomendado: la respuesta discriminativa esta ahi).
# Si False, se usa el trial completo (0 - 2.0s).
USE_P300_WINDOW_ONLY = False

# Parametros de ventaneo para extract_features. Con window_size = n_samples del
# segmento y overlap = 0 se obtiene EXACTAMENTE 1 vector de caracteristicas por
# trial (recomendado para clasificacion por trial).
WINDOW_OVERLAP = 0

# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------
OUTPUT_DIR = "D:/EEG_Python/PIPlieneTiempoReal/OutputOPt_14Canales_LDA_2"
FEATURES_CSV = os.path.join(OUTPUT_DIR, "features_dataset.csv")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "resultados_modelos.csv")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_bci_model.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
