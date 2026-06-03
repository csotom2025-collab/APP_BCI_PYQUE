"""
PASO 2 — Extracción de Amplitudes Temporales (Diezmado BCI)
============================================================
Lee las ventanas de voltaje crudo y guarda vectores de amplitud temporal pura.
Genera dos archivos por época: uno para la clase P300 y otro para la clase Post (Ruido).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
USUARIO  = 'UserPony'
SAMPLING_RATE = 250

COMANDOS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","Ñ","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "0","1","2","3","4","5","6","7","8","9",
    "DEL","HOME","ENTER"
]

CANALES = ["Oz","Po7","Po4","Po3","P4","P3","Po8","Pz",
           "Fz","F2","F3","F4","AF3","Cz","AF4","F1"]

CARPETA_ENTRADA = Path(f'results/{USUARIO}/Separados_Avg')
CARPETA_SALIDA  = Path(f'results/{USUARIO}/Features_Avg')
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# NUEVA EXTRACCIÓN: DIEZMADO TEMPORAL (DOWN-SAMPLING EN VOLTAJE)
# ══════════════════════════════════════════════════════════════
def extraer_features_df(df: pd.DataFrame, canales: list) -> dict:
    """
    ¡CAMBIADO!: En lugar de FFT, extrae los puntos de amplitud en el tiempo.
    Toma las primeras 150 muestras de la ventana y extrae 1 de cada 10.
    Genera 15 características secuenciales por canal.
    """
    feats = {}
    cols_ok = [c for c in canales if c in df.columns]
    
    for canal in cols_ok:
        datos = df[canal].to_numpy().astype(float)
        
        # Homogeneizar tamaño limitando a 150 puntos (0.6 segundos de señal)
        datos_recortados = datos[:150]
        
        # Downsampling a ~25Hz tomando un punto cada 10 muestras
        datos_diezmados = datos_recortados[::10] # Retorna exactamente 15 puntos
        
        for i, val in enumerate(datos_diezmados):
            feats[f'sample_{i}_{canal}'] = float(val)
            
    return feats


if __name__ == '__main__':
    print('=' * 65)
    print('PASO 2 — Features Temporales Puras (Amplitudes)')
    print(f'  Canales Seleccionados : {len(CANALES)}')
    print('=' * 65 + '\n')

    total_ok  = 0
    total_err = 0

    for cmd in COMANDOS:
        idx = 0
        while True:
            f_p300 = CARPETA_ENTRADA / f'{USUARIO}_{cmd}_{idx}_p300.csv'
            f_post = CARPETA_ENTRADA / f'{USUARIO}_{cmd}_{idx}_post.csv'

            if not f_p300.exists() or not f_post.exists():
                break   

            try:
                df_p300 = pd.read_csv(f_p300)
                df_post = pd.read_csv(f_post)

                canales = CANALES if CANALES else list(df_p300.columns)

                # Extraer amplitudes temporales crudas (mismo nombres de columnas para ambos)
                feats_p300 = extraer_features_df(df_p300, canales)
                feats_post = extraer_features_df(df_post, canales)

                # Guardar archivo P300 (Será nuestra Clase Target = 1)
                feats_p300['cmd_label'] = str(cmd)
                salida_p300 = CARPETA_SALIDA / f'{USUARIO}_{cmd}_{idx}_p300_features.csv'
                pd.DataFrame([feats_p300]).to_csv(salida_p300, index=False)

                # Guardar archivo Post (Será nuestra Clase Non-Target = 0)
                feats_post['cmd_label'] = str(cmd)
                salida_post = CARPETA_SALIDA / f'{USUARIO}_{cmd}_{idx}_post_features.csv'
                pd.DataFrame([feats_post]).to_csv(salida_post, index=False)
                
                total_ok += 2

            except Exception as e:
                print(f'    ✗ Error en {USUARIO}_{cmd}_{idx}: {e}')
                total_err += 1

            idx += 1

        if idx > 0:
            print(f'  {cmd:<8} → {idx} épocas procesadas (P300 + Post)')

    print(f'\n  ✓ Vectores de amplitud guardados : {total_ok}')
    print(f'  Listo → {CARPETA_SALIDA}\n')