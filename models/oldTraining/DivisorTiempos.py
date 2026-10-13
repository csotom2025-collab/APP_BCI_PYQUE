import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import re


class SeparacionTiempos:
    """
    Clase para separar trials de EEG según fases temporales (P300).
    
    Flujo de una grabación (Trial de 2 seg):
    - Pre-estímulo (0.0 - 0.5s): Corrección de línea base
    - Estímulo (0.5s): Flash en matriz, marcador binario
    - Respuesta Evocada P300 (0.5 - 1.2s): Aumento de voltaje si letra correcta
    - Post-estímulo (1.2 - 2.0s): Recuperación
    """
    
    def __init__(self, sampling_rate=250):
        """
        Args:
            sampling_rate (int): Frecuencia de muestreo en Hz (default: 250 Hz de los archivos)
        """
        self.sampling_rate = sampling_rate
        
        # Definir intervalos de tiempo en segundos
        self.tiempos = {
            'pre_estimulo': (0.0, 0.5),    # 0-125 muestras
            'estimulo': (0.5, 0.5),         # muestra 125
            'p300': (0.5, 1.2),             # 125-300 muestras (Respuesta Evocada)
            'post_estimulo': (1.2, 2.0),    # 300-500 muestras
            'trial_completo': (0.0, 2.0)    # 0-500 muestras
        }
        
        # Convertir tiempos a muestras (samples)
        self.muestras = {}
        for key, (t_ini, t_fin) in self.tiempos.items():
            inicio = int(t_ini * self.sampling_rate)
            fin = int(t_fin * self.sampling_rate)
            self.muestras[key] = (inicio, fin)
    
    def dividir_tiempo(self, partes):
        """
        Método original de la clase (se mantiene por compatibilidad)
        """
        if partes <= 0:
            raise ValueError("El número de partes debe ser mayor que cero.")
        return self.tiempos['trial_completo'][1] / partes
    
    def normalize_eeg(self, signals):
        """
        Normaliza las señales EEG por canal usando z-score (Opcional).
        """
        signals = np.asarray(signals, dtype=float)
        means = np.mean(signals, axis=1, keepdims=True)
        stds = np.std(signals, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return (signals - means) / stds

    def preprocess_eeg(self, signals, lowcut=1.0, highcut=40.0, fs=None):
        """
        Preprocesa las señales EEG aplicando filtro bandpass + notch.
        """
        if fs is None:
            fs = self.sampling_rate

        signals = np.asarray(signals, dtype=float)

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        filtered_signals = np.zeros_like(signals)
        for i in range(signals.shape[0]):
            filtered_signals[i, :] = signal.filtfilt(b, a, signals[i, :])

        # Notch filter de 40 Hz para remover ruido de línea
        F_notch = 40.0
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(F_notch, Q, fs)
        for i in range(signals.shape[0]):
            filtered_signals[i, :] = signal.filtfilt(b_notch, a_notch, filtered_signals[i, :])

        return filtered_signals
    
    def extraer_info_archivo(self, nombre_archivo,unknown=False):
        """
        Extrae user, letra y trial del nombre del archivo.
        """
        print("nombre_archivo",nombre_archivo)
        if unknown:
            match = re.search(r'([A-Za-z0-9]+)_UNKNOWN_(\d+)', nombre_archivo)

            if match:
                usuario = match.group(1)
                numero = int(match.group(2))
                print(f"Usuario: {usuario}, Número: {numero}")
                print(f"Usuario: {usuario}, Número: {numero}")
                return usuario, numero

        
        match = re.match(r'([A-Za-z0-9_]+)_([A-Z0-9ñÑ])_(\d+)', nombre_archivo)
        if match:
            return match.group(1), match.group(2), int(match.group(3))

        print("retonanrodnoa idfaon ")
        return None
    
    def determinar_tipo_comando(self, letra):
        """
        Determina si la letra es Char, Digit o Comando
        """
        if letra.isdigit():
            return 'Digit'
        elif letra.upper() in 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ':
            return 'Char'
        else:
            return 'Comandos'
    
    def procesar_archivo(self, ruta_csv, carpeta_salida_base='results/user',unknown=False):
        """
        Procesa un archivo CSV y genera los archivos separados con la corrección requerida.
        """
        ruta_csv = Path(ruta_csv)
        print(f"Procesando archivo: {ruta_csv}")
        if not ruta_csv.exists():
            return {'exito': False, 'error': f'Archivo no encontrado: {ruta_csv}'}
        
        nombre_sin_ext = ruta_csv.stem
        if not unknown:
            info = self.extraer_info_archivo(nombre_sin_ext)
            
            if not info:
                return {'exito': False, 'error': f'No se pudo extraer info del nombre: {nombre_sin_ext}'}
        
            user, letra, trial_num = info
            tipo_comando = self.determinar_tipo_comando(letra)
        else:
            info = self.extraer_info_archivo(nombre_sin_ext,unknown=True)
                        
            if not info:
                return {'exito': False, 'error': f'No se pudo extraer info del nombre: {nombre_sin_ext}'}
        
            user, trial_num = info
            letra = "UNKNOWN"
            tipo_comando = self.determinar_tipo_comando(letra)
           
        print("infgo",info)
        try:
            df = pd.read_csv(ruta_csv)
            cols_canales = list(df.columns[1:])
            
            if len(cols_canales) == 0:
                return {'exito': False, 'error': 'No se encontraron columnas de canales'}

            canales_array = df[cols_canales].to_numpy().T
            
            try:
                # 1. FILTRADO (Retorna floats)
                canales_filtrados = self.preprocess_eeg(canales_array, fs=self.sampling_rate)
                
                # 2. CORRECCIÓN DE LÍNEA BASE REAL
                inicio_pre, fin_pre = self.muestras['pre_estimulo']
                linea_base_media = np.mean(canales_filtrados[:, inicio_pre:fin_pre], axis=1, keepdims=True)
                canales_procesados = canales_filtrados - linea_base_media
                
                # --- SOLUCIÓN AL ERROR DE DTYPE 'int64' ---
                # Forzamos a que las columnas del DataFrame sean float antes de guardar los datos decimales
                df[cols_canales] = df[cols_canales].astype(float)
                
                # Guardamos los datos corregidos sin usar .loc para evitar conflictos de alineación estricta
                df[cols_canales] = canales_procesados.T
                
            except Exception as e:
                return {'exito': False, 'error': f'Error en preprocesamiento EEG: {e}'}
            
            # Crear carpeta de salida
            carpeta_salida = Path(carpeta_salida_base) / tipo_comando / 'Separados'
            carpeta_salida.mkdir(parents=True, exist_ok=True)
            
            # 1. Extraer P300 (0.5 - 1.2s)
            inicio_p300, fin_p300 = self.muestras['p300']
            df_p300 = df.iloc[inicio_p300:fin_p300][cols_canales].reset_index(drop=True)
            archivo_p300 = carpeta_salida / f"{user}_{letra}_{trial_num}_P300.csv"
            df_p300.to_csv(archivo_p300, index=False)
            
            # 2. Extraer trial completo (0 - 2.0s)
            inicio_trial, fin_trial = self.muestras['trial_completo']
            df_trial = df.iloc[inicio_trial:fin_trial][cols_canales].reset_index(drop=True)
            archivo_trial = carpeta_salida / f"{user}_{letra}_{trial_num}.csv"
            df_trial.to_csv(archivo_trial, index=False)
            
            # 3. Extraer post_estimulo (1.2 - 2.0s)
            inicio_post, fin_post = self.muestras['post_estimulo']
            df_post = df.iloc[inicio_post:fin_post][cols_canales].reset_index(drop=True)
            archivo_post = carpeta_salida / f"{user}_{letra}_{trial_num}_post_estimulo.csv"
            df_post.to_csv(archivo_post, index=False)
            
            return {
                'exito': True,
                'user': user,
                'letra': letra,
                'trial': trial_num,
                'tipo': tipo_comando,
                'archivo_p300': str(archivo_p300),
                'archivo_trial': str(archivo_trial),
                'archivo_post': str(archivo_post),
                'muestras_p300': len(df_p300),
                'muestras_trial': len(df_trial),
                'muestras_post': len(df_post),
                'canales': len(cols_canales),
            }
        
        except Exception as e:
            return {'exito': False, 'error': f'Error procesando archivo: {str(e)}'}
    
    def procesar_carpeta(self, carpeta_entrada, patron='*.csv', carpeta_salida_base='results/user'):
        """
        Procesa todos los archivos CSV de una carpeta
        """
        carpeta_entrada = Path(carpeta_entrada)
        if not carpeta_entrada.exists():
            print(f"✗ Carpeta no encontrada: {carpeta_entrada}")
            return []
        
        archivos = list(carpeta_entrada.glob(patron))
        print(f"\nEncontrados {len(archivos)} archivos CSV\n")
        
        resultados = []
        for archivo in archivos:
            print(f"Procesando: {archivo.name}")
            resultado = self.procesar_archivo(archivo, carpeta_salida_base)
            resultados.append(resultado)
            if resultado['exito']:
                print(f"  ✓ Completado\n")
            else:
                print(f"  ✗ Error: {resultado['error']}\n")
        
        return resultados


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    separador = SeparacionTiempos(sampling_rate=250)
    
    print("="*70)
    print("SEPARADOR DE TRIALS P300 - EEG (CORREGIDO)")
    print("="*70 + "\n")
    
    # Puedes cambiar los parámetros aquí para probar con User3 u otros usuarios
    usuario = 'User3'
    tpComando = 'Numbers'
    # Lista de números o letras según corresponda tu carpeta
    comandos = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] 
    
    for trial in range(30):  
        for cmd in comandos:
            # Ruta dinámica ajustada al formato de tus carpetas
            resultado = separador.procesar_archivo(
                f'captures/{usuario}/{tpComando}/{usuario}_{cmd}_{trial}.csv',
                carpeta_salida_base=f'results/{usuario}'
            )
            
            if resultado['exito']:
                print(f"  Usuario: {resultado['user']} | Comando: {resultado['letra']} | Trial: {resultado['trial']} -> OK")
            else:
                print(f"  ✗ Error en {cmd}_{trial}: {resultado['error']}")