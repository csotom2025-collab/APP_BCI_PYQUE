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
        Normaliza las señales EEG por canal usando z-score.

        Args:
            signals (numpy.array): Array de señales EEG (n_channels, n_samples)

        Returns:
            numpy.array: Señales normalizadas
        """
        signals = np.asarray(signals, dtype=float)
        means = np.mean(signals, axis=1, keepdims=True)
        stds = np.std(signals, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return (signals - means) / stds

    def preprocess_eeg(self, signals, lowcut=1.0, highcut=40.0, fs=None):
        """
        Preprocesa las señales EEG aplicando filtro bandpass + notch.

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

        # Asegurar que la entrada es float para filtrar
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
    
    def extraer_info_archivo(self, nombre_archivo):
        """
        Extrae user, letra y trial del nombre del archivo.
        Formato esperado: User{n}_{letra}_{trial}.csv o {user}_{letra}_{trial}.csv
        
        Returns:
            tuple: (user, letra, trial_num) o None si no coincide el patrón
        """
        # Intentar extraer con patrón User27_A_0 o usermar_0_0
        match = re.match(r'(User\d+|[A-Za-z0-9]+)_([A-Z0-9])_(\d+)', nombre_archivo)
        if match:
            return match.group(1), match.group(2), int(match.group(3))
        
        # Patrón alternativo
        match = re.match(r'([A-Za-z0-9_]+)_([A-Z0-9])_(\d+)', nombre_archivo)
        if match:
            return match.group(1), match.group(2), int(match.group(3))
        
        return None
    
    def determinar_tipo_comando(self, letra):
        """
        Determina si la letra es Char, Digit o Comando
        
        Returns:
            str: 'Char', 'Digit' o 'Comandos'
        """
        if letra.isdigit():
            return 'Digit'
        #checa de la letra esté entre A-Z (mayúscula)
        elif letra.upper() in 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ':
            return 'Char'
        else:
            return 'Comandos'
    
    def procesar_archivo(self, ruta_csv, carpeta_salida_base='results/user'):
        """
        Procesa un archivo CSV y genera dos CSVs separados (P300 y trial completo)
        
        Args:
            ruta_csv (str o Path): Ruta al archivo CSV de entrada
            carpeta_salida_base (str): Ruta base para guardar resultados
        
        Returns:
            dict: Información sobre el procesamiento
        """
        ruta_csv = Path(ruta_csv)
        
        # Validar que el archivo existe
        if not ruta_csv.exists():
            return {'exito': False, 'error': f'Archivo no encontrado: {ruta_csv}'}
        
        # Extraer información del nombre del archivo
        nombre_sin_ext = ruta_csv.stem  # Nombre sin .csv
        info = self.extraer_info_archivo(nombre_sin_ext)
        
        if not info:
            return {'exito': False, 'error': f'No se pudo extraer info del nombre: {nombre_sin_ext}'}
        
        user, letra, trial_num = info
        tipo_comando = self.determinar_tipo_comando(letra)
        
        try:
            # Leer CSV
            df = pd.read_csv(ruta_csv)
            print(f"✓ CSV cargado: {ruta_csv.name} ({len(df)} muestras)")
            
            # Obtener columnas de canales (todas excepto la primera que es timestamp)
            cols_canales = list(df.columns[1:])
            
            if len(cols_canales) == 0:
                return {'exito': False, 'error': 'No se encontraron columnas de canales'}

            # Normalizar y preprocesar las señales EEG antes de dividir por tiempos
            canales_array = df[cols_canales].to_numpy().T
            try:
                canales_normalizados = self.normalize_eeg(canales_array)
                canales_filtrados = self.preprocess_eeg(canales_normalizados, fs=self.sampling_rate)
                df.loc[:, cols_canales] = canales_filtrados.T
                print(f"  ✓ Normalización + preprocesamiento EEG aplicado: {len(cols_canales)} canales")
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
            print(f"  ✓ P300 guardado: {archivo_p300.name} ({len(df_p300)} muestras)")
            
            # 2. Extraer trial completo (0 - 2.0s)
            inicio_trial, fin_trial = self.muestras['trial_completo']
            df_trial = df.iloc[inicio_trial:fin_trial][cols_canales].reset_index(drop=True)
            
            archivo_trial = carpeta_salida / f"{user}_{letra}_{trial_num}.csv"
            df_trial.to_csv(archivo_trial, index=False)
            print(f"  ✓ Trial completo guardado: {archivo_trial.name} ({len(df_trial)} muestras)")
            
            # 3. Extraer post_estimulo (1.2 - 2.0s)
            inicio_post, fin_post = self.muestras['post_estimulo']
            df_post = df.iloc[inicio_post:fin_post][cols_canales].reset_index(drop=True)

            archivo_post = carpeta_salida / f"{user}_{letra}_{trial_num}_post_estimulo.csv"
            df_post.to_csv(archivo_post, index=False)

            print(f"  ✓ Post-estímulo guardado: {archivo_post.name} ({len(df_post)} muestras)")
            
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
                'canales': len(cols_canales)
            }
        
        except Exception as e:
            return {'exito': False, 'error': f'Error procesando archivo: {str(e)}'}
    
    def procesar_carpeta(self, carpeta_entrada, patron='*.csv', carpeta_salida_base='results/user'):
        """
        Procesa todos los archivos CSV de una carpeta
        
        Args:
            carpeta_entrada (str o Path): Carpeta con archivos CSV
            patron (str): Patrón de archivos a procesar
            carpeta_salida_base (str): Ruta base para guardar resultados
        
        Returns:
            list: Lista de resultados para cada archivo procesado
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
    
    # Procesar un archivo individual
    print("="*70)
    print("SEPARADOR DE TRIALS P300 - EEG")
    print("="*70 + "\n")
    
    usuario= 'Usermar'
    tpComando = 'Letters'
    letras=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # Ejemplo: procesar archivo User27_A_0.csv
    for trial in range(30):  # Procesar trials 0, 1 y 2
        for letra in letras:
            resultado = separador.procesar_archivo(
                f'{usuario}/{tpComando}/{usuario}_{letra}_{trial}.csv',
                carpeta_salida_base=f'results/{usuario}'
            )
            
            if resultado['exito']:
                print(f"\n✓ Procesamiento exitoso:")
                print(f"  Usuario: {resultado['user']}")
                print(f"  Letra: {resultado['letra']}")
                print(f"  Trial: {resultado['trial']}")
                print(f"  Tipo: {resultado['tipo']}")
                print(f"  Canales: {resultado['canales']}")
                print(f"  Muestras P300: {resultado['muestras_p300']}")
                print(f"  Muestras Trial: {resultado['muestras_trial']}")
                print(f"  Muestras Post: {resultado['muestras_post']}")
            else:
                print(f"\n✗ Error: {resultado['error']}")
            
            # Para procesar una carpeta completa (descomentar):
            # resultados = separador.procesar_carpeta('User27/Letters/', carpeta_salida_base='results/user')