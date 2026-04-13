import pandas as pd
import os
import re
import time
import numpy as np
import mne
from windows.SerialMonitorWindow import SignalsWindow
from PyQt6.QtCore import QTimer,Qt,QThread,pyqtSignal
# import matplotlib.plt as plt
# def guardar_grafica(df):
#     plt.plot(df)
#     pass


class controllerSaveCapture:
    def __init__(self, serial_monitor: SignalsWindow):
        self.serial_monitor = serial_monitor
        self.edf_full_path = None

    def save_capture(self):
        filename = self.full_path
        df = self.serial_monitor.return_recorded_data()
        
        # Limpiar el DataFrame eliminando .0 innecesarios
        df = self.clean_df_file(df)
        self.save_capture_edf()
        print(f"Guardando captura en: {filename}")
        # Crear la ruta si no existe
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            #print(f"Directorio creado: {os.path.abspath(directory)}")
        
        # Guardar el DataFrame actual en un archivo CSV
        try:
            df.to_csv(filename, index=False)
            print("Captura guardada exitosamente.")
        except Exception as e:
            print(f"Error al guardar la captura: {e}")
    def clean_df_file(self, df):
        """Limpia el DataFrame eliminando decimales innecesarios (.0)"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            # Verificar si la columna es numérica
            if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Convertir a float primero para manejar NaN
                float_values = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Verificar si todos los valores son enteros (sin parte decimal)
                if float_values.notna().all():
                    # Comprobar si todos los valores son enteros
                    is_integer = (float_values == float_values.astype(int)).all()
                    
                    if is_integer:
                        # Convertir a int, eliminando el .0
                        df_clean[col] = float_values.astype(int)
                    else:
                        # Mantener como float si tiene decimales reales
                        df_clean[col] = float_values
        
        return df_clean

    def clean_existing_file(self, file_path):
        """Limpia un archivo CSV existente eliminando .0 innecesarios"""
        try:
            # Leer el archivo
            df = pd.read_csv(file_path)
            
            # Limpiar el DataFrame
            df_clean = self.clean_df_file(df)
            
            # Guardar el archivo limpio
            df_clean.to_csv(file_path, index=False)
            
            print(f"Archivo limpiado exitosamente: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error al limpiar archivo {file_path}: {e}")
            return False

    def clean_all_captures(self, user=None):
        """Limpia todos los archivos de captura eliminando .0 innecesarios"""
        import glob
        
        path = 'captures'
        cleaned_count = 0
        
        if user:
            # Limpiar solo archivos de un usuario específico
            pattern = f"{path}/{user}/**/*.csv"
        else:
            # Limpiar todos los archivos CSV
            pattern = f"{path}/**/*.csv"
        
        for file_path in glob.glob(pattern, recursive=True):
            if self.clean_existing_file(file_path):
                cleaned_count += 1
        
        print(f"Archivos limpiados: {cleaned_count}")
        return cleaned_count
    def save_capture_edf(self):
        filename = self.edf_full_path
        df = self.serial_monitor.return_recorded_data()
        
        # Limpiar el DataFrame eliminando .0 innecesarios
        df = self.clean_df_file(df)
        
        # --- PASO 1: Quitar la columna de tiempo si existe ---
        # if 'Tm' in df.columns:
        #     df_signals = df.drop(columns=['Tm'])
        # else:
        #     df_signals = df
        df_signals = df
        try:
            # Extraer datos (canales, muestras)
            data = df_signals.values.T
            
            # --- PASO 2: Limpieza y Escalamiento ---
            data = np.nan_to_num(data, nan=0.0)
            
            # IMPORTANTE: MNE espera VOLTIOS. 
            # Si tu BCI da valores grandes (como 200,000), divídelos.
            # Esto asegura que el texto en la cabecera EDF sea corto (ej. "0.0001")
            data = data * 1e-6  # Convertir de microvoltios a voltios
            
            # --- PASO 3: Crear Info de MNE ---
            ch_names = list(df_signals.columns)
            sfreq = 250  # Ajusta a la frecuencia real de tu dispositivo
            ch_types = ['eeg'] * len(ch_names)
            
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            raw = mne.io.RawArray(data, info)
            
            # Exportar
            raw.export(filename, fmt='edf', overwrite=True)
            print(f"Éxito: {filename}")
            
        except Exception as e:
            print(f"Error al guardar EDF: {e}")
            # Tu respaldo en CSV...

    def start_capture(self, user, character_type, character, duration):
        path = 'captures'
        path_user= f"{path}/{user}/{character_type}/"
        filename = f"{user}_{character}_"
        numero ="0"
        ext = ".csv"

        self.full_path = path_user + filename + numero + ext
        # Generar también la ruta EDF
        self.edf_full_path = path_user + filename + numero + ".edf"
        
        #print("fulpath: ",full_path)
        if os.path.exists(self.full_path):
            lita = os.listdir(path_user)
            lista_separada = [file.split('_') for file in lita]
            lista_label = [f for f in lista_separada if f[1] == character]
            nums = [int(file[-1][:-4]) for file in lista_label]
            sorted_nums = sorted(nums)
            ultimo_numero = sorted_nums[-1]
            nuevoNum = str(ultimo_numero + 1)
            self.full_path = path_user + filename + nuevoNum + ext
            self.edf_full_path = path_user + filename + nuevoNum + ".edf"
            
        self.serial_monitor.start_recording(duration)
        qtime = QTimer()
        qtime.singleShot(duration*1000+600, self.save_capture)
        #qtime.singleShot(duration*1000+600, self.save_capture_edf)
        
