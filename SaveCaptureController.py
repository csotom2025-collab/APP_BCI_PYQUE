import pandas as pd
import os

from SerialMonitor import SignalsWindow


class controllerSaveCapture:
    def __init__(self, serial_monitor: SignalsWindow):
        self.serial_monitor = serial_monitor

    def save_capture(self, filename, df):
        print(f"Guardando captura en: {filename}")
        # Crear la ruta si no existe
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directorio creado: {os.path.abspath(directory)}")
        
        # Guardar el DataFrame actual en un archivo CSV
        print(f"Guardando captura en: {filename}")
        try:
            df.to_csv(filename, index=False)
            print("Captura guardada exitosamente.")
        except Exception as e:
            print(f"Error al guardar la captura: {e}")

    def start_capture(self, user, character_type, character):
        recording_duration = 5 
        path = 'captures/'  
        filename = f"{path}/{user}/{user}_{character_type}_{character}.csv"
        print("Iniciando captura...")
        data = self.serial_monitor.start_recording(recording_duration)
        self.save_capture(filename,data)