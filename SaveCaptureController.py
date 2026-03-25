import pandas as pd
import os
import re

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

    def start_capture(self, user, character_type, character,duration):
        path = 'captures'
        path_user= f"{path}/{user}/{character_type}/"
        filename = f"{user}_{character}_"
        numero ="0"
        ext = ".csv"

        full_path=  path_user+filename+ numero + ext
        print("fulpath: ",full_path)
        if os.path.exists(full_path):
            print("YA existe")
            lita = os.listdir(path_user)
            ultf=lita[-1]
            sub=ultf.split('_')
            lastnum=sub[-1]
            nuevoNum = str(int(lastnum[0]) + 1)
            full_path=  path_user+filename+ nuevoNum + ext
            

        print("Iniciando captura...")
        data = self.serial_monitor.start_recording(duration)
        self.save_capture(full_path,data)