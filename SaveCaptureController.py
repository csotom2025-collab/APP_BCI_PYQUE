import pandas as pd
import os
import re
import time
from SerialMonitor import SignalsWindow
from PyQt6.QtCore import QTimer,Qt,QThread,pyqtSignal
# import matplotlib.plt as plt
# def guardar_grafica(df):
#     plt.plot(df)
#     pass



def generate_edf_file(filename):
    ext=".edf"
    status="reposo"
class controllerSaveCapture:
    def __init__(self, serial_monitor: SignalsWindow):
        self.serial_monitor = serial_monitor

    def save_capture(self):
        filename=self.full_path
        df = self.serial_monitor.return_recorded_data()
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

    def start_capture(self, user, character_type, character,duration):
        path = 'captures'
        path_user= f"{path}/{user}/{character_type}/"
        filename = f"{user}_{character}_"
        numero ="0"
        ext = ".csv"

        self.full_path=  path_user+filename+ numero + ext
        #print("fulpath: ",full_path)
        if os.path.exists(self.full_path):
            lita = os.listdir(path_user)
            nums = [int(file.split('_')[-1][:-4]) for file in lita]
            sorted_nums=sorted(nums)
            ultimo_numero =sorted_nums[-1]
            nuevoNum = str(ultimo_numero+1)
            self.full_path=  path_user+filename+ nuevoNum + ext
            

        print("Iniciando captura...")
        self.serial_monitor.start_recording(duration)
        qtime = QTimer()
        qtime.singleShot(duration*1000+400,self.save_capture)
        #time.sleep(duration)
        # data = 
        # self.save_capture(full_path,data)