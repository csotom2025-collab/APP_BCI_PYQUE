from PyQt6.QtWidgets import QCheckBox, QMessageBox, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit,QMainWindow, QApplication, QGridLayout
import sys
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import QTimer, QTimer
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

class RecordingShowedWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_users()

    def setup_ui(self):
        self.setWindowTitle("Recording Showed Window")
        self.setGeometry(100, 100, 400, 300)
        # self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # self.label = QLabel("Recording Showed Window")
        # self.layout.addWidget(self.label)

        self.combo_box_users = QComboBox()
        self.combo_box_users.currentTextChanged.connect(self.load_dirs)
        

        self.combo_box_dirs = QComboBox()
        self.combo_box_users.currentTextChanged.connect(self.load_files)

        self.combo_box_files = QComboBox()

        self.button_show_plot = QPushButton("Mostrar Grafica")
        self.button_show_plot.clicked.connect(self.show_plot)

        self.checkbox_notch = QCheckBox("Aplicar filtro Notch")
        self.checkbox_baseline = QCheckBox("Eliminar línea base")

        self.checkbox_overlay_ploting = QCheckBox("Sobreponer señales")

        self.button_load_users = QPushButton("Actualizar Usuarios")
        self.button_load_users.clicked.connect(self.load_users)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(QLabel("Selecciona un usuario:"), 1, 0)
        self.grid_layout.addWidget(self.combo_box_users, 1, 1)
        self.grid_layout.addWidget(self.button_load_users, 1, 2)
        self.grid_layout.addWidget(QLabel("Selecciona un directorio:"), 2, 0)
        self.grid_layout.addWidget(self.combo_box_dirs, 2, 1)
        self.grid_layout.addWidget(QLabel("Selecciona un archivo:"), 3, 0)
        self.grid_layout.addWidget(self.combo_box_files, 3, 1)
        self.grid_layout.addWidget(self.button_show_plot, 4, 0, 1, 3)
        self.grid_layout.addWidget(QLabel("Opciones de procesamiento:"), 5, 0, 1, 3)
        self.grid_layout.addWidget(self.checkbox_notch, 6, 0)
        self.grid_layout.addWidget(self.checkbox_baseline, 6, 1)
        self.grid_layout.addWidget(QLabel("Opciones de visualización:"), 7, 0, 1, 3)
        self.grid_layout.addWidget(self.checkbox_overlay_ploting, 8, 0, 1, 3)
        self.grid_layout.addWidget(QPushButton("Cerrar Ventana"), 9, 0, 1, 3)
        self.grid_layout.itemAtPosition(9, 0).widget().clicked.connect(self.close_window)

        self.layout.addLayout(self.grid_layout)
        self.hide()
    
    def load_users(self):
        self.combo_box_users.clear()
        users = os.listdir("captures")
        self.combo_box_users.addItems(users)

    def load_dirs(self):
        user = self.combo_box_users.currentText()
        user_path = os.path.join("captures", user)
        dirs = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
        self.combo_box_dirs.clear()
        self.combo_box_dirs.addItems(dirs)
    def load_files(self):
        user = self.combo_box_users.currentText()
        dir_selected = self.combo_box_dirs.currentText()
        dir_path = os.path.join("captures", user, dir_selected)
        files = os.listdir(dir_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        self.combo_box_files.clear()
        self.combo_box_files.addItems(csv_files)

    def show_plot(self):
        user = self.combo_box_users.currentText()
        dir_selected = self.combo_box_dirs.currentText()
        filename = self.combo_box_files.currentText()
        file_path = os.path.join("captures", user, dir_selected, filename)
        letter = filename.split('_')[1]
        capture_number = filename.split('_')[2].split('.')[0]
        use_notch = self.checkbox_notch.isChecked()
        clear_baseline = self.checkbox_baseline.isChecked()
        overlay_plotting = self.checkbox_overlay_ploting.isChecked()
        # print("using_notch",use_notch)
        # print("clear_baseline", clear_baseline)
        # print("overlay_plotting", overlay_plotting)
        # print(file_path)
        # print("mostrando ")
        if overlay_plotting:
            self.graficar_captura_sobrepuesta(user,letter,capture_number,apply_notch=use_notch,clear_baseline=clear_baseline,file_path=file_path)
        else:
            self.graficar_captura(user,letter,capture_number,apply_notch=use_notch,clear_baseline=clear_baseline,file_path=file_path)

    def close_window(self):
        
        self.close()
        plt.close('all')
    def graficar_captura(self,user, letter, capture_number, apply_notch=False, clear_baseline=False,file_path=None):
        """
        Grafica una captura específica del usuario
        
        Args:
            user: Nombre del usuario
            letter: Letra o número a graficar
            capture_number: Número de captura
            apply_notch: Si es True, aplica filtro notch a 50 Hz para eliminar ruido de línea
            clear_baseline: Si es True, elimina la línea base de las señales
        """
        try:
            plt.style.use('dark_background')
            # Intentar Letter primero
            filename = f'captures/{user}/Letters/{user}_{letter}_{capture_number}.csv'
            
            # Si no existe, intentar Numbers
            if not os.path.exists(filename):
                filename = f'captures/{user}/Numbers/{user}_{letter}_{capture_number}.csv'
            
            # Si aún no existe, intentar Controls
            if not os.path.exists(filename):
                filename = f'captures/{user}/Controls/{user}_{letter}_{capture_number}.csv'
            
            if not os.path.exists(filename):
                print(f"❌ Archivo no encontrado: {filename}")
                return False
            if file_path is not None:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(filename)

            # Mostrar información del archivo
            # print(f"\n✅ Cargado: {filename}")
            # print(f"   Forma del dataset: {df.shape}")
            # print(f"   Canales disponibles: {df.columns.tolist()}")
            if apply_notch:
                print(f"   ✓ Filtro notch aplicado (50 Hz)")
            if clear_baseline:
                print(f"   ✓ Eliminación de línea base aplicada")

            # Crear gráficas por canal
            fig, axes = plt.subplots(4, 2, figsize=(10, 6))
            fig.suptitle(f'Señales EEG - {user} {letter} #{capture_number} { "con notch" if apply_notch else "sinotch"} {"sin linea base" if clear_baseline else ""}', fontsize=10)

            # Aplanar el array de axes para ite
            # ar fácilmente
            axes = axes.flatten()

            df['Tm'] = [i for i in range(len(df))]
            # Plotear cada canal con Time en el eje X
            channels = [col for col in df.columns if col != 'Tm']
            for idx, column in enumerate(channels):
                signal_data = df[column].values.copy()
                
                # Aplicar eliminación de línea base
                if clear_baseline:
                    baseline = np.mean(signal_data[:int(len(signal_data) * 0.1)])
                    signal_data = signal_data - baseline
                
                # Aplicar filtro notch a 50 Hz
                if apply_notch:
                    fs = 250  # Frecuencia de muestreo (ajustar según tus datos)
                    freq_notch = 50  # Frecuencia de línea (50 Hz)
                    quality = 30  # Factor de calidad
                    b, a = signal.iirnotch(freq_notch, quality, fs)
                    signal_data = signal.filtfilt(b, a, signal_data)
                
                axes[idx].plot(df['Tm'], signal_data, linewidth=0.8, color='yellow', label=column)
                #axes[idx].set_title(f'Canal: {column}')
                #axes[idx].set_xlabel('Tiempo (muestra)')
                axes[idx]
                axes[idx].set_ylabel('Amplitud')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend(loc="upper right", fontsize=8)


            plt.tight_layout()
            #plt.legend() 
            plt.show(block=False)  # No bloquear para permitir más ventanas

            return True
            
        except Exception as e:
            print(f"❌ Error al procesar el archivo: {e}")
            return False

    def graficar_captura_sobrepuesta(self,user, letter, capture_number, apply_notch=False, clear_baseline=False,file_path=None):
        """
        Grafica una captura específica del usuario en una nueva ventana sin bloquear.
        
        Args:
            user: Nombre del usuario
            letter: Letra o número a graficar
            capture_number: Número de captura
            apply_notch: Si es True, aplica filtro notch a 50 Hz para eliminar ruido de línea
            clear_baseline: Si es True, elimina la línea base de las señales
        """
        try:
            # Activar el modo interactivo de matplotlib para evitar bloqueos globales
            plt.ion() 
            
            # 1. Intentar Letter primero
            filename = f'captures/{user}/Letters/{user}_{letter}_{capture_number}.csv'
            
            # Si no existe, intentar Numbers
            if not os.path.exists(filename):
                filename = f'captures/{user}/Numbers/{user}_{letter}_{capture_number}.csv'
            
            # Si aún no existe, intentar Controls
            if not os.path.exists(filename):
                filename = f'captures/{user}/Controls/{user}_{letter}_{capture_number}.csv'
            
            if not os.path.exists(filename):
                print(f"❌ Archivo no encontrado: {filename}")
                return False
            
            if file_path is not None:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(filename)
            #print(df.head())
            # Mostrar información del archivo
            # print(f"\n✅ Cargado: {filename}")
            # print(f"   Forma del dataset: {df.shape}")
            # print(f"   Canales disponibles: {df.columns.tolist()}")
            if apply_notch:
                print(f"   ✓ Filtro notch aplicado (50 Hz)")
            if clear_baseline:
                print(f"   ✓ Eliminación de línea base aplicada")

            channels = [col for col in df.columns if col != 'Tm']
            print(channels)
            
            color_array = ['brown','orange','yellow','green','blue','purple','gray','white'] * 2
            plt.style.use('dark_background')
            
            # =========================================================================
            # CRUCIAL: Crear una NUEVA ventana (Figura) única para esta llamada
            # =========================================================================
            fig = plt.figure(figsize=(11, 6)) 
            fig.canvas.manager.set_window_title(f'Señales EEG - {user} {letter} #{capture_number} { "con notch" if apply_notch else ""} {"sin linea base" if clear_baseline else ""}')
            
            # Graficar los canales en la figura actual
            for idx, column in enumerate(channels):
                signal_data = df[column].values.copy()
                # Aplicar filtro notch a 50 Hz
                fs = 250  # Frecuencia de muestreo (ajustar según tus datos)
                if apply_notch:
                    freq_notch = 60  # Frecuencia de línea (50 Hz)
                    quality = 30  # Factor de calidad
                    b, a = signal.iirnotch(freq_notch, quality, fs)
                    signal_data = signal.filtfilt(b, a, signal_data)
                
                # Aplicar eliminación de línea base
                if clear_baseline:
                    n_muestras_baseline = int(0.5 * fs) # 0.5 segundos * 250 Hz = 125 muestras
                    # Calculamos el promedio de esas primeras 125 muestras en el arreglo 1D
                    baseline_mean = np.mean(signal_data[:n_muestras_baseline])
                    # Restamos el promedio a toda la señal
                    signal_data = signal_data - baseline_mean
                
                
                plt.plot(df['Tm'], signal_data, linewidth=0.8, color=color_array[idx], label=column)
            
            plt.title(f'Señales EEG - {user} {letter} #{capture_number} { "con notch" if apply_notch else ""} {"sin linea base" if clear_baseline else ""}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # =========================================================================
            # CRUCIAL: Dibujar la ventana sin bloquear la ejecución
            # =========================================================================
            plt.draw()             # Fuerza el renderizado de la nueva figura
            plt.pause(0.1)         # Pequeña pausa necesaria para que el backend de GUI procese el evento
            
            return True
        except Exception as e:
            print(f"❌ Error al procesar el archivo: {e}")
            return False
# def main():
#     app = QApplication(sys.argv)
#     window = RecordingShowedWindow()
#     window.show()
#     sys.exit(app.exec())
# main()