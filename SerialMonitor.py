import sys
import serial
import time
import threading
import queue
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QCheckBox,QScrollArea
from PyQt6.QtCore import QTimer,Qt
import pyqtgraph as pg
from testDataReader import CSVReader  # Import for test mode

class SerialReader(threading.Thread):
    def __init__(self, port, baudrate, data_queue,sixteen_mode):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self.sixteen_mode = sixteen_mode
        self.running = False
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flush()
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except:
                pass
            time.sleep(0.05)
            while self.ser.in_waiting:
                _ = self.ser.readline()
            self.ser.write(b'x')
            self.running = True
            while self.running:
                if self.ser.in_waiting > 0:
                    raw = self.ser.readline()
                    line = raw.decode('utf-8', errors='replace').strip()
                    if line.endswith(','):
                        line = line[:-1]
                    values = line.split(",")
                    if len(values) == (9 if not self.sixteen_mode else 17):
                        self.data_queue.put(values)
                    #print(f"Received: {line}")
                    
        except Exception as e:
            print(f"Serial error: {e}")
        finally:
            if self.ser:
                self.ser.close()

    def stop(self):
        self.ser.write(b's')
        self.running = False

class SignalsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        self.data_queue = queue.Queue()
        self.df_eight = pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"])
        self.eight_channels =["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.sixteen_channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        self.df_sixteen= pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"])
        self.serial_thread = None
        self.port = 'COM5'
        self.baudrate = 112500
        self.test_mode = True  # Flag for test mode
        self.sixteen_channels_mode=True
        self.channels = self.sixteen_channels if self.sixteen_channels_mode else self.eight_channels
        self.df = self.df_sixteen if self.sixteen_channels_mode else self.df_eight
        self.setup_ui()

    def setup_ui(self):
        # UI Elements
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_serial)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_serial)
        self.stop_button.setEnabled(False)
        self.test_checkbox = QCheckBox("Modo Prueba (CSV)")
        self.test_checkbox.stateChanged.connect(self.toggle_test_mode)
        self.channels_checkbox = QCheckBox("16 canales")
        self.channels_checkbox.stateChanged.connect(self.toggle_channels)
        self.recording = False

        # Plot setup
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots = []
        for i, channel in enumerate(self.channels):
            #plot = self.plot_widget.addPlot(row=i//2, col=i%2, title=f'Canal: {channel}',)
            plot = self.plot_widget.addPlot(row=i, col=0,rowspan=1, colspan=1, title=f'Canal: {channel}')
            plot.setLabel('left', 'Amplitud')
            plot.setLabel('bottom', 'Tiempo')
            self.plots.append(plot)
        self.plot_widget.setFixedHeight(2500 if not self.sixteen_channels_mode else 3200)
        #Scrolling setup
        self.scrolling_area = QScrollArea()
        self.scrolling_area.setWidgetResizable(True)
        self.scrolling_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrolling_area.setWidget(self.plot_widget)
        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Port:"))
        self.label_port = QLabel(self.port)
        control_layout.addWidget(self.label_port)
        control_layout.addWidget(QLabel("Baudrate:"))
        self.label_baudrate = QLabel(str(self.baudrate))
        control_layout.addWidget(self.label_baudrate)
        control_layout.addWidget(self.channels_checkbox)
        control_layout.addWidget(self.test_checkbox)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.scrolling_area)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer for updating plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1200)  # Update every second

    def toggle_test_mode(self, state):
        self.test_mode = state == 2  # Checked
    def toggle_channels(self,state):
        print("16 canales")
        self.sixteen_channels_mode = state== 2
    def start_serial(self):
        if self.test_mode:
            print("Starting test mode with CSV data.")
            self.serial_thread = CSVReader('datosLectura.csv', self.data_queue,self.sixteen_channels_mode)
        else:
            self.port = self.port
            self.baudrate = self.baudrate
            print(f"Starting serial on {self.port} at {self.baudrate} baud.")
            self.serial_thread = SerialReader(self.port, self.baudrate, self.data_queue,self.sixteen_channels_mode)
        
        self.serial_thread.start()  # Start the thread properly to avoid blocking the GUI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.join()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_plot(self,row=None):
        #print("Updating plot")
        # print(self.data_queue.qsize())

        while not self.data_queue.empty():
            values = self.data_queue.get()
            row = []
            for v in values:
                try:
                    row.append(float(v.strip()))
                except ValueError:
                    row.append(np.nan)
            if len(row) == (17 if self.sixteen_channels_mode else 9) :
                self.df.loc[len(self.df)] = row
            
        if not self.df.empty:
            df_plot = self.df.tail(500)
            print(df_plot['ch1'].iloc[-1])
            for idx, plot in enumerate(self.plots):

                channel = self.channels[idx]
                plot.clear()
                plot.plot(df_plot['Tm'].values, df_plot[channel].values, pen='b')

    def update_serial_config(self, port, baudrate):
        """Actualiza la configuracion del puerto serial"""
        self.port = port
        self.baudrate = baudrate
        self.label_port.setText(self.port)
        self.label_baudrate.setText(str(self.baudrate))
        if self.serial_thread and self.serial_thread.is_alive():
            self.stop_serial()
            self.start_serial()

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()

    def start_recording(self, duration=2):
        """
        Captura datos que se lean sin afectar visualizacion retorna el df .
        """
        print(f"Empezando a grabar por {duration} segundos...")
        
        # Cola TEMPORAL solo para esta grabación (NO toca la cola principal)
        self.recording = True
        sixteen_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        eigth_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        recording_df = pd.DataFrame( columns = (sixteen_columns if self.sixteen_channels_mode else eigth_columns))
        sample_count = 0
        start_time = time.time()
        self.recording =True        
        while time.time() - start_time < duration:
            try:
                # COPIA los datos de la cola principal SIN consumirla
                #print("AVER AQUU",self.data_queue.qsize())
                if not self.data_queue.empty():
                    values = self.data_queue.get_nowait()  
                    #print("otravezzz",self.data_queue.qsize())
                    # Procesa para grabación
                    row = [float(v.strip()) if v.strip() else np.nan for v in values]
                    if len(row) == (9 if not self.sixteen_channels_mode else 17):
                        recording_df.loc[len(recording_df)] = row
                        self.df.loc[len(self.df)] = row
                        sample_count += 1
                self.update_plot()

                time.sleep(0.001)
                
            except queue.Empty:
                continue
        
        print(f"Grabación completada. {sample_count} muestras.")
        return recording_df



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalsWindow()
    window.show()
    sys.exit(app.exec())