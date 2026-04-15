import sys
import serial
import time
import threading
import queue
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QCheckBox,QScrollArea
from PyQt6.QtCore import QTimer,Qt,QThread,pyqtSignal
import pyqtgraph as pg
from utils.testDataReader import CSVReader  # Import for test mode
from collections import deque
class SerialReader(QThread):
    error_signal = pyqtSignal(str)
    def __init__(self, port, baudrate, data_queue,sixteen_mode):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self.sixteen_mode = sixteen_mode
        self.running = False
        self.ser = None

    def run(self):
        last_n=1
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.flush()
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except:
                pass
            time.sleep(0.04)
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
            self.error_signal.emit(f"No se pudo abrir el puerto {self.port}. Verifique la conexión y la configuración.")
        finally:
            if self.ser:
                self.ser.close()


                

    def stop(self):
        if self.ser is not None:
            try:
                self.ser.write(b's')
            except Exception:
                pass
        self.running = False
        self.quit()

class RecordingThread(QThread):
    def __init__(self):
        super().__init__()
        self.recording_queue = queue.Queue()
        self.running=False
        self.recording=False
        self.recording_df=None
        self.daemon=True

    def start_recording(self,columns_df,duration):
        self.columns_df=columns_df
        self.duration=duration
        self.recording=True
        self.start_time = time.time()
        self.recording_df=pd.DataFrame(columns=columns_df)
        while not self.recording_queue.empty():
            try:
                self.recording_queue.get_nowait()
            except queue.Empty:
                break
        print("Empezando grabacion hilo recording columnas",columns_df)
        self.start()

    def stop_recording(self):
        self.recording=False
        self.running=False
        self.quit()
        
    def run(self):
        self.running=True
        while self.running:
            if time.time() - self.start_time >= self.duration:
                self.stop_recording()
                break
            try:
                values = self.recording_queue.get(timeout=0.1)
                if self.recording:
                    try:
                        row = [float(v.strip()) if isinstance(v,str) else float(v) for v in values]
                        if len(row) == len(self.columns_df) :
                            self.recording_df.loc[len(self.recording_df)] = row
                    except (ValueError,TypeError):
                        pass
            except queue.Empty:
                continue
    def stop(self):
        self.running=False
class PlottingThread(QThread):
    update_plots = pyqtSignal(list,list,list)
    def __init__(self,data_buffer,buffer_lock,plots ,channels):
        super().__init__()
        self.data_buffer=data_buffer
        self.buffer_lock =  buffer_lock
        self.plots=plots
        self.channels=channels    
        self.running=False
        self.daemon=True
        self.treshold_red = -40000
        self.treshold_blue=40000
        self.treshold=0
        
    def run(self):
        self.running=True
        while self.running :
            try:
                with self.buffer_lock:
                    if len(self.data_buffer)>0:
                        buffer_copy = list(self.data_buffer)
                    else:
                        buffer_copy =[]


                if len(buffer_copy) >0:
                    data_tuples = buffer_copy
                    times = [float (d[0]) for d in data_tuples if d]
                    all_values =[]
                    
                    for idx  in range(len((self.channels))):
                        values = [float(d[idx+1]) if d[idx+1].strip() else np.nan for d in data_tuples if d]
                        all_values.append(values)

                    if len(times) > 0 and len(all_values) >0:
                        self.update_plots.emit(times,all_values,self.channels)

            except Exception as e:
                pass
                #print(f"Error plottinh thread {e}")
            time.sleep(.7)
    def stop(self):
        self.running=False
        self.quit()
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
        self.baudrate = 330400
        self.test_mode = False  # Flag for test mode
        self.sixteen_channels_mode=True
        self.channels = self.sixteen_channels if self.sixteen_channels_mode else self.eight_channels
        self.df = self.df_sixteen if self.sixteen_channels_mode else self.df_eight
        self.setup_ui()
        self.data_buffer =deque(maxlen=800)
        self.buffer_lock = threading.Lock()
        self.recording_thread = RecordingThread()
        #self.recording_thread.finished_record.connect(self.on_recording_finished)
        self.plotting_thread = None

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
        self.plot_widget.setFixedHeight(2200 if not self.sixteen_channels_mode else 3200)
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
        #control_layout.addWidget(self.channels_checkbox)
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
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start(1000)  # Update every second

    def toggle_test_mode(self, state):
        self.test_mode = state == 2  # Checked
    def toggle_channels(self,state):
        print("16 canales")
        self.sixteen_channels_mode = state== 2


    def on_plot_update(self,times,all_values,channels):
        try:
            for idx,plot in enumerate(self.plots):
                plot.clear()
                if idx < len(all_values):
                    values = all_values[idx]
                    if len(times)>0 and len(values) >0:
                        plot.plot(times,values,pen='b')
        except Exception as e:
            print(f"Erorr updating plot {e}")
    def start_serial(self):
        if self.test_mode:
            print("Starting test mode with CSV data.")
            self.serial_thread = CSVReader('datosLectura.csv', self.data_queue,self.sixteen_channels_mode)
        else:
            self.port = self.port
            self.baudrate = self.baudrate
            print(f"Starting serial on {self.port} at {self.baudrate} baud.")
            self.serial_thread = SerialReader(self.port, self.baudrate, self.data_queue,self.sixteen_channels_mode)
            self.serial_thread.error_signal.connect(self.on_serial_error)
        self.serial_thread.start()
        self.plotting_thread = PlottingThread(self.data_buffer,self.buffer_lock,self.plots,self.channels)
        self.plotting_thread.update_plots.connect(self.on_plot_update)
        self.plotting_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.timer =QTimer()
        self.timer.timeout.connect(self.process_queue_to_buffer)
        self.timer.start(10)

    def on_serial_error(self, msg):
        QMessageBox.critical(self, "Error de Puerto Serial", msg)


        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.serial_thread:
            self.serial_thread = None
    def stop_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()
        if self.plotting_thread:    
            self.plotting_thread.stop()
            self.plotting_thread.wait()
        if hasattr(self,'timer'):
            self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def process_queue_to_buffer(self):
        
        while not self.data_queue.empty():
            #print("Actualizando BUFFER")
            try:
                values =self.data_queue.get_nowait()
                with self.buffer_lock:
                    self.data_buffer.append(values)

                if self.recording_thread.recording:
                    self.recording_thread.recording_queue.put(values)
            except queue.Empty:
                break

    def update_plot(self,row=None):

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
    def stop_recording(self):
        df = self.recording_thread.stop_recording()
        return df
    def start_recording(self, duration=2):
        """
        Captura datos que se lean sin afectar visualizacion retorna el df .
        """
        print(f"Empezando a grabar por {duration} segundos...")
        sixteen_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        eigth_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.recording_thread.start_recording(columns_df = (sixteen_columns if self.sixteen_channels_mode else eigth_columns),duration=duration)   

    def return_recorded_data(self):
        df =self.recording_thread.recording_df
        return df

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalsWindow()
    window.show()
    sys.exit(app.exec())