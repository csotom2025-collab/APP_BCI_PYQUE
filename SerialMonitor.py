import sys
import serial
import time
import threading
import queue
import pandas as pd
import numpy as np
from collections import deque
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QCheckBox
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from testDataReader import CSVReader  # Import for test mode

class SerialReader(threading.Thread):
    def __init__(self, port, baudrate, data_queue):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
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
                    if len(values) == 9:
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

class RecordingThread(threading.Thread):
    """Hilo dedicado para grabar datos en tiempo real sin interferir con el plotting"""
    def __init__(self, data_buffer, buffer_lock):
        super().__init__()
        self.data_buffer = data_buffer  # Referencia al buffer compartido
        self.buffer_lock = buffer_lock  # Lock para acceso seguro
        self.running = False
        self.recording = False
        self.recording_df = None
        self.channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.daemon = True  # Hilo daemon para que no bloquee la salida

    def start_recording(self):
        """Inicia la grabación"""
        self.recording = True
        self.recording_df = pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"])
        print("Grabación iniciada en hilo separado...")

    def stop_recording(self):
        """Detiene la grabación y retorna los datos"""
        self.recording = False
        print(f"Grabación completada. {len(self.recording_df)} muestras grabadas.")
        return self.recording_df

    def run(self):
        """Ejecuta el hilo de grabación continuamente"""
        self.running = True
        while self.running:
            if self.recording:
                # Acceso seguro al buffer con lock
                with self.buffer_lock:
                    if len(self.data_buffer) > 0:
                        # Lee el último dato sin removerlo del buffer
                        latest_data = self.data_buffer[-1]
                        try:
                            row = [float(v.strip()) if isinstance(v, str) else float(v) for v in latest_data]
                            if len(row) == 9:
                                self.recording_df.loc[len(self.recording_df)] = row
                        except (ValueError, TypeError, AttributeError):
                            pass
            time.sleep(0.001)  # Pequeño delay para no saturar CPU

    def stop(self):
        """Detiene el hilo"""
        self.running = False

class PlottingThread(threading.Thread):
    """Hilo dedicado para actualizar gráficos sin congelar la interfaz"""
    def __init__(self, data_buffer, buffer_lock, plots, channels):
        super().__init__()
        self.data_buffer = data_buffer
        self.buffer_lock = buffer_lock
        self.plots = plots
        self.channels = channels
        self.running = False
        self.daemon = True

    def run(self):
        """Actualiza los gráficos continuamente"""
        self.running = True
        while self.running:
            try:
                # Acceso seguro al buffer
                with self.buffer_lock:
                    if len(self.data_buffer) > 0:
                        # Copia los datos del buffer para procesar
                        buffer_copy = list(self.data_buffer)
                
                # Procesa FUERA del lock para no bloquear otras operaciones
                if len(buffer_copy) > 0:
                    data_tuples = buffer_copy
                    for idx, plot in enumerate(self.plots):
                        try:
                            plot.clear()
                            times = [float(d[0]) for d in data_tuples if d]
                            values = [float(d[idx + 1]) if d[idx + 1].strip() else np.nan 
                                     for d in data_tuples if d]
                            if len(times) > 0 and len(values) > 0:
                                plot.plot(times, values, pen='b')
                        except (ValueError, IndexError, TypeError):
                            pass
            except Exception as e:
                print(f"Error en plotting thread: {e}")
            
            time.sleep(1)  # Actualizar gráficos cada segundo

    def stop(self):
        """Detiene el hilo"""
        self.running = False

class SignalsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Buffer compartido thread-safe y lock
        self.data_buffer = deque(maxlen=500)  # Guardar últimos 500 datos
        self.buffer_lock = threading.Lock()
        
        # Queue para el SerialReader (mantener compatibilidad)
        self.data_queue = queue.Queue()
        self.channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.serial_thread = None
        self.port = 'COM5'
        self.baudrate = 112500
        self.test_mode = False
        
        # Threads de grabación y plotting
        self.recording_thread = RecordingThread(self.data_buffer, self.buffer_lock)
        self.plotting_thread = None
        
        # UI Elements
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_serial)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_serial)
        self.stop_button.setEnabled(False)
        
        self.record_button = QPushButton("Grabar")
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)
        
        self.test_checkbox = QCheckBox("Modo Prueba (CSV)")
        self.test_checkbox.stateChanged.connect(self.toggle_test_mode)

        # Plot setup
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots = []
        for i, channel in enumerate(self.channels):
            plot = self.plot_widget.addPlot(row=i//2, col=i%2, title=f'Canal: {channel}')
            plot.setLabel('left', 'Amplitud')
            plot.setLabel('bottom', 'Tiempo')
            self.plots.append(plot)

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Port:"))
        self.label_port = QLabel(self.port)
        control_layout.addWidget(self.label_port)
        control_layout.addWidget(QLabel("Baudrate:"))
        self.label_baudrate = QLabel(str(self.baudrate))
        control_layout.addWidget(self.label_baudrate)
        control_layout.addWidget(self.test_checkbox)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.record_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.plot_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Iniciar el thread de grabación
        self.recording_thread.start()

    def toggle_test_mode(self, state):
        self.test_mode = state == 2  # Checked

    def start_serial(self):
        if self.test_mode:
            print("Starting test mode with CSV data.")
            self.serial_thread = CSVReader('./captures/MVP_Letters_A.csv', self.data_queue)
        else:
            self.port = self.port
            self.baudrate = self.baudrate
            print(f"Starting serial on {self.port} at {self.baudrate} baud.")
            self.serial_thread = SerialReader(self.port, self.baudrate, self.data_queue)
        
        self.serial_thread.start()
        
        # Iniciar thread de plotting cuando comienza serial
        self.plotting_thread = PlottingThread(self.data_buffer, self.buffer_lock, self.plots, self.channels)
        self.plotting_thread.start()    
        
        # Timer para procesar datos de la queue al buffer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue_to_buffer)
        self.timer.start(10)  # Procesar cada 10ms
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)

    def process_queue_to_buffer(self):
        """Procesa datos de la queue del SerialReader al buffer compartido"""
        while not self.data_queue.empty():
            try:
                values = self.data_queue.get_nowait()
                # Agregar con lock al buffer compartido
                with self.buffer_lock:
                    self.data_buffer.append(values)
            except queue.Empty:
                break

    def stop_serial(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.join(timeout=2)
        
        if self.plotting_thread:
            self.plotting_thread.stop()
            self.plotting_thread.join(timeout=2)
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)

    def start_recording(self):
        """Inicia grabación en el thread dedicado"""
        self.recording_thread.start_recording()

    def stop_recording(self):
        """Detiene la grabación y guarda datos"""
        df = self.recording_thread.stop_recording()
        # Aquí puedes guardar a CSV si lo deseas
        if not df.empty:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            df.to_csv(f"recording_{timestamp}.csv", index=False)
            print(f"Datos guardados en recording_{timestamp}.csv")
        return df

    def update_serial_config(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.label_port.setText(self.port)
        self.label_baudrate.setText(str(self.baudrate))
        if self.serial_thread and self.serial_thread.is_alive():
            self.stop_serial()
            self.start_serial()

    def closeEvent(self, event):
        # Detener grabación si estaba activa
        if self.recording_thread.recording:
            self.stop_recording()
        
        self.stop_serial()
        self.recording_thread.stop()
        event.accept()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalsWindow()
    window.show()
    sys.exit(app.exec())