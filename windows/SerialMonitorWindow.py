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
from scipy.signal import butter, filtfilt, iirnotch

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class EEGFilter:
    def __init__(self, fs=250, low=0.5, high=40, notch_freq=60, use_bandpass=False, use_notch=True):
        self.fs = fs
        self.use_notch = use_notch
        self.use_bandpass = use_bandpass
        
        # Filtro Notch (Eliminación de línea de potencia)
        if self.use_notch:
            self.bn, self.an = iirnotch(notch_freq, Q=15, fs=fs)
        
        # Filtro Bandpass (Pasa-bandas)
        if self.use_bandpass:
            self.b, self.a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')

    def apply(self, signal):
        """
        Aplica el filtro a un arreglo 1D completo (historial del canal).
        Compatible con PlottingThread y RecordingThread.
        """
        if len(signal) < 10:
            return signal
        
        # Asegurar que es un array de numpy listo para operar
        filtered = np.array(signal, dtype=np.float32)
        
        # EVITAR BUG DE NaNs: Si hay cables sueltos o errores de lectura, filtfilt se rompe.
        # Reemplazamos temporalmente los NaNs con 0.0 para proteger el filtro.
        nans = np.isnan(filtered)
        if np.any(nans):
            filtered[nans] = 0.0
        
        try:
            # 1. Aplicar bandpass primero (si está habilitado)
            if self.use_bandpass:
                filtered = filtfilt(self.b, self.a, filtered)
            
            # 2. Aplicar notch (Elimina los 50/60 Hz de la red eléctrica)
            if self.use_notch:
                filtered = filtfilt(self.bn, self.an, filtered)
                
        except Exception as e:
            print(f"Error interno en el filtro: {e}")
            
        # Restauramos los NaNs originales para que pyqtgraph sepa que ahí faltó un dato
        if np.any(nans):
            filtered[nans] = np.nan
            
        return filtered
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
        self.reading_registers = False
        self.registers={}

    def printRegisters(self):
        # self.ser.write(b'r')
        # return
        if self.ser is not None and self.ser.is_open:
            try:
                self.reading_registers = True
                print("\n--- Solicitando Registros ---")
                
                # PASO 1: Detener la transmisión de datos (Comando 's' o 'STOP')
                # Esto es vital para que el ADS deje de escupir números
                self.ser.write(b's') 
                time.sleep(0.1)
                
                # PASO 2: Limpiar TODO lo que quedó en el cable
                self.ser.reset_input_buffer()
                
                # PASO 3: Ahora sí, pedir registros
                self.ser.write(b'r')
                
                # PASO 4: Esperar a que el Arduino termine de escribir
                time.sleep(0.2) 
                
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
                    registers = line.split(",")
                    reg_name = registers[0]
                    reg_addr = registers[1]
                    reg_value = registers[2]
                    binary_value = bin(int(reg_value, 16))[2:].zfill(8)
                    self.registers[reg_name] = (reg_addr, reg_value, binary_value)
                    #print(f"{reg_name} ({reg_addr}): {reg_value}, bin: {binary_value}")
                    # if line:
                    #     print(line)

                # PASO 5: Volver a activar el flujo de datos ('x' o 'START')
                self.ser.write(b'x')
                
                self.reading_registers = False
            except Exception as e:
                print(f"Error: {e}")
                self.reading_registers = False
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
             # --- Variables para medir muestras por segundo ---
            sample_count = 0
            start_time = time.time()
            # ------------------------------------------------

            while self.running:
                if not self.reading_registers:
                
                    if self.ser.in_waiting > 0:
                        raw = self.ser.readline()
                        line = raw.decode('utf-8', errors='replace').strip()
                        if line.endswith(','):
                            line = line[:-1]
                        values = line.split(",")
                        if len(values) == (9 if not self.sixteen_mode else 17):
                            self.data_queue.put(values)
                            sample_count += 1
                        # elif len(values) == 11:
                        #     registers = values
                        #     reg_name = registers[0]
                        #     reg_addr = registers[1]
                        #     reg_value = registers[2]
                        #     binary_value = bin(int(reg_value, 16))[2:].zfill(8)
                        #     self.registers[reg_name] = (reg_addr, reg_value, binary_value)
                        #     print(f"{reg_name} ({reg_addr}): {reg_value}, bin: {binary_value}")
                else:
                    #time.sleep(0.2)  # Evitar un bucle muy rápido mientras se leen registros
                    time.sleep(0.2)  # Evitar un bucle muy rápido mientras se leen registros


                # # if self.ser.in_waiting:
                # #         buffer += self.ser.read(self.ser.in_waiting)

                # #     # Procesar frames completos
                # #     while len(buffer) >= frame_size:
                # #         # Buscar header
                # #         if buffer[0] != 0xA5:
                # #             buffer.pop(0)
                # #             continue

                # #         # Verificar footer
                # #         if buffer[frame_size - 1] != 0x5A:
                # #             buffer.pop(0)
                # #             continue

                # #         frame = buffer[:frame_size]
                # #         buffer = buffer[frame_size:]

                # #         # -------- Decodificación --------
                # #         sample = int.from_bytes(frame[1:5], 'little')

                # #         channels = []
                # #         idx = 5
                # #         for _ in range(N):
                # #             raw = frame[idx:idx+3]

                # #             # convertir 24-bit signed
                # #             val = int.from_bytes(raw, byteorder='big', signed=True)
                # #             channels.append(val)

                # #             idx += 3

                # #         # Enviar a tu queue
                # #         # Convertir a strings como antes (igual que readline)
                # #         values = [str(sample)] + [str(ch) for ch in channels]

                # #         self.data_queue.put(values)
    
                    #print(f"Received: {line}")
                    # current_time = time.time()
                    # if current_time - start_time >= 1.0:
                    #     #print(f"Muestras por segundo: {sample_count}")
                    #     sample_count = 0
                    #     start_time = current_time
                    
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
    recording_finished = pyqtSignal()  # Emitida cuando la grabación termina
    
    def __init__(self):
        super().__init__()
        self.recording_queue = queue.Queue()
        self.running=False
        self.recording=False
        self.recording_df=None
        self.daemon=True
        # Unificar: 50 Hz para línea de potencia (Europa/Sudamérica)
        # Si estás en USA, cambiar a notch_freq=60
        self.filter = EEGFilter(fs=250, low=0.5, high=40, notch_freq=60, use_bandpass=False, use_notch=True)

    def start_recording(self,columns_df,duration):
        print(f"Empezando a grabar por {duration} segundos...")
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
        #print("Empezando grabacion hilo recording columnas",columns_df)
        self.start()

    def stop_recording(self):
        self.recording=False
        self.running=False
        self.quit()
    def get_recorded_data(self):
        V_REF = 4.5
        GAIN = 24  # Ajusta esto si usas otra ganancia en el ADS1299
        LSB_UNIT = V_REF / (GAIN * (2**23 - 1))
        df_microvolts = pd.DataFrame(columns=self.recording_df.columns)
        # ... dentro de tu hilo ...
        for idx, ch in enumerate(self.recording_df.columns):
            # 1. Obtener valores crudos
            raw_values = self.recording_df[ch].values
            values = np.array(raw_values, dtype=np.float32)

            # 2. Convertir a Microvoltios (uV) antes de filtrar
            # Aplicamos la fórmula: count * LSB_UNIT * 1e6
            values = values * LSB_UNIT * 1000000 
            df_microvolts[ch] = values
        return df_microvolts
    def get_filtered_recording_df(self):
        if self.recording_df is None:
            return None
        if self.recording_df.empty:
            return self.recording_df

        df_filtered = self.recording_df.copy()
        channel_columns = [c for c in df_filtered.columns if c.lower() != 'tm']

        # # # for ch in channel_columns:
        # # #     try:
        # # #         values = df_filtered[ch].astype(np.float32).values
        # # #         filtered_values = self.filter.apply(values)
        # # #         df_filtered[ch] = filtered_values
        # # #     except Exception:
        # # #         pass
        V_REF = 4.5
        GAIN = 24  # Ajusta esto si usas otra ganancia en el ADS1299
        LSB_UNIT = V_REF / (GAIN * (2**23 - 1))

        # ... dentro de tu hilo ...
        for idx, ch in enumerate(channel_columns):
            # 1. Obtener valores crudos
            raw_values = df_filtered[ch].values
            values = np.array(raw_values, dtype=np.float32)

            # 2. Convertir a Microvoltios (uV) antes de filtrar
            # Aplicamos la fórmula: count * LSB_UNIT * 1e6
            values = values * LSB_UNIT * 1000000 

            # 3. Aplicar filtros (ahora sobre valores reales)
            print(f"Aplicando filtro al canal {ch}...")
            values_filt = self.filter.apply(values)
            # print("raw",values[:20])
            # print("filtered",values_filt[:20])
            
            df_filtered[ch] = values_filt
        return df_filtered

    def run(self):
        self.running=True
        while self.running:
            if time.time() - self.start_time >= self.duration:
                # Cuando se alcanza el tiempo, esperar a vaciar la cola (max 500ms)
                deadline = time.time() + 0.4
                while time.time() < deadline:
                    try:
                        values = self.recording_queue.get(timeout=0.05)
                        if self.recording:
                            try:
                                row = [float(v.strip()) if isinstance(v,str) else float(v) for v in values]
                                if len(row) == len(self.columns_df):
                                    self.recording_df.loc[len(self.recording_df)] = row
                            except (ValueError, TypeError):
                                pass
                    except queue.Empty:
                        break
                self.recording = False
                self.running = False
                self.recording_finished.emit()  # Señal de que grabación terminó
                break
            try:
                values = self.recording_queue.get(timeout=0.1)
                if self.recording:
                    try:
                        row = [float(v.strip()) if isinstance(v,str) else float(v) for v in values]
                        if len(row) == len(self.columns_df):
                            self.recording_df.loc[len(self.recording_df)] = row
                    except (ValueError, TypeError):
                        pass
            except queue.Empty:
                continue
    def stop(self):
        self.running=False
class PlottingThread(QThread):
    update_plots = pyqtSignal(list,list,list)
    def __init__(self,data_buffer,buffer_lock,plots ,channels,apply_filter):
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
        # Unificar con RecordingThread: 50 Hz, sin bandpass por ahora
        self.filter = EEGFilter(fs=250, low=0.1, high=50, notch_freq=60, use_bandpass=False, use_notch=True)
        self.applied_filter = apply_filter
        
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
                    
                    # # for idx  in range(len((self.channels))):
                    # #     values = [float(d[idx+1]) if d[idx+1].strip() else np.nan for d in data_tuples if d]
                       
                    # #     if self.applied_filter:
                    # #         values = np.array(values, dtype=np.float32)
                    # #         values = self.filter.apply(values)
                    # #         all_values.append(values.tolist())
                    # #     else:
                    # #         all_values.append(values)

                    # Definir constantes antes del bucle o en el __init__
                    V_REF = 4.5
                    GAIN = 24  # Ajusta esto si usas otra ganancia en el ADS1299
                    LSB_UNIT = V_REF / (GAIN * (2**23 - 1))

                    ### con conversion a microvoltios y filtrado aplicado sobre valores reales
                    for idx in range(len(self.channels)):
                        # 1. Obtener valores crudos
                        raw_values = [float(d[idx+1]) if d[idx+1].strip() else np.nan for d in data_tuples if d]
                        values = np.array(raw_values, dtype=np.float32)

                        # 2. Convertir a Microvoltios (uV) antes de filtrar
                        # Aplicamos la fórmula: count * LSB_UNIT * 1e6
                        values = values * LSB_UNIT * 1000000

                        # 3. Aplicar filtros (ahora sobre valores reales)
                        if self.applied_filter:
                            values = self.filter.apply(values)
                            all_values.append(values.tolist())
                        else:
                            all_values.append(values.tolist())

                    if len(times) > 0 and len(all_values) >0:
                        self.update_plots.emit(times,all_values,self.channels)

            except Exception as e:
                pass
                #print(f"Error plottinh thread {e}")
            time.sleep(.5)
    def stop(self):
        self.running=False
        self.quit()
    def set_apply_filter(self, apply_filter):
        self.applied_filter = apply_filter
class SignalsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        self.data_queue = queue.Queue()
        self.columnasTest = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        self.columnasReal = ["Tm","Oz","Po7","Po4","Po3","P4","P3","Po8","Pz","Fz","F2","F3","F4","AF3","Cz","AF4","F1"]
        self.df_eight = pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"])


        self.columns = self.columnasReal if True else self.columnasTests
        self.eight_channels =["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.sixteen_channels = self.columns[1:]
        self.df_sixteen= pd.DataFrame(columns=self.columns)
        self.apply_filter = False
        self.overlay_mode = False
        #  ["Oz","Po7","Po4","Po3","P4","P3","Po","Pz","Fz","F2","F3","F4,"AF3","Cz","AF4","F1"]
        self.serial_thread = None
        self.port = 'COM5'
        self.baudrate = 330400 *2
        self.test_mode = False  # Flag for test mode
        self.sixteen_channels_mode=True
        self.channels = self.sixteen_channels if self.sixteen_channels_mode else self.eight_channels
        self.df = self.df_sixteen if self.sixteen_channels_mode else self.df_eight
        self.setup_ui()
        self.data_buffer =deque(maxlen=1200)
        self.buffer_lock = threading.Lock()
        self.recording_thread = RecordingThread()
        self.legend = None
        #self.recording_thread.finished_record.connect(self.on_recording_finished)
        self.plotting_thread = None
        self.colores_canales = ['brown','orange','yellow','g','b','purple','gray','white','brown','orange','yellow','g','b','purple','gray','white']

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
        self.channels_checkbox.setChecked(self.sixteen_channels_mode)
        self.channels_checkbox.stateChanged.connect(self.toggle_channels)
        self.overlay_checkbox = QCheckBox("Superponer canales")
        self.overlay_checkbox.setChecked(self.overlay_mode)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        self.recording = False
        self.filter_checkbox = QCheckBox("Aplicar Filtro")
        self.filter_checkbox.setChecked(self.apply_filter)
        self.filter_checkbox.stateChanged.connect(self.set_apply_filter)
        # self.button_print_registers = QPushButton("Print Registers")
        # self.button_print_registers.clicked.connect(self.check_registers)
        self.button_check_loff_statp = QPushButton("Check LOFF_STATP")
        self.button_check_loff_statp.clicked.connect(self.checkRegisterLOFF_STATP)
        # Plot setup
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots = []
        self._rebuild_plots()
        # Scrolling setup
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
        control_layout.addWidget(self.overlay_checkbox)
        control_layout.addWidget(self.test_checkbox)
        control_layout.addWidget(self.filter_checkbox)
        #control_layout.addWidget(self.button_print_registers)
        control_layout.addWidget(self.button_check_loff_statp)
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

    def toggle_channels(self, state):
        sixteen_mode = state == 2
        if sixteen_mode == self.sixteen_channels_mode:
            return
        was_running = self.serial_thread is not None and self.serial_thread.isRunning()
        if was_running:
            self.stop_serial()
        self.set_channel_mode(sixteen_mode)
        if was_running:
            self.start_serial()

    def set_channel_mode(self, sixteen_mode: bool):
        self.sixteen_channels_mode = sixteen_mode
        self.channels = self.sixteen_channels if self.sixteen_channels_mode else self.eight_channels
        self.df = self.df_sixteen if self.sixteen_channels_mode else self.df_eight
        self._rebuild_plots()
    def toggle_overlay(self, state):
        self.overlay_mode = state == 2
        self._rebuild_plots()
    def set_apply_filter(self, state):
        self.apply_filter = state == 2
        if self.plotting_thread:
            self.plotting_thread.set_apply_filter(self.apply_filter)


    def _rebuild_plots(self):
        self.plots = []
        self.legend = None
        self.plot_widget.clear()
        if self.overlay_mode:
            plot = self.plot_widget.addPlot(row=0, col=0, rowspan=1, colspan=1, title='Canales superpuestos')
            plot.setLabel('left', 'Amplitud')
            plot.setLabel('bottom', 'Tiempo')
            plot.showGrid(x=True, y=True, alpha=0.3)
            self.legend = plot.addLegend(offset=(10, 10))
            self.plots.append(plot)
            self.plot_widget.setFixedHeight(600)
            return

        for i, channel in enumerate(self.channels):
            plot = self.plot_widget.addPlot(row=i, col=0, rowspan=1, colspan=1, title=f'Canal: {channel}')
            plot.setLabel('left', 'Amplitud')
            plot.setLabel('bottom', 'Tiempo')
            self.plots.append(plot)
        self.plot_widget.setFixedHeight(2200 if not self.sixteen_channels_mode else 3200)

    def on_plot_update(self,times,all_values,channels):
        try:
            if self.overlay_mode and len(self.plots) > 0:
                plot = self.plots[0]
                plot.clear()
                if self.legend is not None:
                    self.legend.clear()
                for idx, values in enumerate(all_values):
                    if len(times) > 0 and len(values) > 0:
                        pen = pg.mkPen(color=pg.intColor(idx, hues=max(3, len(all_values))), width=1)
                        plot.plot(times, values, pen=self.colores_canales[idx], name=str(channels[idx]))
                return

            for idx, plot in enumerate(self.plots):
                plot.clear()
                if idx < len(all_values):
                    values = all_values[idx]
                    if len(times) > 0 and len(values) > 0:
                        plot.plot(times, values, pen='b')
        except Exception as e:
            print(f"Erorr updating plot {e}")
    def start_serial(self):
        if self.test_mode:
            print("Starting test mode with CSV data.")
            self.serial_thread = CSVReader('datosLectura16.csv', self.data_queue,self.sixteen_channels_mode)
        else:
            self.port = self.port
            self.baudrate = self.baudrate
            print(f"Starting serial on {self.port} at {self.baudrate} baud.")
            self.serial_thread = SerialReader(self.port, self.baudrate, self.data_queue,self.sixteen_channels_mode)
            self.serial_thread.error_signal.connect(self.on_serial_error)
        self.serial_thread.start()
        self.plotting_thread = PlottingThread(self.data_buffer,self.buffer_lock,self.plots,self.channels,self.apply_filter)
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
        self.recording_thread.stop_recording()
        if self.apply_filter:
            return self.recording_thread.get_filtered_recording_df()
        else:
            return self.recording_thread.get_recorded_data()

    def start_recording(self, duration=2):
        """
        Captura datos que se lean sin afectar visualizacion retorna el df .
        """
        
        #sixteen_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        sixteen_columns = self.columns
        eigth_columns = ["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.recording_thread = RecordingThread()
        self.recording_thread.start_recording(columns_df = (sixteen_columns if self.sixteen_channels_mode else eigth_columns),duration=duration)   

    def return_recorded_data(self):
        if self.apply_filter:
            return self.recording_thread.get_filtered_recording_df()
        else:            
            return self.recording_thread.get_recorded_data()
    def check_registers(self,print_output=True):
        if self.serial_thread :
            self.serial_thread.printRegisters()
        if not print_output:
            return
        QTimer().singleShot(600, self.printRegisters)  # Esperar medio segundo antes de revisar el registro
    def printRegisters(self):
        self.registers = self.serial_thread.registers
        print("\n--- Registros ADS1299 ---")
        for reg_name, (reg_addr, reg_value, binary_value) in self.registers.items():
            print(f"{reg_name} ({reg_addr}): {reg_value}, bin: {binary_value}")
    def checkRegisterLOFF_STATP(self):
        self.check_registers(print_output=False)
        QTimer().singleShot(600, self.print_disconnected_channels)  # Esperar medio segundo antes de revisar el registro
    def print_disconnected_channels(self):
        self.registers = self.serial_thread.registers
        loff_statp = self.registers.get("LOFF_STATP")
        if loff_statp is None:
            print("No se pudo obtener el registro LOFF_STATP.")
            return
        loff_statp = loff_statp[2] 
        print(f"LOFF_STATP: {loff_statp}")
        status = [int(bit) for bit in loff_statp]
        status = np.array(status)
        status = status[::-1]  # Invertir para que el bit 0 esté a la derecha
        canales_np_8= np.array(self.channels[:8])
        
        disconnected_channels = canales_np_8[status == 1]
        if len(disconnected_channels) > 0:
            print(f"Canales desconectados: {', '.join(disconnected_channels)}")
        else:
            print("Todos los canales están conectados correctamente.")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalsWindow()
    window.show()
    sys.exit(app.exec())