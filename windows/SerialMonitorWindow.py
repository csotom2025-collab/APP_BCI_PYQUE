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
from scipy.signal import butter, filtfilt, freqz, iirnotch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
import binascii
import hid  # pip install hidapi  (lectura Emotiv EPOC+)
from Crypto.Cipher import AES  # pip install pycryptodome  (decriptado Emotiv)

class EEGFilter:
    def __init__(self, fs=250, low=0.5, high=40, notch_freq=60, use_bandpass=False, use_notch=True):
        self.fs = fs
        self.use_notch = use_notch
        self.use_bandpass = use_bandpass
        
        # Filtro Notch (Eliminación de línea de potencia)
        if self.use_notch:
            self.bn, self.an = iirnotch(notch_freq, Q=15, fs=fs)
        
        # #PLot filtro notch para referencia
        # w, h = freqz(self.bn, self.an, fs=fs)
        # plt.plot(w, 20 * np.log10(abs(h)))
        # plt.title("Respuesta en frecuencia del filtro Notch")
        # plt.xlabel("Frecuencia (Hz)")
        # plt.ylabel("Magnitud (dB)")
        # plt.grid(True)
        # plt.show()
        
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
            # if self.use_bandpass:
            #     filtered = filtfilt(self.b, self.a, filtered)
            
            # 2. Aplicar notch (Elimina los 50/60 Hz de la red eléctrica)
            if self.use_notch:
                filtered = filtfilt(self.bn, self.an, filtered)
                
        except Exception as e:
            print(f"Error interno en el filtro: {e}")
            
        # Restauramos los NaNs originales para que pyqtgraph sepa que ahí faltó un dato
        if np.any(nans):
            filtered[nans] = np.nan
            
        return filtered
# ============================================================================
# SOPORTE EMOTIV EPOC+
# ----------------------------------------------------------------------------
# Todo este bloque permite leer la diadema Emotiv EPOC+ y entregar los datos
# con el MISMO formato que usa SerialReader (filas de texto puestas en
# data_queue), para que el resto de la aplicacion (filtro, superposicion,
# grabado, ploteo) funcione exactamente igual sin importar el origen.
# ============================================================================

EMOTIV_SERIAL_NUMBER = "SN201512233578GM"
EMOTIV_IS_RESEARCH = False
EMOTIV_PACKET_SIZE = 32

# Nombres de los 14 canales EEG que entrega la EPOC+ (orden fijo).
EMOTIV_CHANNELS = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1',
                    'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

_emotiv_sensors_14_bits = {
    'F3': [10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
    'FC5': [28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9],
    'AF3': [46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27],
    'F7': [48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45],
    'T7': [66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63],
    'P7': [84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65],
    'O1': [102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83],
    'O2': [140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121],
    'P8': [158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139],
    'T8': [160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157],
    'F8': [178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175],
    'AF4': [196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177],
    'FC6': [214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195],
    'F4': [216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213],
}

_emotiv_quality_bits = [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

_emotiv_sensor_quality_bit = {
    0: "F3", 64: 'F3', 1: 'FC5', 65: 'FC5', 2: 'AF3', 66: 'AF3',
    3: 'F7', 67: 'F7', 4: 'T7', 68: 'T7', 5: 'P7', 69: 'P7',
    6: 'O1', 70: 'O1', 7: 'O2', 71: 'O2', 8: 'P8', 72: 'P8',
    9: 'T8', 73: 'T8', 10: 'F8', 74: 'F8', 11: 'AF4', 75: 'AF4',
    12: 'FC6', 76: 'FC6', 80: 'FC6', 13: 'F4', 77: 'F4',
    14: 'F8', 78: 'F8', 15: 'AF4', 79: 'AF4',
}

_emotiv_battery_values = {
    255: 100, 254: 100, 253: 100, 252: 100, 251: 100, 250: 100, 249: 100, 248: 100,
    247: 99, 246: 97, 245: 93, 244: 89, 243: 85, 242: 82, 241: 77, 240: 72,
    239: 66, 238: 62, 237: 55, 236: 46, 235: 32, 234: 20, 233: 12, 232: 6,
    231: 4, 230: 3, 229: 2, 228: 2, 227: 2, 226: 1, 225: 0, 224: 0,
}

_emotiv_sensors_mapping = {name: {'value': 0.0, 'quality': 0} for name in EMOTIV_CHANNELS}
_emotiv_sensors_mapping['X'] = {'value': 0.0, 'quality': 0}
_emotiv_sensors_mapping['Y'] = {'value': 0.0, 'quality': 0}
_emotiv_sensors_mapping['Z'] = {'value': '?', 'quality': 0}
_emotiv_sensors_mapping['Unknown'] = {'value': 0, 'quality': 0}


def _emotiv_get_level(data, bits):
    level = 0
    for i in range(13, -1, -1):
        level <<= 1
        b = (bits[i] // 8) + 1
        o = bits[i] % 8
        level |= (data[b] >> o) & 1
    return level * 0.5151515151


class EmotivPacketDecoder:
    """Decodifica paquetes 'old format' (el que corresponde a series que no empiezan con UD2016)."""

    def __init__(self):
        self.sensors = {k: dict(v) for k, v in _emotiv_sensors_mapping.items()}
        self.battery_raw = None
        self.battery_percent = None

    def decode(self, raw_data):
        counter = raw_data[0]
        if counter > 127:
            self.battery_raw = counter
            self.battery_percent = _emotiv_battery_values.get(counter)
            counter = 128

        for name, bits in _emotiv_sensors_14_bits.items():
            if 'GYRO' in name:
                continue
            self.sensors[name]['value'] = _emotiv_get_level(raw_data, bits)

        quality_value = _emotiv_get_level(raw_data, _emotiv_quality_bits)
        target = _emotiv_sensor_quality_bit.get(raw_data[0])
        if target:
            self.sensors[target]['quality'] = quality_value
        else:
            self.sensors['Unknown']['quality'] = quality_value

        return self.sensors, counter, self.battery_raw, self.battery_percent


def _emotiv_crypto_key(serial_number, is_research=False):
    k = bytearray(16)
    s = serial_number
    k[0] = ord(s[-1]); k[1] = 0; k[2] = ord(s[-2])
    if is_research:
        k[3] = ord('H'); k[4] = ord(s[-1]); k[5] = 0; k[6] = ord(s[-2]); k[7] = ord('T')
        k[8] = ord(s[-3]); k[9] = 0x10; k[10] = ord(s[-4]); k[11] = ord('B')
    else:
        k[3] = ord('T'); k[4] = ord(s[-3]); k[5] = 0x10; k[6] = ord(s[-4]); k[7] = ord('B')
        k[8] = ord(s[-1]); k[9] = 0; k[10] = ord(s[-2]); k[11] = ord('H')
    k[12] = ord(s[-3]); k[13] = 0; k[14] = ord(s[-4]); k[15] = ord('P')
    return bytes(k)


def _emotiv_new_crypto_key(serial_number):
    s = serial_number
    order = [-1, -2, -2, -3, -3, -3, -2, -4, -1, -4, -2, -2, -4, -4, -2, -1]
    return bytes(ord(s[i]) for i in order)


def _emotiv_build_cipher(serial_number, is_research=False):
    if serial_number.startswith("UD2016"):
        key = _emotiv_new_crypto_key(serial_number)
    else:
        key = _emotiv_crypto_key(serial_number, is_research)
    return AES.new(key, AES.MODE_ECB)


def _emotiv_decrypt_packet(cipher, data):
    return cipher.decrypt(data[:16]) + cipher.decrypt(data[16:32])


def _emotiv_find_device():
    for dev in hid.enumerate():
        product = (dev.get("product_string") or "").lower()
        manufacturer = (dev.get("manufacturer_string") or "").lower()
        if "emotiv" in product or "emotiv" in manufacturer or "epoc" in product:
            return dev
    return None


class EmotivReader(QThread):
    """
    Lee la diadema Emotiv EPOC+ y pone filas en data_queue con el MISMO
    formato que SerialReader: [Tm, valor_canal_1, ..., valor_canal_14],
    todo como strings, para reusar sin cambios el resto del pipeline
    (PlottingThread, RecordingThread, filtro, guardado).

    Nota de escala: el resto de la app asume que los valores crudos vienen
    de un ADC ADS1299 y les aplica la conversion
        microvolts = raw_counts * (V_REF / (GAIN * (2**23 - 1))) * 1e6
    Como la Emotiv ya entrega un valor aproximado en microvolts
    directamente (ver _emotiv_get_level), aqui se hace la conversion
    INVERSA antes de encolar el dato, para que cuando el pipeline
    existente aplique su formula de nuevo, el resultado final en pantalla
    sea el microvoltaje correcto de la Emotiv (sin tocar el resto del
    codigo).
    """
    error_signal = pyqtSignal(str)
    quality_signal = pyqtSignal(dict)   # {canal: valor_calidad}
    battery_signal = pyqtSignal(int)    # porcentaje de bateria (0-100)

    V_REF = 4.5
    GAIN = 24
    LSB_UNIT = V_REF / (GAIN * (2 ** 23 - 1))

    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue
        self.running = False
        self.cipher = _emotiv_build_cipher(EMOTIV_SERIAL_NUMBER, EMOTIV_IS_RESEARCH)
        self.decoder = EmotivPacketDecoder()
        self._inv_scale = 1.0 / (self.LSB_UNIT * 1000000.0)

    def run(self):
        device_info = _emotiv_find_device()
        if device_info is None:
            self.error_signal.emit("No se encontro la diadema Emotiv EPOC+ conectada (revisa el USB/dongle).")
            return
        try:
            h = hid.device()
            h.open_path(device_info["path"])
            h.set_nonblocking(0)
        except Exception as ex:
            self.error_signal.emit(f"No se pudo abrir el dispositivo Emotiv: {ex}")
            return

        self.running = True
        start_time = time.time()
        last_quality_emit = 0.0
        while self.running:
            try:
                data = h.read(EMOTIV_PACKET_SIZE, timeout_ms=1000)
            except Exception as ex:
                self.error_signal.emit(f"Error de lectura Emotiv: {ex}")
                continue
            if not data:
                continue
            raw = bytes(data[:EMOTIV_PACKET_SIZE])
            if len(raw) != EMOTIV_PACKET_SIZE:
                continue
            try:
                decrypted = _emotiv_decrypt_packet(self.cipher, raw)
            except Exception:
                continue
            sensors, counter, battery_raw, battery_percent = self.decoder.decode(list(decrypted))

            elapsed_ms = (time.time() - start_time) * 1000.0
            row = [str(elapsed_ms)]
            for ch in EMOTIV_CHANNELS:
                pseudo_counts = sensors[ch]['value'] * self._inv_scale
                row.append(str(pseudo_counts))
            self.data_queue.put(row)

            # Calidad: se actualiza a lo sumo cada 200ms para no saturar la UI
            # (los valores de calidad igual llegan de a un canal por paquete).
            now = time.time()
            if now - last_quality_emit > 0.2:
                quality = {ch: sensors[ch]['quality'] for ch in EMOTIV_CHANNELS}
                self.quality_signal.emit(quality)
                last_quality_emit = now

            if battery_percent is not None:
                self.battery_signal.emit(battery_percent)
        h.close()

    def stop(self):
        self.running = False
        self.quit()


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
                else:
                    #time.sleep(0.2)  # Evitar un bucle muy rápido mientras se leen registros
                    time.sleep(0.2)  # Evitar un bucle muy rápido mientras se leen registros

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
            if ch.lower() == 'tm':
                df_microvolts[ch] = self.recording_df[ch]
                continue
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
        channel_columns = [c for c in df_filtered.columns if c.lower() != 'tm' ]
        print(f"Aplicando filtro a canales: {channel_columns}")

        # # # # for ch in channel_columns:
        # # # #     try:
        # # # #         values = df_filtered[ch].astype(np.float32).values
        # # # #         filtered_values = self.filter.apply(values)
        # # # #         df_filtered[ch] = filtered_values
        # # # #     except Exception:
        # # # #         pass
        
        V_REF = 4.5
        GAIN = 24  # Ajusta esto si usas otra ganancia en el ADS1299
        LSB_UNIT = V_REF / (GAIN * (2**23 - 1))

        # ... dentro de tu hilo ...
        for idx, ch in enumerate(channel_columns):            # 1. Obtener valores crudos
            raw_values = df_filtered[ch].values
            values = np.array(raw_values, dtype=np.float32)

            # 2. Convertir a Microvoltios (uV) antes de filtrar
            # Aplicamos la fórmula: count * LSB_UNIT * 1e6
            values = values * LSB_UNIT * 1000000 

            # 3. Aplicar filtros (ahora sobre valores reales)
            #print(f"Aplicando filtro al canal {ch}...")
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
        # self.eight_channels =["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.eight_channels =["Oz","Cz","F3","F4","Fz","P3","P4","Pz"]
        self.sixteen_channels = self.columns[1:]
        self.df_sixteen= pd.DataFrame(columns=self.columns)
        self.apply_filter = False
        self.overlay_mode = False
        #  ["Oz","Po7","Po4","Po3","P4","P3","Po","Pz","Fz","F2","F3","F4,"AF3","Cz","AF4","F1"]
        self.serial_thread = None
        self.port = 'COM9'
        self.baudrate = 230400 *2
        self.test_mode = False  # Flag for test mode
        self.sixteen_channels_mode=False
        self.use_emotiv = False  # Flag para leer desde la diadema Emotiv en vez del puerto serial
        self.emotiv_channels = EMOTIV_CHANNELS
        self.emotiv_quality = {ch: 0 for ch in EMOTIV_CHANNELS}
        self.emotiv_battery_percent = None
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
        self.emotiv_checkbox = QCheckBox("Usar Emotiv EPOC+")
        self.emotiv_checkbox.setChecked(self.use_emotiv)
        self.emotiv_checkbox.stateChanged.connect(self.toggle_emotiv)
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
        control_layout.addWidget(self.emotiv_checkbox)
        self.emotiv_battery_label = QLabel("Bateria Emotiv: -")
        self.emotiv_battery_label.setVisible(False)
        control_layout.addWidget(self.emotiv_battery_label)
        #control_layout.addWidget(self.button_print_registers)
        control_layout.addWidget(self.button_check_loff_statp)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.scrolling_area)
        self.emotiv_overlay_quality_label = QLabel("")
        self.emotiv_overlay_quality_label.setWordWrap(True)
        self.emotiv_overlay_quality_label.setVisible(False)
        main_layout.addWidget(self.emotiv_overlay_quality_label)

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

    def toggle_emotiv(self, state):
        use_emotiv = state == 2
        if use_emotiv == self.use_emotiv:
            return
        was_running = self.serial_thread is not None and self.serial_thread.isRunning()
        if was_running:
            self.stop_serial()
        self.use_emotiv = use_emotiv
        # Mientras se usa Emotiv no tiene sentido elegir 8/16 canales ni modo prueba,
        # ya que la diadema siempre entrega 14 canales fijos.
        self.channels_checkbox.setEnabled(not use_emotiv)
        self.test_checkbox.setEnabled(not use_emotiv)
        self.emotiv_battery_label.setVisible(use_emotiv)
        if use_emotiv:
            self.channels = self.emotiv_channels
            self.df = pd.DataFrame(columns=["Tm"] + self.emotiv_channels)
            self.label_port.setText("Emotiv EPOC+ (HID)")
            self.label_baudrate.setText("-")
        else:
            self.set_channel_mode(self.sixteen_channels_mode)
            self.label_port.setText(self.port)
            self.label_baudrate.setText(str(self.baudrate))
            self.emotiv_overlay_quality_label.setVisible(False)
        self._rebuild_plots()
        self.update_emotiv_quality_display()
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
        self.update_emotiv_quality_display()
    def set_apply_filter(self, state):
        self.apply_filter = state == 2
        if self.plotting_thread:
            self.plotting_thread.set_apply_filter(self.apply_filter)

    def on_emotiv_quality(self, quality_dict):
        """Recibe la calidad de canal que emite EmotivReader y la refleja en la UI."""
        self.emotiv_quality.update(quality_dict)
        self.update_emotiv_quality_display()

    def on_emotiv_battery(self, percent):
        self.emotiv_battery_percent = percent
        self.emotiv_battery_label.setText(f"Bateria Emotiv: {percent}%")

    def update_emotiv_quality_display(self):
        """
        Modo separado (un subplot por canal): la calidad se muestra en el
        titulo de cada grafica.
        Modo superpuesto: la calidad de todos los canales se muestra junta
        en el label de abajo, ya que ahi no hay un titulo por canal.
        """
        if not self.use_emotiv:
            return
        if self.overlay_mode:
            texto = "  |  ".join(
                f"{ch}: {int(self.emotiv_quality.get(ch, 0))}" for ch in self.emotiv_channels
            )
            self.emotiv_overlay_quality_label.setText("Calidad de señal (Emotiv): " + texto)
            self.emotiv_overlay_quality_label.setVisible(True)
        else:
            self.emotiv_overlay_quality_label.setVisible(False)
            for idx, ch in enumerate(self.channels):
                if idx < len(self.plots) and ch in self.emotiv_quality:
                    self.plots[idx].setTitle(f"Canal: {ch}  -  calidad: {int(self.emotiv_quality[ch])}")

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
        if self.use_emotiv:
            print("Starting Emotiv EPOC+ reader.")
            self.serial_thread = EmotivReader(self.data_queue)
            self.serial_thread.error_signal.connect(self.on_serial_error)
            self.serial_thread.quality_signal.connect(self.on_emotiv_quality)
            self.serial_thread.battery_signal.connect(self.on_emotiv_battery)
        elif self.test_mode:
            print("Starting test mode with CSV data.")
            self.serial_thread = CSVReader('Labuena.csv', self.data_queue,self.sixteen_channels_mode)
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
        emotiv_columns = ["Tm"] + self.emotiv_channels
        self.recording_thread = RecordingThread()
        if self.use_emotiv:
            columns_df = emotiv_columns
        elif self.sixteen_channels_mode:
            columns_df = sixteen_columns
        else:
            columns_df = eigth_columns
        self.recording_thread.start_recording(columns_df=columns_df, duration=duration)

    def return_recorded_data(self):
        if self.apply_filter:
            return self.recording_thread.get_filtered_recording_df()
        else:            
            return self.recording_thread.get_recorded_data()
    def check_registers(self,print_output=True):
        if self.use_emotiv:
            print("Check LOFF_STATP no aplica en modo Emotiv (es especifico del ADS1299 por serial).")
            return
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
        if self.use_emotiv:
            print("Check LOFF_STATP no aplica en modo Emotiv.")
            return
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