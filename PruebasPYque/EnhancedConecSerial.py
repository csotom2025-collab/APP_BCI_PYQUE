import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import numpy as np
import threading
import queue

class DynamicSerialConnector:
    def __init__(self, port='COM3', baudrate=230400, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.running = False
        self.data_queue = queue.Queue()
        self.df = pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"])
        self.channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.update_count = 0
        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 10))
        self.fig.suptitle('Señales EEG por Canal - SignalTest (Tiempo Real)')
        self.axes = self.axes.flatten()
        plt.ion()
        plt.show()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            self.ser.flush()
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            time.sleep(0.05)
            while self.ser.in_waiting:
                _ = self.ser.readline()
            ini = 'x'
            self.ser.write(ini.encode('utf-8'))
            print(f"Connected to {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def reconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("Attempting to reconnect...")
        return self.connect()

    def read_data(self):
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    raw = self.ser.readline()
                    line = raw.decode('utf-8', errors='replace').strip()
                    if line.endswith(','):
                        line = line[:-1]
                    values = line.split(",")
                    if len(values) == 9:
                        self.data_queue.put(values)
            except Exception as e:
                print(f"Error reading data: {e}")
                if not self.reconnect():
                    break

    def process_data(self):
        with open('datosLectura.csv', 'w', encoding='utf-8') as datos:
            datos.write("Tm,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n")
            while self.running:
                try:
                    values = self.data_queue.get(timeout=1)
                    print(f"Datos recibidos: {','.join(values)}")
                    datos.write(f"{','.join(values)}\n")
                    datos.flush()
                    row = []
                    for v in values:
                        try:
                            row.append(float(v))
                        except Exception:
                            row.append(np.nan)
                    if len(row) == len(self.df.columns):
                        self.df.loc[len(self.df)] = row
                    self.update_count += 1
                    if self.update_count % 100 == 0:
                        self.update_plot()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error processing data: {e}")

    def update_plot(self):
        df_plot = self.df.tail(100)
        for idx, channel in enumerate(self.channels):
            self.axes[idx].clear()
            self.axes[idx].plot(df_plot['Tm'].values, df_plot[channel].values, 'b-', linewidth=1)
            self.axes[idx].set_title(f'Canal: {channel}')
            self.axes[idx].set_xlabel('Tiempo')
            self.axes[idx].set_ylabel('Amplitud')
            self.axes[idx].grid(True, alpha=0.3)
            ymin = float(df_plot[channel].min())
            ymax = float(df_plot[channel].max())
            if ymax - ymin < 1e-6:
                mid = (ymin + ymax) / 2
                self.axes[idx].set_ylim(mid - 0.5, mid + 0.5)
            else:
                margin = (ymax - ymin) * 0.1
                self.axes[idx].set_ylim(ymin - margin, ymax - margin)
            xmin = float(df_plot['Tm'].min())
            xmax = float(df_plot['Tm'].max())
            if xmax > xmin:
                self.axes[idx].set_xlim(xmin, xmax)
        plt.tight_layout()
        plt.pause(0.01)

    def start(self):
        if not self.connect():
            return
        self.running = True
        self.read_thread = threading.Thread(target=self.read_data)
        self.process_thread = threading.Thread(target=self.process_data)
        self.read_thread.start()
        self.process_thread.start()
        print("Started dynamic serial connection. Press Ctrl+C to stop.")

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        plt.close()
        print("Connection closed.")

if __name__ == "__main__":
    connector = DynamicSerialConnector()
    try:
        connector.start()
        while connector.running:
            time.sleep(1)
    except KeyboardInterrupt:
        connector.stop()