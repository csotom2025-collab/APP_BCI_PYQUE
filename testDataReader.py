import pandas as pd
import time
import threading
import queue

class CSVReader(threading.Thread):
    def __init__(self, csv_file, data_queue:queue.Queue, sixteen_mode, delay=0.001):
        super().__init__()
        self.csv_file = csv_file
        self.data_queue = data_queue
        self.delay = delay  # Delay between sending rows, in seconds
        self.running = False
        self.eight_channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
        self.sixteen_channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","ch9","ch10","ch11","ch12","ch13","ch14","ch15","ch16"]
        self.channels =(self.sixteen_channels if sixteen_mode else self.eight_channels)
    def run(self):
        try:
            df = pd.read_csv(self.csv_file)
            self.running = True
            for index, row in df.iterrows():
                if not self.running:
                    break
                # Convert row to list of strings, like serial data
                values = [str(row['Tm'])] + [str(row[ch]) for ch in self.channels]
                self.data_queue.put(values)
                #print(f"Simulated: {','.join(values)}")
                time.sleep(self.delay)
                #print(self.data_queue.qsize())
        except Exception as e:
            print(f"CSV read error: {e}")

    def stop(self):
        self.running = False