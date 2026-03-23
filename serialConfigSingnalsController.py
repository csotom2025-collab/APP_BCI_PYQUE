from PyQt6.QtWidgets import QComboBox, QMessageBox, QWidget, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit
from SerialMonitor import SignalsWindow
class ControllerSerialConfig:
    def __init__(self, signals_window):
        self.signals_window = signals_window

    def update_serial_config(self, port, baudrate):
        if not baudrate.isdigit():
            print("Baudrate debe ser un número.")
            QMessageBox.warning(None, "Error de Configuración", "Baudrate debe ser un número.")
            return
        if int(baudrate) <= 0:
            print("Baudrate debe ser un número positivo.")
            QMessageBox.warning(None, "Error de Configuración", "Baudrate debe ser un número positivo.")
            return
        
        self.signals_window.update_serial_config(port, int(baudrate))
