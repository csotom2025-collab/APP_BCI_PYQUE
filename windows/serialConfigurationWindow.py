from PyQt6.QtWidgets import QComboBox, QWidget, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit
from windows.SerialMonitorWindow import SignalsWindow
from windows.gridWindow import KeyboardWindow

class SerialConfiguration(QWidget):
    def __init__(self,controller_serial_config):
        super().__init__()
        self.controller_serial_config = controller_serial_config
        self.setWindowTitle("Serial Data Capture")
        self.layout = QGridLayout()
        self.port = QComboBox()
        self.port.addItems([f"COM{i}" for i in range(10)])
        self.baudrate = QLineEdit()


        self.layout.addWidget(QLabel("Puerto:"), 0, 0)
        self.layout.addWidget(self.port, 0, 1)
        self.layout.addWidget(QLabel("Baudrate:"), 1, 0)
        self.layout.addWidget(self.baudrate, 1, 1)
        self.button_configure = QPushButton("Configurar")
        self.button_configure.clicked.connect(self.configure_serial)
        self.layout.addWidget(self.button_configure, 2, 0, 1, 2)
        self.setLayout(self.layout)
        self.move(100, 150)


    def configure_serial(self):
            self.controller_serial_config.update_serial_config(self.get_port(), self.get_baudrate())
    def get_port(self):
        text = self.port.currentText()
        return text
        
    def get_baudrate(self):
        text = self.baudrate.text()
        return text
