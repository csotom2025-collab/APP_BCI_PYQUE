from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from InterfasPyQue import SignalsWindow
from KeyboardCaptureController import ControllerKeyboardCapture
from captureWindow import CaptureWindow
from serialWindow import SerialWindow
from gridWindow import KeyboardWindow
from trainWindow import TrainWindow
from serialConfigSingnalsController import ControllerSerialConfig

class Menu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals_window = SignalsWindow()
        self.keyboard_window = KeyboardWindow()
        self.train_window = TrainWindow()
        self.capture_window = CaptureWindow()
        self.controller_serial_config = ControllerSerialConfig(self)
        self.serial_window = SerialWindow(self.controller_serial_config)
        self.controller_keyboard = ControllerKeyboardCapture(self.keyboard_window)  # Inicializa el controlador sin ventanas por ahora
        self.capture_window.controller_keyboard = self.controller_keyboard

        self.port =2222
        self.baudioRate = 9600

        self.setup_ui()

        
    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Menú Principal")
        button_signals = QPushButton("Visualizar Señales EEG")
        button_signals.clicked.connect(self.open_signals_window)
        button_grid = QPushButton("Mostrar Grid")
        button_grid.clicked.connect(self.open_grid_window)
        button_train = QPushButton("Entrenamiento")
        button_train.clicked.connect(self.open_train_window)
        button_capture = QPushButton("Captura de datos")
        button_capture.clicked.connect(self.open_capture_window)
        button_serial_config = QPushButton("Configurar Serial")
        button_serial_config.clicked.connect(self.open_config_serial_window)


        self.label_port = QLabel(f"Puerto: {self.port}")
        self.label_baudrate = QLabel(f"Baudrate: {self.baudioRate}")

        self.setLayout(self.layout)
        self.layout.addWidget(self.label)
        self.layout.addWidget(button_signals)
        self.layout.addWidget(button_grid)
        self.layout.addWidget(button_train)
        self.layout.addWidget(button_capture)
        self.layout.addWidget(button_serial_config)
        self.layout.addWidget(self.label_port)
        self.layout.addWidget(self.label_baudrate)
        
        self.move(0,0)

    def open_signals_window(self):
        self.signals_window.show()
    def open_grid_window(self):
        self.keyboard_window.show()
    def open_train_window(self):
        self.train_window.show()
    def open_capture_window(self):
        self.capture_window.show()
    def open_config_serial_window(self):
        self.serial_window.show()
    def update_serial_config(self, port, baudrate):
        self.port = port
        self.baudioRate = baudrate
        self.label_port.setText(f"Puerto: {self.port}")
        self.label_baudrate.setText(f"Baudrate: {self.baudioRate}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú Principal")
        self.menu = Menu(self)
        self.setCentralWidget(self.menu)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()