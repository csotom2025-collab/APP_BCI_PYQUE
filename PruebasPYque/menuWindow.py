from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from InterfasPyQue import SignalsWindow
from captureWindow import CaptureWindow
from gridWindow import GridButtonWindow
from trainWindow import TrainWindow
class Menu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.setLayout(self.layout)
        self.layout.addWidget(self.label)
        self.layout.addWidget(button_signals)
        self.layout.addWidget(button_grid)
        self.layout.addWidget(button_train)
        self.layout.addWidget(button_capture)
        

    def open_signals_window(self):
        self.signals_window = SignalsWindow()
        self.signals_window.show()
    def open_grid_window(self):
        self.grid_window = GridButtonWindow()
        self.grid_window.show()
    def open_train_window(self):
        self.train_window = TrainWindow()
        self.train_window.show()
    def open_capture_window(self):
        self.capture_window = CaptureWindow()
        self.capture_window.show()
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