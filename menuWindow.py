from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QCloseEvent
from controllers.SaveCaptureController import controllerSaveCapture
from windows.SerialMonitorWindow import SignalsWindow
from controllers.KeyboardCaptureController import ControllerKeyboardCapture
from windows.captureWindow import CaptureWindow
from windows.serialConfigurationWindow import SerialConfiguration
from windows.gridWindow import KeyboardWindow
from windows.trainWindow import TrainWindow
from controllers.serialConfigSingnalsController import ControllerSerialConfig
class Menu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals_window = SignalsWindow()
        self.keyboard_window = KeyboardWindow()
        self.train_window = TrainWindow()
        self.controller_serial_config = ControllerSerialConfig(self.signals_window)
        self.serial_window = SerialConfiguration(self.controller_serial_config)
        self.controller_keyboard = ControllerKeyboardCapture(self.keyboard_window)  # Inicializa el controlador sin ventanas por ahora
        self.controller_save_capture = controllerSaveCapture(self.signals_window)  # Inicializa el controlador de guardado con la ventana de señales
        self.capture_window = CaptureWindow(self.controller_keyboard, self.controller_save_capture)  # Pasa ambos controladores a la ventana de captura
        self.capture_window.controller_save_capture = self.controller_save_capture  # Asigna el controlador de
        

        self.setup_ui()

        
    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Menú Principal")
        button_signals = QPushButton("Visualizar Señales EEG")
        button_signals.clicked.connect(self.open_signals_window)
        button_grid = QPushButton("Mostrar Teclado")
        button_grid.clicked.connect(self.open_grid_window)
        button_train = QPushButton("Entrenamiento")
        button_train.clicked.connect(self.open_train_window)
        button_capture = QPushButton("Captura de datos")
        button_capture.clicked.connect(self.open_capture_window)
        button_serial_config = QPushButton("Configurar Serial")
        button_serial_config.clicked.connect(self.open_config_serial_window)


        self.setLayout(self.layout)
        self.layout.addWidget(self.label)
        self.layout.addWidget(button_signals)
        self.layout.addWidget(button_grid)
        self.layout.addWidget(button_train)
        self.layout.addWidget(button_capture)
        self.layout.addWidget(button_serial_config)

        
        self.move(100, 150)
    def closeEvent(self, event: QCloseEvent):
        """Handler to confirm close or perform cleanup."""
        reply = QMessageBox.question(self, 'Message', 
                                    "Are you sure you want to quit?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()  # Close the window
        else:
            event.ignore()  # Keep the window open

    def quit_application(self):
        QApplication.quit()
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
    

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú Principal")
        self.menu = Menu(self)

        self.setCentralWidget(self.menu)
    
    def closeEvent(self, event: QCloseEvent):
        """Cierra todas las ventanas secundarias y la aplicación."""
        #Acepta el evento de cierre de la ventana principal
        event.accept()
        QApplication.quit()
    
    def close_application(self):
        self.close()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()