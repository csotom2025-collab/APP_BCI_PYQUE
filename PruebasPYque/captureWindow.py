from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit,QMainWindow, QApplication, QGridLayout
import sys
from InterfasPyQue import SignalsWindow

class serialWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Serial Data Capture")
        self.layout = QGridLayout()
        self.port = QLineEdit()
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
            port = self.get_port()
            baudrate = self.get_baudrate()
            print(f"Configurando serial puerto {port} con baudrate {baudrate}")
            self.signal_window = SignalsWindow() ### pasar parametros de puerto y baud 
            self.signal_window.show()
    def get_port(self):
        text = self.port.text()
        return int(text)
        
    def get_baudrate(self):
        text = self.baudrate.text()
        return int(text)

class CaptureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captura de datos")        
        self.layout = QVBoxLayout()
        self.combo_box_users = QComboBox()
        self.combo_box_users.currentIndexChanged.connect(self.show_path)
        self.button_start_capture = QPushButton("Iniciar captura")
        self.button_start_capture.clicked.connect(self.show_serial_configuration)


        self.character = QLineEdit()
        self.combo_box_character_type = QComboBox()
        self.combo_box_character_type.addItems(["Letters", "Numbers", "Controls"])
        

        self.layout.addWidget(self.combo_box_users)
        self.layout.addWidget(self.character)
        self.layout.addWidget(self.combo_box_character_type)
        self.layout.addWidget(self.button_start_capture)

        self.setLayout(self.layout)
        self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        self.show_users()

        self.move(300, 350)


    def show_users(self):
        self.combo_box_users.clear()
        self.combo_box_users.addItems(self.data.keys())

    def show_path(self):
        user = self.combo_box_users.currentText()
        path = self.data.get(user, "No path found")
        print(f"Ruta del usuario seleccionado: {path}")
    
    def show_serial_configuration(self):
        self.serial_window = serialWindow()
        self.serial_window.show()
    def start_capture(self):
        self.signal_window = SignalsWindow()
        self.signal_window.show()
        user = self.combo_box_users.currentText()
        path = self.data.get(user)
        print(f"Capturando datos para {user} en la ruta: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    window.show()
    sys.exit(app.exec())