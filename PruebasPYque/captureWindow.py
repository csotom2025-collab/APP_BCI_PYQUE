from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit,QMainWindow, QApplication, QGridLayout
import sys
from KeyboardCaptureController import ControllerKeyboardCapture




LETTERS = ["A", "E", "I", "O", "U",
    "S", "R", "N", "L", "D",
    "C", "T", "M", "P", "B",
    "G", "V", "Y", "Q", "H",
    "F", "Z", "J", "Ñ", "X",
    "K ", "W"]
NUMBERS = ["1", "2", "3",
    "4", "5", "6",
    "7", "8", "9", "0"]

CONTROLS = ["ESP", "BORR", "ENTER"]
class CaptureWindow(QWidget):
    def __init__(self,ControllerKeyboard=None):
        super().__init__()
        self.setWindowTitle("Captura de datos")        
        self.setup_ui()
        self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        self.show_users()

        self.controller_keyboard = ControllerKeyboard
        
    def setup_ui(self):
        self.layout = QGridLayout()
        self.combo_box_users = QComboBox()
        self.combo_box_users.currentIndexChanged.connect(self.show_path)

        self.combo_box_character_type = QComboBox()
        self.combo_box_character_type.addItems(["Letters", "Numbers", "Controls"])
        self.combo_box_character_type.currentIndexChanged.connect(self.update_character_options)

        self.combo_box_character = QComboBox()

        self.button_start_capture = QPushButton("Iniciar captura")
        self.button_start_capture.clicked.connect(self.start_capture)

        self.button_simulation = QPushButton("Iniciar simulación")
        self.button_simulation.clicked.connect(self.start_simulation)

        self.layout.addWidget(QLabel("Seleccionar usuario:"), 0, 0)
        self.layout.addWidget(self.combo_box_users, 0, 1)
        self.layout.addWidget(QLabel("Tipo de caracter:"), 1, 0)
        self.layout.addWidget(self.combo_box_character_type, 1, 1)
        self.layout.addWidget(QLabel("Caracter :"), 2, 0)
        self.layout.addWidget(self.combo_box_character, 2, 1)
        self.layout.addWidget(self.button_start_capture, 3, 0, 1, 2)
        self.layout.addWidget(self.button_simulation, 4, 0, 1, 2)

        self.setLayout(self.layout)
        self.move(300, 350)

    def show_users(self):
        self.combo_box_users.clear()
        self.combo_box_users.addItems(self.data.keys())

    def update_character_options(self):
        self.combo_box_character.clear()
        character_type = self.combo_box_character_type.currentText()
        if character_type == "Letters":
            self.combo_box_character.addItems(LETTERS)
        elif character_type == "Numbers":
            self.combo_box_character.addItems(NUMBERS)
        elif character_type == "Controls":
            self.combo_box_character.addItems(CONTROLS)

    def show_path(self):
        user = self.combo_box_users.currentText()
        path = self.data.get(user, "No path found")
        print(f"Ruta del usuario seleccionado: {path}")

    def start_capture(self):
        user = self.combo_box_users.currentText()
        path = self.data.get(user)
        character = self.combo_box_character.currentText()
        character_type = self.combo_box_character_type.currentText()
        self.controller_keyboard.flash_character(character)
        self.controller_keyboard.save_capture(user, path, character_type, character)

    def start_simulation(self):
        self.controller_keyboard.start_simulation()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    window.show()
    sys.exit(app.exec())