from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit,QMainWindow, QApplication, QGridLayout
import sys
from controllers.KeyboardCaptureController import ControllerKeyboardCapture
from controllers.SaveCaptureController import controllerSaveCapture
from windows.gridWindow import BlackScreen, KeyboardWindow
from PyQt6.QtGui import QCloseEvent


LETTERS = ["A", "E", "I", "O", "U",
    "S", "R", "N", "L", "D",
    "C", "T", "M", "P", "B",
    "G", "V", "Y", "Q", "H",
    "F", "Z", "J", "Ñ", "X",
    "K ", "W"]
NUMBERS = ["1", "2", "3",
    "4", "5", "6",
    "7", "8", "9", "0"]

CONTROLS = ["───", "⟵", "↩"]
class CaptureWindow(QWidget):
    def __init__(self,ControllerKeyboard:ControllerKeyboardCapture=None,ControllerSaveCapture:controllerSaveCapture=None):
        super().__init__()
        self.setWindowTitle("Captura de datos")        
        self.setup_ui()
        self.data = [f"User{i}"for i in range(20)]
        #self.show_users()
        self.update_character_options()
        self.controller_keyboard = ControllerKeyboard
        self.controller_save_capture = ControllerSaveCapture

    def setup_ui(self):
        self.layout = QGridLayout()
        # self.combo_box_users = QComboBox()
        # self.combo_box_users.currentIndexChanged.connect(self.show_path)
        self.user_edit_line = QLineEdit()
        self.combo_box_character_type = QComboBox()
        self.combo_box_character_type.addItems(["Letters", "Numbers", "Controls"])
        self.combo_box_character_type.currentIndexChanged.connect(self.update_character_options)

        self.path_edit_line = QLineEdit()

        self.combo_box_character = QComboBox()

        self.button_start_capture = QPushButton("Iniciar captura")
        self.button_start_capture.clicked.connect(self.start_capture)

        self.button_simulation = QPushButton("Iniciar Captura N veces")
        self.button_simulation.clicked.connect(self.start_simulation)
        self.duration_recording_edit_line = QLineEdit()
        self.duration_recording_edit_line.setText("2")
        self.n_times_edit_line = QLineEdit()
        self.n_times_edit_line.setText("1")

        self.grid_button = QPushButton("Mostrar Grid")
        self.grid_button.clicked.connect(self.show_grid)


        # self.layout.addWidget(QLabel("Seleccionar usuario:"), 0, 0)
        # self.layout.addWidget(self.combo_box_users, 0, 1)
        self.layout.addWidget(QLabel("Numero de Usuario User:"), 0, 0)
        self.layout.addWidget(self.user_edit_line, 0, 1)
        # self.layout.addWidget(QLabel("Ruta de guardado:"), 1, 0)
        # self.layout.addWidget(self.path_edit_line, 1, 1)
        self.layout.addWidget(QLabel("Tipo de caracter:"), 2, 0)
        self.layout.addWidget(self.combo_box_character_type, 2, 1)
        self.layout.addWidget(QLabel("Caracter :"), 3, 0)
        self.layout.addWidget(self.combo_box_character, 3, 1)
        self.layout.addWidget(QLabel("Duracion grabacion"), 4, 0)
        self.layout.addWidget(self.duration_recording_edit_line,4,1)
        self.layout.addWidget(self.button_start_capture, 5, 0, 1, 2)
        self.layout.addWidget(self.button_simulation, 6, 0, 1, 2)
        self.layout.addWidget(self.grid_button, 7, 0, 1, 2)
        # self.layout.addWidget(QLabel("Veces:"), 7, 0, 1, 1)
        # self.layout.addWidget(self.n_times_edit_line, 7, 1, 1, 1)
        self.setLayout(self.layout)
        self.move(300, 350)

    # def show_users(self):
    #     self.combo_box_users.clear()
    #     self.combo_box_users.addItems(self.data)

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
        
        
    def start_capture(self):
        # user = self.combo_box_users.currentText()
        user = "User" + self.user_edit_line.text()
        #path = self.path_edit_line.text()
        character = self.combo_box_character.currentText()
        character_type = self.combo_box_character_type.currentText()
        self.controller_keyboard.flash_character(character)
        duration = self.duration_recording_edit_line.text()
        ###salvar captura
        duration= int(duration)
        self.controller_save_capture.start_capture(user,character_type,character,duration)

    def start_simulation(self):
        self.controller_keyboard.start_simulation()
    def show_grid(self):
        self.keyboard_window = KeyboardWindow(training_mode=True)
        self.keyboard_window.show()
        self.controller_keyboard.keyboard_window = self.keyboard_window  # Actualiza la referencia en el controlador
        self.black_screen = BlackScreen()
        self.black_screen.show()
    def closeEvent(self, event: QCloseEvent):
        """Cierra todas las ventanas secundarias y la aplicación."""
        #Acepta el evento de cierre de la ventana principal
        event.accept()
        self.keyboard_window.close()
        self.black_screen.close()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    window.show()
    sys.exit(app.exec())