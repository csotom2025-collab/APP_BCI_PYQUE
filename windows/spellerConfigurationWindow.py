import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit
from PyQt6.QtCore import QTimer, Qt, QRect
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtCore import QPoint
from windows.gridWindow import KeyboardWindow
from controllers.SaveCaptureController import controllerSaveCapture
class SpellerConfigurationWindow(QWidget):
    def __init__(self, predict_controller=None, save_capture_controller:controllerSaveCapture=None, keyboard_window:KeyboardWindow=None):
        super().__init__()
        self.setWindowTitle("Configuración del Speller")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.move(100, 150)
        self.predict_controller = predict_controller
        self.save_capture_controller = save_capture_controller
        self.keyboard_window = keyboard_window
        self.setup_ui()
    def setup_ui(self):
        self.layout.addWidget(QLabel("Configuración del Speller tiempo Real"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(QLabel("Usuario :"), 1, 0)
        self.user_text = QLineEdit()
        self.user_text.setText("12")
        self.layout.addWidget(self.user_text, 1, 1)
        self.layout.addWidget(QLabel("Modelo:"), 2, 0)
        self.model = QLineEdit()
        self.model.setText("LDA.pt")
        self.layout.addWidget(self.model, 2, 1)
        self.save_button = QPushButton("Guardar Configuración")
        self.save_button.clicked.connect(self.save_configuration)
        self.start_capture_trial_button = QPushButton("Iniciar CapturaOnlinetiral")
        self.start_capture_trial_button.clicked.connect(self.start_capture_trial)
        self.layout.addWidget(self.start_capture_trial_button,4,0,1,2)
        self.layout.addWidget(self.save_button, 3, 0, 1, 2)
        self.hide()
    def save_configuration(self):
        self.user = "User" + self.user_text.text()
        self.model_name = self.model.text()
        self.model_path = "Results/" + self.user + "/models/" + self.model_name
        self.set_model_path(self.model_path)
        print(f"Guardadno Configuración: Usuario={self.user}, Modelo={self.model_name}, Ruta={self.model_path}")

    def set_model_path(self, model_path):
        self.predict_controller.set_model_path(model_path)
    
    def start_capture_trial(self):
        character_type = "Speller"
        character = "UNKNOWN"
        duration =2
        self.start_chess_flashes()
        if self.save_capture_controller:
            self.new_record_path = self.save_capture_controller.start_capture(self.user, character_type, character, duration,callback=None,online=True)
        qtime = QTimer()
        qtime.singleShot(2*1000+900, self.predict_character)
    def predict_character(self):
        if self.predict_controller:
            self.predict_controller.predict(self.new_record_path)
    def start_chess_flashes(self):
        self.keyboard_window.start_paradigm(times=5)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellerConfigurationWindow()
    window.show()
    sys.exit(app.exec())