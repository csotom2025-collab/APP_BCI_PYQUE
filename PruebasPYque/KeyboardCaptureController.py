import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit,QMainWindow, QApplication, QGridLayout
from PyQt6.QtCore import QTimer
from gridWindow import KeyboardWindow


FLASH_ROUNDS = 2      # cuántas rondas de flash por botón
FLASH_DURATION = 0.5 # segundos que dura cada flash
PAUSE_BETWEEN = 0.9  # pausa entre flashes

class ControllerKeyboardCapture:
    def __init__(self, WindowKeyboard: KeyboardWindow):
        self.keyboard_window = WindowKeyboard

    def flash_character(self, character,duration=FLASH_DURATION):
        for button in self.keyboard_window.findChildren(QPushButton):
            if button.text() == character:
                self.keyboard_window.flash_button(button, duration)
                break
        
    def save_capture(self, user, path, character_type, character):
        print(f"Guardando  para {user} en la ruta: {path} Caracter: {character}, Tipo: {character_type}")

    def start_simulation(self):
        LETTERS = ["A", "E", "I", "O", "U", "S", "R", "N", "L", "D", "C", "T", "M", "P", "B", "G", "V", "Y", "Q", "H", "F", "Z", "J", "Ñ", "X","K ", "W"]   
        hello_world = "HELLO WORLD"
        lista_simbolos = list(hello_world)
        print(lista_simbolos)
        self.flash_idx=0
        self.flash_list = lista_simbolos * FLASH_ROUNDS
        self._flash_step(FLASH_DURATION)        

    def _flash_step(self, flash_time):
        if self.flash_idx >= len(self.flash_list):
            return
        target = self.flash_list[self.flash_idx]
        self.flash_character(target, flash_time)
        self.flash_idx += 1
        QTimer.singleShot(int(PAUSE_BETWEEN * 1000), lambda: self._flash_step(flash_time))