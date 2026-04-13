import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QFont

GRID = [
    ["A", "E", "I", "O", "U"],
    ["S", "R", "N", "L", "D"],
    ["C", "T", "M", "P", "B"],
    ["G", "V", "Y", "Q", "H"],
    ["F", "Z", "J", "Ñ", "X"],
    ["K ", "W", "ESP", "BORR", "ENTER"]
]
GrindNumbers = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
    ["0"],
]

# Estilos para los temas
LIGHT_THEME = {
    "window_bg": "white",
    "button_bg": "#f7efef",
    "button_text": "black",
    "button_border": "#cccccc",
    "input_bg": "white",
    "input_text": "black",
    "theme_name": "Light"
}

DARK_THEME = {
    "window_bg": "#1e1e1e",
    "button_bg": "#2d2d2d",
    "button_text": "#ffffff",
    "button_border": "#444444",
    "input_bg": "#3c3c3c",
    "input_text": "#ffffff",
    "theme_name": "Dark"
}



class OutputLine(QWidget):
    def __init__(self, text):
        super().__init__()
        self.layout = QHBoxLayout()
        self.label = QLineEdit(text)
        self.label.setReadOnly(False)
        # Hacer responsive en lugar de tamaño fijo
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.label.setMinimumHeight(40)
        # Aumentar tamaño de fuente del output
        font = QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.layout.addWidget(self.label)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.layout)
    def set_text(self, text):
        self.label.setText(text)
    def add_character(self, char):
        current = self.label.text()
        self.label.setText(current + char)
        

class KeyboardWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid de botones")
        self.setMinimumSize(600, 500)
        self.current_theme = self.load_theme()
        self.buttons_list = []
        self.setup_ui()
        self.apply_theme(self.current_theme)

    def load_theme(self):
        """Carga el tema guardado o usa el tema claro por defecto"""
        config_file = "theme_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return DARK_THEME if config.get("dark_mode", False) else LIGHT_THEME
            except:
                return LIGHT_THEME
        return LIGHT_THEME

    def save_theme(self):
        """Guarda la configuración del tema actual"""
        config_file = "theme_config.json"
        is_dark = self.current_theme == DARK_THEME
        with open(config_file, 'w') as f:
            json.dump({"dark_mode": is_dark}, f)

    def setup_ui(self):
        self.grid_letters_layout = QGridLayout()
        self.grid_numbers_layout = QGridLayout()
        self.main_layout = QVBoxLayout()
        
        # Layout para el botón de tema
        theme_layout = QHBoxLayout()
        self.theme_button = QPushButton("🌙 Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_button)
        theme_layout.addStretch()
        
        # Layout principal con grid
        self.layout = QGridLayout()
        self.output_line = OutputLine("")
        self.show_grid()
        # Usar proporción de espacio: la salida ocupa 1/6 del espacio
        self.layout.addWidget(self.output_line, 0, 0, 1, 3)
        self.layout.addLayout(self.grid_letters_layout, 1, 0)
        self.layout.addLayout(self.grid_numbers_layout, 1, 1)
        # Configurar el stretch de filas y columnas
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 5)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setSpacing(5)
        
        self.main_layout.addLayout(theme_layout)
        self.main_layout.addLayout(self.layout)
        self.setLayout(self.main_layout)

    def show_grid(self):
        # Crear fuente para los botones
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        
        # Create grid using loops
        for row, letters in enumerate(GRID):
            for col, letter in enumerate(letters):
                button = QPushButton(f"{letter}",)
                button.setFont(font)
                # Usar política de tamaño en lugar de tamaño fijo
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                button.setMinimumSize(40, 40)
                button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
                self.grid_letters_layout.addWidget(button, row, col)
                self.buttons_list.append(button)
        
        for row, numbers in enumerate(GrindNumbers):
            for col, number in enumerate(numbers):
                button = QPushButton(f"{number}")
                button.setFont(font)
                # Usar política de tamaño en lugar de tamaño fijo
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                button.setMinimumSize(40, 40)
                button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
                self.grid_numbers_layout.addWidget(button, row, col)
                self.buttons_list.append(button)
        
        # Configurar stretch para los layouts internos
        for i in range(len(GRID)):
            self.grid_letters_layout.setRowStretch(i, 1)
        for j in range(len(GRID[0])):
            self.grid_letters_layout.setColumnStretch(j, 1)
        
        for i in range(len(GrindNumbers)):
            self.grid_numbers_layout.setRowStretch(i, 1)
        for j in range(max(len(row) for row in GrindNumbers)):
            self.grid_numbers_layout.setColumnStretch(j, 1)

    def button_clicked(self, button):
        print("Boton seleccionado :" + button.text())
        self.flash_button(button)
        self.add_character(button.text())

    def flash_button(self, button, duration=1.0):
        # Aplicar efecto de flash amarillo
        button.setStyleSheet(
            f"QPushButton {{ background-color: yellow; "
            f"color: black; border: 2px solid #cccccc; "
            f"border-radius: 5px; font-weight: bold; }}"
        )
        # Restaurar el tema actual después del flash
        QTimer.singleShot(int(duration * 1000), lambda: self.reapply_button_theme(button))

    def reapply_button_theme(self, button):
        """Reaplica el tema actual a un botón específico"""
        theme = self.current_theme
        button_style = (
            f"QPushButton {{ background-color: {theme['button_bg']}; "
            f"color: {theme['button_text']}; border: 2px solid {theme['button_border']}; "
            f"border-radius: 5px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {self.get_hover_color(theme['button_bg'])}; }}"
        )
        button.setStyleSheet(button_style)

    def toggle_theme(self):
        """Cambia entre tema claro y oscuro"""
        self.current_theme = DARK_THEME if self.current_theme == LIGHT_THEME else LIGHT_THEME
        self.apply_theme(self.current_theme)
        self.save_theme()
        self.update_theme_button()

    def apply_theme(self, theme):
        """Aplica el tema especificado a toda la ventana"""
        # Aplicar tema a la ventana
        self.setStyleSheet(f"QWidget {{ background-color: {theme['window_bg']}; }}")
        
        # Aplicar tema a la linea de entrada
        self.output_line.label.setStyleSheet(
            f"QLineEdit {{ background-color: {theme['input_bg']}; "
            f"color: {theme['input_text']}; border: 2px solid {theme['button_border']}; "
            f"padding: 5px; font-size: 12pt; }}"
        )
        
        # Aplicar tema a todos los botones
        button_style = (
            f"QPushButton {{ background-color: {theme['button_bg']}; "
            f"color: {theme['button_text']}; border: 2px solid {theme['button_border']}; "
            f"border-radius: 5px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {self.get_hover_color(theme['button_bg'])}; }}"
        )
        
        for button in self.buttons_list:
            button.setStyleSheet(button_style)
        
        # Aplicar tema al botón de tema
        self.theme_button.setStyleSheet(button_style)

    def update_theme_button(self):
        """Actualiza el texto del botón de tema según el tema actual"""
        if self.current_theme == DARK_THEME:
            self.theme_button.setText("☀️ Light Mode")
        else:
            self.theme_button.setText("🌙 Dark Mode")

    def get_hover_color(self, color):
        """Genera un color más claro para el hover"""
        # Convertir color hex a RGB y aclarar
        color = color.lstrip('#')
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        # Aclarar el color sumando 30 a cada componente
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def get_pressed_color(self, color):
        """Genera un color más oscuro para el press"""
        # Convertir color hex a RGB y oscurecer
        color = color.lstrip('#')
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        # Oscurecer el color restando 30 a cada componente
        r = max(0, r - 30)
        g = max(0, g - 30)
        b = max(0, b - 30)
        #return f"#{r:02x}{g:02x}{b:02x}"
        return f'#17d831'

    def add_character(self, char):
        if char == "ESP":
            char = " "
        elif char == "BORR":
            current = self.output_line.label.text()
            self.output_line.set_text(current[:-1])  # Eliminar el último carácter
            return
        self.output_line.add_character(char)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeyboardWindow()
    window.show()
    sys.exit(app.exec())
