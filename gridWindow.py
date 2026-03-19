import sys
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit
from PyQt6.QtCore import QTimer

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
CONTROLS = ["ESP", "BORR", "ENTER"]


class OutputLine(QWidget):
    def __init__(self, text):
        super().__init__()
        self.layout = QHBoxLayout()
        self.label = QLineEdit(text)
        self.label.setReadOnly(False)
        self.label.setFixedSize(400, 30)
        self.layout.addWidget(self.label)
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
        self.setup_ui()

    def setup_ui(self):
        self.grid_letters_layout = QGridLayout()
        self.grid_numbers_layout = QGridLayout()
        self.layout = QGridLayout()
        self.grid_controls_layout = QGridLayout()
        self.setLayout(self.layout)
        self.output_line = OutputLine("")
        self.show_grid()
        self.layout.addWidget(self.output_line, 0, 0, 1, 2)
        self.layout.addLayout(self.grid_letters_layout, 1, 0)
        self.layout.addLayout(self.grid_numbers_layout, 1, 1)
        self.layout.addLayout(self.grid_controls_layout, 2, 0)

    def show_grid(self):
        # Create 3x3 grid using  loops
        for row, letters in enumerate(GRID):
            for col, letter in enumerate(letters):
                button = QPushButton(f"{letter}")
                button.setFixedSize(80, 80)  # Set a fixed size for better appearance
                button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
                self.grid_letters_layout.addWidget(button, row, col)
        for row, numbers in enumerate(GrindNumbers):
            for col, number in enumerate(numbers):
                button = QPushButton(f"{number}")
                button.setFixedSize(80, 80)  # Set a fixed size for better appearance
                button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
                self.grid_numbers_layout.addWidget(button, row, col)
        for col, control in enumerate(CONTROLS):
            button = QPushButton(control)
            button.setFixedSize(50, 20)  # Set a fixed size for better appearance
            button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
            self.grid_controls_layout.addWidget(button, 0, col)

    def button_clicked(self, button):
        print("Boton seleccionado :" + button.text())
        self.flash_button(button)
        self.add_character(button.text())

    def flash_button(self, button, duration=0.5):
        # Guardar el stylesheet original del botón
        original_style = button.styleSheet()
        button.setStyleSheet(f"{original_style} background-color: yellow;")
        # Restaurar completamente el estilo original después del flash
        QTimer.singleShot(int(duration * 1000), lambda: button.setStyleSheet(original_style))

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
