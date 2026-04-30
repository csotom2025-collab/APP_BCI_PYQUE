import sys
import random
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit, QSizePolicy
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QFont

GRID = [
    ["A", "E", "I", "O", "U","1","2","3"],
    ["S", "R", "N", "L", "D","4","5","6"],
    ["C", "T", "M", "P", "B","7","8","9"],
    ["G", "V", "Y", "Q", "H","0","───", "⟵"],
    ["F", "Z", "J", "Ñ", "X","K", "W","↩"]
]

class KeyboardWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("P300 Speller Paradigm")
        self.resize(800, 600)
        
        # --- Configuración del Paradigma ---
        self.num_epocas = 3  # Cuántas veces se repite el ciclo completo
        self.flash_duration = 120  # ms iluminado
        self.isi_duration = 80    # ms apagado (Inter-Stimulus Interval)
        
        # Estado interno
        self.current_epoca = 0
        self.sequence = [] # Lista de tuplas ('row', index) o ('col', index)
        self.buttons_matrix = [] # Para acceder fácil por [fila][col]
        
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        
        # Input display
        self.output_line = QLineEdit()
        self.output_line.setReadOnly(True)
        self.output_line.setFixedHeight(50)
        self.output_line.setFont(QFont("Arial", 18))
        self.main_layout.addWidget(self.output_line)

        # Grid de botones
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        
        for r, row_data in enumerate(GRID):
            button_row = []
            for c, char in enumerate(row_data):
                btn = QPushButton(char)
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
                btn.setStyleSheet(self.get_normal_style())
                self.grid_layout.addWidget(btn, r, c)
                button_row.append(btn)
            self.buttons_matrix.append(button_row)
        
        self.main_layout.addLayout(self.grid_layout)

        # Botón de control
        self.start_button = QPushButton("INICIAR ESTIMULACIÓN")
        self.start_button.setFixedHeight(50)
        self.start_button.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.start_paradigm)
        self.main_layout.addWidget(self.start_button)

    def get_normal_style(self):
        return "QPushButton { background-color: #111111; color: white; border: 1px solid #444; border-radius: 5px; }"

    def get_highlight_style(self):
        # Estilo para cuando la fila/columna se "ilumina"
        return "QPushButton { background-color: #ffffff; color: #000000; border: 1px solid #ffffff; }"

    def start_paradigm(self):
        """Prepara e inicia el ciclo de épocas"""
        self.start_button.setEnabled(False)
        self.current_epoca = 0
        self.prepare_epoca()

    def prepare_epoca(self):
        """Crea una secuencia aleatoria de todas las filas y todas las columnas"""
        if self.current_epoca < self.num_epocas:
            print(f"Iniciando Época {self.current_epoca + 1}")
            
            filas = [('row', i) for i in range(len(GRID))]
            columnas = [('col', i) for i in range(len(GRID[0]))]
            
            # Combinar y barajar aleatoriamente
            self.sequence = filas + columnas
            random.shuffle(self.sequence)
            
            self.run_next_flash()
        else:
            print("Paradigma finalizado.")
            self.start_button.setEnabled(True)

    def run_next_flash(self):
        """Toma el siguiente elemento de la secuencia y lo ilumina"""
        if not self.sequence:
            # Si terminó la secuencia, vamos a la siguiente época
            self.current_epoca += 1
            QTimer.singleShot(500, self.prepare_epoca)
            return

        type, index = self.sequence.pop(0)
        self.set_highlight(type, index, True)
        
        # Timer para apagar el flash
        QTimer.singleShot(self.flash_duration, lambda: self.end_flash(type, index))

    def end_flash(self, type, index):
        """Apaga el flash y espera el ISI (tiempo de oscuridad) antes del siguiente"""
        self.set_highlight(type, index, False)
        QTimer.singleShot(self.isi_duration, self.run_next_flash)

    def set_highlight(self, type, index, active):
        """Ilumina u oscurece una fila o columna completa"""
        style = self.get_highlight_style() if active else self.get_normal_style()
        
        if type == 'row':
            for col in range(len(GRID[0])):
                self.buttons_matrix[index][col].setStyleSheet(style)
        else: # col
            for row in range(len(GRID)):
                self.buttons_matrix[row][index].setStyleSheet(style)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeyboardWindow()
    window.show()
    sys.exit(app.exec())