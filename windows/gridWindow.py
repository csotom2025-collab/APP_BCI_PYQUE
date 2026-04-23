import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit
from PyQt6.QtCore import QTimer, Qt, QRect
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QPoint

GRID = [
    ["A", "E", "I", "O", "U","1","2","3"],
    ["S", "R", "N", "L", "D","4","5","6"],
    ["C", "T", "M", "P", "B","7","8","9"],
    ["G", "V", "Y", "Q", "H","0","───", "⟵"],
    ["F", "Z", "J", "Ñ", "X","K ", "W","↩"]
    
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
    "window_bg": "#000000",
    "button_bg": "#111111",
    "button_text": "#ffffff",
    "button_border": "#444444",
    "input_bg": "#111111",
    "input_text": "#ffffff",
    "theme_name": "Dark"
}


class BlackScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pantalla Negra")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: black;")
        
        # Variables para el arrastre
        self.dragging = False
        self.drag_position = QPoint()
        self.is_maximized = False
        self.saved_geometry = None
        
        # Crear layout principal
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Crear barra de título personalizada
        self.title_bar = self.create_title_bar()
        self.main_layout.addWidget(self.title_bar)
        
        # Widget central para el contenido
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.central_widget)
        
        self.setLayout(self.main_layout)
        
        # Timer para esconder la barra de título
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide_title_bar)
        self.hide_timer.setSingleShot(True)
        
        # Mostrar la barra inicialmente
        self.show_title_bar()
        
        # Rastrear posición previa del mouse
        self.last_mouse_y = 0
        self.mouse_tracking_enabled = True
        self.setMouseTracking(True)
        
    def create_title_bar(self):
        """Crea la barra de título personalizada"""
        title_bar = QWidget()
        title_bar.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border-bottom: 1px solid #555555;
            }
        """)
        title_bar.setMaximumHeight(40)
        title_bar.setMinimumHeight(40)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        
        # Etiqueta del título (arrastrable)
        title_label = QLabel("Pantalla Negra")
        title_label.setStyleSheet("color: white; font-weight: bold;")
        title_label.mousePressEvent = self.title_bar_mouse_press
        title_label.mouseMoveEvent = self.title_bar_mouse_move
        title_label.mouseReleaseEvent = self.title_bar_mouse_release
        layout.addWidget(title_label)
        
        # Espacio flexible
        layout.addStretch()
        
        # Botón maximizar
        maximize_btn = QPushButton("□")
        maximize_btn.setMaximumWidth(40)
        maximize_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        maximize_btn.clicked.connect(self.toggle_maximize)
        layout.addWidget(maximize_btn)
        
        # Botón minimizar
        minimize_btn = QPushButton("−")
        minimize_btn.setMaximumWidth(40)
        minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        minimize_btn.clicked.connect(self.showMinimized)
        layout.addWidget(minimize_btn)
        
        # Botón cerrar
        close_btn = QPushButton("✕")
        close_btn.setMaximumWidth(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        title_bar.setLayout(layout)
        return title_bar
    
    def show_title_bar(self):
        """Muestra la barra de título"""
        self.title_bar.show()
        # Reinicia el timer para esconder la barra después de 3 segundos
        self.hide_timer.stop()
        self.hide_timer.start(3000)
    
    def hide_title_bar(self):
        """Oculta la barra de título"""
        self.title_bar.hide()
    
    def mouseMoveEvent(self, event):
        """Detecta movimiento del mouse hacia arriba"""
        current_y = event.globalPosition().y()
        screen_rect = self.geometry()
        
        # Si el mouse está en los primeros 50 píxeles de la pantalla o se mueve hacia arriba
        if current_y - screen_rect.y() < 50:
            self.show_title_bar()
        
        self.last_mouse_y = current_y
        super().mouseMoveEvent(event)
    
    def leaveEvent(self, event):
        """Cuando el mouse sale de la ventana"""
        super().leaveEvent(event)
    
    def title_bar_mouse_press(self, event):
        """Detecta el clic en la barra de título para iniciar arrastre"""
        if not self.is_maximized:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.pos()
    
    def title_bar_mouse_move(self, event):
        """Arrastra la ventana con el mouse"""
        if self.dragging and not self.is_maximized:
            self.move(event.globalPosition().toPoint() - self.drag_position)
    
    def title_bar_mouse_release(self, event):
        """Finaliza el arrastre"""
        self.dragging = False
    
    def toggle_maximize(self):
        """Alterna entre modo maximizado y modo ventana"""
        if self.is_maximized:
            # Restaurar a la geometría guardada
            if self.saved_geometry:
                self.setGeometry(self.saved_geometry)
            self.is_maximized = False
        else:
            # Guardar geometría actual y maximizar
            self.saved_geometry = self.geometry()
            screen = self.screen().geometry()
            self.setGeometry(screen)
            self.is_maximized = True

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
    def __init__(self, training_mode=False):
        super().__init__()
        if  training_mode:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Grid de botones")
        dpi = self.screen().physicalDotsPerInch()
        pixels_per_cm = dpi / 2.54
        size_pixels = int(10  * pixels_per_cm)
        self.setFixedSize(size_pixels, size_pixels)
        self.current_theme = self.load_theme()
        self.training_mode = training_mode   
        self.original_font = QFont()
        self.original_font.setPointSize(16)
        self.original_font.setBold(True)
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
        if not self.training_mode:
            self.layout.addWidget(self.output_line, 0, 0, 1, 3)
        self.layout.addLayout(self.grid_letters_layout, 1, 0, 5, 3)
        #self.layout.addLayout(self.grid_numbers_layout, 1, 1)
        # Configurar el stretch de filas y columnas
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 5)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setSpacing(5)
        
        ##self.main_layout.addLayout(theme_layout)
        self.main_layout.addLayout(self.layout)
        self.setLayout(self.main_layout)

    def show_grid(self):
        # Crear fuente para los botones
        # font = QFont()
        # font.setPointSize(16)
        # font.setBold(True)
        
        # Create grid using loops
        for row, letters in enumerate(GRID):
            for col, letter in enumerate(letters):
                button = QPushButton(f"{letter}",)
                button.setFont(self.original_font)
                # Usar política de tamaño en lugar de tamaño fijo
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                button.setMinimumSize(40, 40)
                button.clicked.connect(lambda checked, b=button: self.button_clicked(b))
                self.grid_letters_layout.addWidget(button, row, col)
                self.buttons_list.append(button)
        
        
        # Configurar stretch para los layouts internos
        for i in range(len(GRID)):
            self.grid_letters_layout.setRowStretch(i, 1)
        for j in range(len(GRID[0])):
            self.grid_letters_layout.setColumnStretch(j, 1)
        
       

    def button_clicked(self, button):
        print("Boton seleccionado :" + button.text())
        self.flash_button(button)
        if button.text() == "───":
            self.add_character(" ")
        elif button.text() == "⟵":
            #quitar el último carácter en la salida
            current = self.output_line.label.text()
            self.output_line.set_text(current[:-1])
        elif button.text() == "↩":
            self.add_character("\n")
        else:
            self.add_character(button.text())

    def flash_button(self, button, duration=1.0):
        # Aplicar efecto de flash amarillo
        button.setStyleSheet(
            f"QPushButton {{ background-color: #00FF00; "
            f"color: black; border: 2px solid #cccccc; "
            f"border-radius: 5px; font-weight: bold; }}"
        )
        # Aumentar el tamaño de fuente en 20%
        original_font = button.font()
        larger_font = QFont(original_font)
        larger_font.setPointSize(int(original_font.pointSize() * 1.16))
        button.setFont(larger_font)
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
        button.setFont(self.original_font)

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
