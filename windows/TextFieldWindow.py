from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

class TextField(QWidget):
    def __init__(self, text):
        super().__init__()
        self.layout = QHBoxLayout()
        self.label = QLineEdit(text)
        self.label.setReadOnly(False)
        self.controller = None
        self.button_next_char = QPushButton("Sugerir siguiente carácter")
        self.button_next_char.clicked.connect(self.predict_next_character)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button_next_char)
        self.setLayout(self.layout)
        self.hide()  # Oculta la ventana del TextField inicialmente
    def set_controller(self, controller):
        self.controller = controller
    def set_text(self, text):
        self.label.setText(text)
    def add_character(self, char):
        current = self.label.text()
        self.label.setText(current + char)
    def add_new_line(self):
        current = self.label.text()
        self.label.setText(current + "\n")
    def remove_last_character(self):
        current = self.label.text()
        self.label.setText(current[:-1])
    def space(self):
        current = self.label.text()
        self.label.setText(current + " ")
    def predict_next_character(self):
        if self.controller:
            self.controller.predict_next_character()