import os
import sys
from PyQt6.QtWidgets import QApplication, QComboBox, QGridLayout, QLineEdit, QMainWindow, QPushButton, QWidget, QVBoxLayout, QLabel, QMessageBox





class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        # self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        # self.show_users()
        self.models= ['lstm', 'cnn', 'svm', 'random_forest', 'xgboost']
        self.show_models()
        
    def setup_ui(self):
        self.setWindowTitle("Ventana de Entrenamiento")
        layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.pathLine = QLineEdit("path")
        self.id_user_line = QLineEdit()

        self.combo_box_models = QComboBox()
        self.button_start_training = QPushButton("Iniciar entrenamiento")
        self.button_start_training.clicked.connect(self.start_training)
        self.button_verify_path = QPushButton("Verificar path")
        self.button_verify_path.clicked.connect(self.verify_path)

        layout.addWidget(QLabel("Ventana de Entrenamiento"))
        self.grid_layout.addWidget(QLabel("Id del usuario:"), 0, 0)
        self.grid_layout.addWidget(self.id_user_line, 0, 1)
        self.grid_layout.addWidget(self.button_verify_path, 0, 2)
        self.grid_layout.addWidget(QLabel("Ruta del usuario seleccionado:"), 1, 0)
        self.grid_layout.addWidget(self.pathLine, 1, 1, 1, 2)
        self.grid_layout.addWidget(QLabel("Selecciona un modelo:"), 2, 0)
        self.grid_layout.addWidget(self.combo_box_models, 2, 1)
        self.grid_layout.addWidget(self.button_start_training, 3, 0, 1, 3)


        layout.addLayout(self.grid_layout)
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)
        self.move(200, 200)

    
    def show_path(self):
        user = self.get_user()
        path = os.path.join("captures", f"User{user}")
        self.pathLine.setText(path)

    def verify_path(self):
        user = self.get_user().strip()
        if not user:
            QMessageBox.warning(self, "Usuario requerido", "Ingrese el ID del usuario antes de verificar.")
            return

        path = os.path.join("captures", f"User{user}")
        if os.path.isdir(path):
            self.pathLine.setText(path)
            QMessageBox.information(self, "Ruta encontrada", f"La carpeta del usuario existe:\n{path}")
        else:
            self.pathLine.setText("")
            QMessageBox.warning(self, "Ruta no existe", f"No existe la carpeta del usuario:\n{path}")

    def show_models(self):
        self.combo_box_models.clear()
        self.combo_box_models.addItems(self.models)

    def get_model(self):
        model = self.combo_box_models.currentText()
        return model
    
    def get_user(self):
        user_id = self.id_user_line.text()
        return user_id
    
    def get_path(self):
        user = self.get_user().strip()
        if not user:
            return ""
        return os.path.join("captures", f"User{user}")
    
    def start_training(self):    
        user = self.get_user()
        path = self.get_path()
        model = self.get_model()
        print(f"Entrenando modelo {model} con los datos de {user} que se encuentran en la ruta: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainWindow()
    window.show()
    sys.exit(app.exec())