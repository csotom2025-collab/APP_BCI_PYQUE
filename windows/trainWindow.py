import os
import sys
from PyQt6.QtWidgets import QApplication, QComboBox, QGridLayout, QLineEdit, QMainWindow, QPushButton, QWidget, QVBoxLayout, QLabel, QMessageBox

from controllers.TrainingController import controllerTraining





class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        # self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        # self.show_users()
        self.models= ['lstm', 'cnn', 'svm', 'random_forest', 'xgboost']
        self.show_models()
        self.show_users()
        
    def setup_ui(self):
        self.setWindowTitle("Ventana de Entrenamiento")
        layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.user_combobox = QComboBox()

        self.combo_box_models = QComboBox()
        self.button_start_training = QPushButton("Iniciar entrenamiento")
        self.button_start_training.clicked.connect(self.start_training)

        layout.addWidget(QLabel("Ventana de Entrenamiento"))
        self.grid_layout.addWidget(QLabel("Seleccione usuario:"), 0, 0)
        self.grid_layout.addWidget(self.user_combobox, 0, 1)
        self.grid_layout.addWidget(QLabel("Selecciona un modelo:"), 2, 0)
        self.grid_layout.addWidget(self.combo_box_models, 2, 1)
        self.grid_layout.addWidget(self.button_start_training, 3, 0, 1, 3)


        layout.addLayout(self.grid_layout)
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)
        self.move(200, 200)

    

    

    def show_models(self):
        self.combo_box_models.clear()
        self.combo_box_models.addItems(self.models)

    def get_model(self):
        model = self.combo_box_models.currentText()
        return model
    
    def get_user(self):
        user_id = self.user_combobox.currentText()
        return user_id
    
    def get_path(self):
        user = self.get_user()
        if not user:
            return ""
        return os.path.join("captures", f"{user}")
    def show_users(self):
        self.user_combobox.clear()
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            QMessageBox.warning(self, "Directorio no encontrado", f"No se encontró el directorio '{captures_dir}'.")
            return

        users = [d for d in os.listdir(captures_dir) if os.path.isdir(os.path.join(captures_dir, d)) and d.startswith("User")]
        if not users:
            QMessageBox.information(self, "Usuarios no encontrados", "No se encontraron carpetas de usuarios en el directorio 'captures'.")
            return

        self.user_combobox.addItems(users)
    def start_training(self):    
        user = self.get_user()
        path = self.get_path()
        model = self.get_model()
        print(f"Entrenando modelo {model} con los datos de {user} que se encuentran en la ruta: {path}")
        training_controller = controllerTraining()
        training_controller.train_model(user, path, "LDA")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainWindow()
    window.show()
    sys.exit(app.exec())