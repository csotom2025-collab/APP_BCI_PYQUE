import sys
from PyQt6.QtWidgets import QApplication, QComboBox, QGridLayout, QLineEdit, QMainWindow, QPushButton, QWidget, QVBoxLayout, QLabel

from InterfasPyQue import SignalsWindow
from gridWindow import GridButtonWindow



class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ventana de Entrenamiento")
        layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.pathLine = QLineEdit("path")
        self.combo_box_users = QComboBox()
        self.combo_box_users.currentIndexChanged.connect(self.show_path)
        self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        self.show_users()


        self.combo_box_models = QComboBox()
        self.models= ['lstm', 'cnn', 'svm', 'random_forest', 'xgboost']
        self.show_models()

        self.button_start_training = QPushButton("Iniciar entrenamiento")
        self.button_start_training.clicked.connect(self.start_training)

        layout.addWidget(QLabel("Ventana de Entrenamiento"))
        self.grid_layout.addWidget(QLabel("Selecciona un usuario:"), 0, 0)
        self.grid_layout.addWidget(self.combo_box_users, 0, 1)
        self.grid_layout.addWidget(QLabel("Ruta del usuario seleccionado:"), 1, 0)
        self.grid_layout.addWidget(self.pathLine, 1, 1)
        self.grid_layout.addWidget(QLabel("Selecciona un modelo:"), 2, 0)
        self.grid_layout.addWidget(self.combo_box_models, 2, 1)
        self.grid_layout.addWidget(self.button_start_training, 3, 0, 1, 2)


        layout.addLayout(self.grid_layout)
        self.setLayout(layout)
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)
        self.move(200, 200)

    def show_users(self):
        self.combo_box_users.clear()
        self.combo_box_users.addItems(self.data.keys())
    
    def show_path(self):
        user = self.combo_box_users.currentText()
        path = self.data.get(user, "No path found")
        self.pathLine.setText(path)
    def show_models(self):
        self.combo_box_models.clear()
        self.combo_box_models.addItems(self.models)
    def get_model(self):
        model = self.combo_box_models.currentText()
        return model
    
    def get_user(self):
        user = self.combo_box_users.currentText()
        return user
    
    def get_path(self):
        user = self.get_user()
        path = self.data.get(user)
        return path
    
    def start_training(self):
        # self.window_signals = SignalsWindow()
        # self.window_signals.show()
        # self.gridButtonWindow = GridButtonWindow()
        # self.gridButtonWindow.show()    
        user = self.get_user()
        path = self.get_path()
        model = self.get_model()
        print(f"Entrenando modelo {model} con los datos de {user} que se encuentran en la ruta: {path}")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = TrainWindow()
#     window.show()
#     sys.exit(app.exec())