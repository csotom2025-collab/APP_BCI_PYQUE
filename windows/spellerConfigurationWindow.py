import sys
import json
import os
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget, QPushButton, QGridLayout, QLineEdit, QComboBox
from PyQt6.QtCore import QTimer, Qt, QRect
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtCore import QPoint
from windows.gridWindow import KeyboardWindow
from controllers.SaveCaptureController import controllerSaveCapture
from controllers.predictorController import PredictorController
from windows.gridWindow import BlackScreen
from models.newTraining.lda_utils import SafeLDA

class SpellerConfigurationWindow(QWidget):
    def __init__(self, predict_controller:PredictorController=None, save_capture_controller:controllerSaveCapture=None, keyboard_window:KeyboardWindow=None):
        super().__init__()
        self.setWindowTitle("Configuración del Speller")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.move(100, 150)
        self.predict_controller = predict_controller
        self.save_capture_controller = save_capture_controller
        self.keyboard_window = keyboard_window
        self.models_directory = "./trainingOutputs/models"
        self.black_screen = BlackScreen(self.keyboard_window)
        self.setup_ui()
    def setup_ui(self):
        self.layout.addWidget(QLabel("Configuración del Speller tiempo Real"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(QLabel("Usuario :"), 1, 0)
        self.user_list = QComboBox()
        self.layout.addWidget(self.user_list, 1, 1)
        self.models_available = QComboBox()
        self.layout.addWidget(QLabel("Modelo:"), 2, 0)
        self.layout.addWidget(self.models_available, 2, 1)
        self.user_list.currentIndexChanged.connect(self.load_models)
        self.load_users()
        # self.save_button = QPushButton("Guardar Configuración")
        # self.save_button.clicked.connect(self.save_configuration)
        #self.layout.addWidget(self.save_button, 3, 0, 1, 2)
        self.start_capture_trial_button = QPushButton("Iniciar CapturaOnlinetiral")
        self.start_capture_trial_button.clicked.connect(self.start_capture_trial)
        self.layout.addWidget(self.start_capture_trial_button,4,0,1,2)
        self.hide()
    def save_configuration(self):
        self.user = self.user_list.currentText()
        self.model_name = self.models_available.currentText()
        self.model_path = self.models_directory + "/" + self.user + "/" + self.model_name
        print(f"Guardadno Configuración: Usuario={self.user}, Modelo={self.model_name}, Ruta={self.model_path}")
        self.set_model_path(self.model_path)
    def load_models(self):
        self.models_available.clear()
        if os.path.exists(self.models_directory):
            users = [name for name in os.listdir(self.models_directory) if os.path.isdir(os.path.join(self.models_directory, name))]
            #print("Usuarios encontrados en el directorio de modelos:", users)
            user =self.user_list.currentText()
            if user and user in users:
                user_model_dir = os.path.join(self.models_directory, user)
                models = [model for model in os.listdir(user_model_dir) if model.endswith(".joblib")]
                self.models_available.addItems(models)
            else:
                print(f"No se encontró el usuario {user} en el directorio de modelos.")
        else:
            print(f"El directorio de modelos {self.models_directory} no existe.")
    def set_model_path(self, model_path):
        print("modelskjfa",model_path)
        self.predict_controller.set_model_path(model_path)
    #resultsALL\UserJorge\Resultados_Clasificadores\LDA_General\modeloos\LDA_General_Estadisticas_Red_Neuronal_MLP.joblib
    def load_users(self,users=None):
        if not users:
            users_directory = "captures"
            if os.path.exists(users_directory):
                users = [name for name in os.listdir(users_directory) if os.path.isdir(os.path.join(users_directory, name))]
                self.user_list.addItems(users)
            else:
                print(f"El directorio {users_directory} no existe.")
        else:
            self.user_list.addItems(users)
    def start_capture_trial(self):
        self.save_configuration()
        character_type = "recordings"
        character = "UNKNOWN"
        duration =2
        self.show_grid()
        self.start_chess_flashes()
        if self.save_capture_controller:
            self.new_record_path = self.save_capture_controller.start_capture(self.user, character_type, character, duration,callback=None,online=True)
        qtime = QTimer()
        qtime.singleShot(2*1000+900, self.predict_character)

    def show_grid(self):
        if not self.isBlackScreenVisible() and self.black_screen:
                self.black_screen.close()
                self.black_screen = None
                self.keyboard_window = None
        if not self.keyboard_window:
            self.keyboard_window = KeyboardWindow(training_mode=False)
            # self.keyboard_window.show()  # No mostrar por separado, ahora está en black_screen
            self.black_screen = BlackScreen(self.keyboard_window)
            self.black_screen.show()
        else:
            self.show_grid_after_rest()
        

    def show_grid_after_rest(self):
        if not self.keyboard_window or not self.isBlackScreenVisible():
            print("No hay ventana de teclado, mostrando mensaje de error")
            QMessageBox.information(self, "Advertencia, No Hay teclado", "Primero debes mostrar el teclado para iniciar la captura de datos")
            return False
        self.keyboard_window.show_grid_after_rest()
        return True

    def hide_grid(self):
        if self.keyboard_window:
            self.keyboard_window.hide_grid()
    def isBlackScreenVisible(self):
        return self.black_screen and self.black_screen.isVisible()
    def predict_character(self):
        caracter_predicho = "None"
        if self.predict_controller:
            caracter_predicho = self.predict_controller.predict(self.user,self.new_record_path)
            print("Caracter predicho:", caracter_predicho)
        self.keyboard_window.add_character(caracter_predicho)
    def start_chess_flashes(self):
        self.keyboard_window.start_paradigm(times=2)


    def closeEvent(self, event: QCloseEvent):
        """Cierra todas las ventanas secundarias y la aplicación."""
        #Acepta el evento de cierre de la ventana principal
        event.accept()
        if self.keyboard_window:
            self.keyboard_window.close()
            self.black_screen.close()
    def quit(self):
        self.close()
        if self.keyboard_window:
            self.keyboard_window.close()
        if self.black_screen:
            self.black_screen.close()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellerConfigurationWindow()
    window.show()
    sys.exit(app.exec())