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
from models.newTraining import config

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
        self.models_directory = config.MODELS_DIR
        self.model_categories = {
            "Modelo jerárquico completo": None,
            "Cascadas completas (Top 5)": "cascadas",
            "Top general": "general",
            "Top color": "color",
            "Top supergrupo": "super",
            "Top letras": "letras",
            "Top números": "numeros",
            "Top controles": "controles",
        }
        self.black_screen = BlackScreen(self.keyboard_window)
        self.setup_ui()
    def setup_ui(self):
        self.layout.addWidget(QLabel("Configuración del Speller tiempo Real"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(QLabel("Usuario :"), 1, 0)
        self.user_list = QComboBox()
        self.layout.addWidget(self.user_list, 1, 1)
        self.model_category = QComboBox()
        self.model_category.addItems(self.model_categories)
        self.layout.addWidget(QLabel("Tipo de modelo:"), 2, 0)
        self.layout.addWidget(self.model_category, 2, 1)
        self.models_available = QComboBox()
        self.layout.addWidget(QLabel("Modelo:"), 3, 0)
        self.layout.addWidget(self.models_available, 3, 1)
        self.user_list.currentIndexChanged.connect(self.load_models)
        self.model_category.currentIndexChanged.connect(self.load_models)
        self.load_users()
        self.stimulus_mode = QComboBox()
        self.stimulus_mode.addItem("Patrón de ajedrez (actual)", "chess")
        self.stimulus_mode.addItem("Teclado completo (una vez)", "all")
        self.layout.addWidget(QLabel("Estímulo:"), 4, 0)
        self.layout.addWidget(self.stimulus_mode, 4, 1)
        # self.save_button = QPushButton("Guardar Configuración")
        # self.save_button.clicked.connect(self.save_configuration)
        #self.layout.addWidget(self.save_button, 3, 0, 1, 2)
        self.start_capture_trial_button = QPushButton("Iniciar CapturaOnlinetiral")
        self.start_capture_trial_button.clicked.connect(self.start_capture_trial)
        self.layout.addWidget(self.start_capture_trial_button,5,0,1,2)
        self.hide()
    def save_configuration(self):
        self.user = self.user_list.currentText()
        self.model_path = self.models_available.currentData()
        if not self.predict_controller:
            QMessageBox.warning(
                self, "Predictor no disponible",
                "No se configuró el controlador de inferencia."
            )
            return False
        if not self.user or not self.model_path:
            QMessageBox.warning(
                self, "Configuración incompleta",
                "Selecciona un usuario y un modelo disponible antes de iniciar."
            )
            return False
        self.model_name = self.models_available.currentText()
        print(
            f"Guardando configuración: Usuario={self.user}, "
            f"Modelo={self.model_name}, Ruta={self.model_path}"
        )
        try:
            self.set_model_path(self.model_path)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "No se pudo cargar el modelo", str(exc))
            return False
        return True
    def load_models(self):
        self.models_available.clear()
        if os.path.exists(self.models_directory):
            users = [
                name for name in os.listdir(self.models_directory)
                if os.path.isdir(os.path.join(self.models_directory, name))
            ]
            user = self.user_list.currentText()
            if user and user in users:
                user_model_dir = os.path.join(self.models_directory, user)
                category = self.model_categories[
                    self.model_category.currentText()
                ]
                if category is None:
                    model_path = os.path.join(
                        user_model_dir, "hierarchical_bundle_optimizado.joblib"
                    )
                    if os.path.isfile(model_path):
                        self.models_available.addItem(
                            "Modelo jerárquico completo", model_path
                        )
                elif category == "cascadas":
                    category_dir = os.path.join(user_model_dir, "top_cascadas")
                    if os.path.isdir(category_dir):
                        models = sorted(
                            model for model in os.listdir(category_dir)
                            if model.startswith("hierarchical_bundle_optimizado_top_")
                            and model.endswith(".joblib")
                            and os.path.isfile(os.path.join(category_dir, model))
                        )
                        for model in models:
                            self.models_available.addItem(
                                model, os.path.join(category_dir, model)
                            )
                else:
                    category_dir = os.path.join(
                        user_model_dir, "top_modelos", category
                    )
                    if os.path.isdir(category_dir):
                        models = sorted(
                            model for model in os.listdir(category_dir)
                            if model.endswith(".joblib")
                            and os.path.isfile(os.path.join(category_dir, model))
                        )
                        for model in models:
                            self.models_available.addItem(
                                model, os.path.join(category_dir, model)
                            )
            else:
                print(f"No se encontró el usuario {user} en el directorio de modelos.")
        else:
            print(f"El directorio de modelos {self.models_directory} no existe.")
    def set_model_path(self, model_path):
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
        if not self.save_configuration():
            return
        if not self.save_capture_controller:
            QMessageBox.warning(
                self, "Captura no disponible",
                "No se configuró el controlador de captura."
            )
            return
        character_type = "recordings"
        character = "UNKNOWN"
        duration =2
        self.show_grid()
        self.new_record_path = self.save_capture_controller.start_capture(
            self.user, character_type, character, duration, callback=None, online=True
        )
        if self.stimulus_mode.currentData() == "all":
            QTimer.singleShot(100, self.start_full_keyboard_flash)
        else:
            QTimer.singleShot(100, self.start_chess_flashes)
        QTimer.singleShot(duration * 1000 + 1000, self.predict_character)

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
            try:
                caracter_predicho = self.predict_controller.predict(
                    self.user, self.new_record_path
                )
            except (OSError, ValueError, RuntimeError) as exc:
                QMessageBox.warning(self, "No se pudo predecir", str(exc))
                return
            print("Caracter predicho:", caracter_predicho)
            details = self.predict_controller.last_prediction_details
            if "categoria" in details:
                probabilities = details.get("probabilidades", {})
                probability_text = "\n".join(
                    f"{label}: {probability:.1%}"
                    for label, probability in probabilities.items()
                )
                QMessageBox.information(
                    self,
                    "Resultado de prueba del modelo",
                    (
                        f"Modelo: {details['categoria']} #{details['rank']}\n"
                        f"Etapa: {details['nivel']} | Grupo: {details['grupo']}\n"
                        f"Clasificador: {details['clasificador']}\n"
                        f"F1 macro CV: {details['f1_macro_cv']:.3f}\n"
                        f"Predicción: {caracter_predicho}\n\n"
                        f"Probabilidades:\n{probability_text}"
                    ),
                )
                return
            if "color_predicho" in details:
                selected_model = self.predict_controller.model
                cascade_rank = (
                    selected_model.get("modelo_top_rank")
                    if isinstance(selected_model, dict) else None
                )
                QMessageBox.information(
                    self,
                    "Resultado de prueba de la cascada",
                    (
                        f"Modelo: cascada completa top {cascade_rank or '?'}\n"
                        f"Color: {details['color_predicho']}\n"
                        f"Supergrupo: {details['grupo_predicho']}\n"
                        f"Comando: {caracter_predicho}"
                    ),
                )
        if self.keyboard_window:
            self.keyboard_window.add_character(caracter_predicho)
    def start_chess_flashes(self):
        self.keyboard_window.start_paradigm(times=2)

    def start_full_keyboard_flash(self):
        self.keyboard_window.flash_all_buttons()


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