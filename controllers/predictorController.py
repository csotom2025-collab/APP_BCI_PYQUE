

import os


class PredictorController:
    def __init__(self, model=None):
        self.model = model
        self.model = model

    def predict(self, path):
        file_path = path
        if not os.path.exists(file_path):
            print(f"Error: El archivo no se encuentra en la ruta {file_path}")
            return
        print(f"Prediciendo con el modelo {self.model} usando el archivo {file_path}")

    def set_model_path(self, model_path):
        if not os.path.exists(model_path):
            print(f"Error: El modelo no se encuentra en la ruta {model_path}")
            return
        self.model.load(model_path)
        print(f"Modelo cargado desde {model_path}")