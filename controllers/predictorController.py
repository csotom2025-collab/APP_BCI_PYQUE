import os
import sys
import pandas as pd
import joblib

import models.newTraining.lda_utils as lda_utils_module
from models.newTraining.lda_utils import SafeLDA
from models.oldTraining.pipeline_completo_lda import PipelineCompletoLDA
from models.newTraining.BCIpredictorhierarchical import HierarchicalBCIPredictor
sys.modules["lda_utils"] = lda_utils_module

class PredictorController:
    def __init__(self, model=None):
        self.model = model
        self.bci_predictor = None
        self.model_path = None
        self.last_prediction_details = {}


    def predict(self,user, recording_path):
        file_path = recording_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"El archivo de grabación no se encuentra en la ruta {file_path}"
            )
        print(f"Prediciendo con el modelo {self.model_path} usando el archivo {file_path}")
        if (
            isinstance(self.model, dict)
            and "pipeline" in self.model
            and "label_encoder" in self.model
            and "feature_columns" in self.model
            and "categoria" in self.model
        ):
            self.bci_predictor = HierarchicalBCIPredictor(usuario=user)
            prediction, self.last_prediction_details = (
                self.bci_predictor.predict_with_model_artifact_from_csv(
                    file_path, self.model
                )
            )
            return prediction

        self.bci_predictor = HierarchicalBCIPredictor(
            usuario=user, model_path=self.model_path
        )
        prediction, self.last_prediction_details = (
            self.bci_predictor.predict_from_csv(file_path)
        )
        return prediction

    def set_model_path(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encuentra en la ruta {model_path}")
        self.model_path = model_path
        self.model = joblib.load(model_path)
        print(f"Modelo cargado desde {model_path}")


    def get_prediction(self,path_file):
        file = pd.read_csv(path_file)
        print("archivio_entreado",file.shape)
        pipeline_lda = PipelineCompletoLDA(base_output_dir="onlineCaptures/"+ "UserJorge")
        separado,trial =pipeline_lda.separar_archivo_por_ruta(path_file, usuario="UserJorge",unknown=True)
        print("separado",separado)
        df = pipeline_lda.extraer_caracteristicas_file(separado,usuario="UserJorge",subcarpeta="Unknown",letra='UNK',trial=trial,save_output=True,online=True)
        df['label'] = 'label'
        print("df",df.shape)
        print(df)
        print("optimizando_lda")
        resultados = pipeline_lda.optimizar_lda_caracteristicas(usuario="UserJorge",caracteristicas=df,label_col='label')
        print(resultados)

        return
    def get_predictionBCIHierarchichal(self,path_file):
        comando, detalle = self.bci_predictor.predict_from_csv(path_file)
        print("Predicción obtenida del archivo:", path_file)
        print(comando)
        return comando