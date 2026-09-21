import os
import pandas as pd
import joblib

from models.pipeline_completo_lda import PipelineCompletoLDA
class PredictorController:
    def __init__(self, model=None):
        self.model = model
        
    def predict(self, path):
        file_path = path
        if not os.path.exists(file_path):
            print(f"Error: El archivo no se encuentra en la ruta {file_path}")
            return
        if not self.model:
            print("Error: No se ha cargado ningún modelo para la predicción.")
            return
        print(f"Prediciendo con el modelo {self.model} usando el archivo {file_path}")
        self.get_prediction(file_path)

    def set_model_path(self, model_path):
        if not os.path.exists(model_path):
            print(f"Error: El modelo no se encuentra en la ruta {model_path}")
            return
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