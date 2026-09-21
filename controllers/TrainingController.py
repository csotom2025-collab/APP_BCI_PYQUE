import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline_completo_lda import PipelineCompletoLDA

class controllerTraining():
    def __init__(self):
        pass

    def train_model(self,user,dataPath,modelType):
        if not os.path.exists(dataPath):
            print("no existe la ruta")
            return
        if modelType== "LDA":
            print("lda")
            self.pipeline_lda = PipelineCompletoLDA(base_output_dir="resultsALL/")
            self.pipeline_lda.separar_archivos_csv(usuario=user)
            print("SE SEPARARON LOS ARCHIVOS ")
            print("Obteniendo caracteristicas")
            self.pipeline_lda.obtener_caracteristicas_usuario(usuario=user)
            print("EXTRAIDAS TODAS LAS CARACTERISTICAS")
            print("Optimizando LDA")
            self.pipeline_lda.optimizacion_lda(user=user)
            print("Entrenado Modelo")
            #self.pipeline_lda.entrenar_modelo(user, tipo='LDA_General', grupo='TODAS', modelo='ALL',save_model=True,model_format='joblib')
            self.pipeline_lda.entrenar_todos_los_modelos(usuario=user, save_plots=True, show_plots=True,save_models=True,model_format = 'joblib')
            
            print("LDA")
        if modelType == "CNN":
            print("cnn")
        #df = self.create_df(dataPath)


    def save_model(self,path):
        pass

    def get_label(self,file_name:str):
        label =file_name.split('_')[1]
        return label

    def generate_data_set_by_files(self,path,files):
        """
            generamos el dataset de los archivos ,separados por canales y con su label
        """
        sample_count = 250
        csv_files = [file_name for file_name in files if file_name.endswith('.csv')]
        if not csv_files:
            return {}

        columnas = pd.read_csv(path + f'/{csv_files[0]}').columns # obtenemos los headers del archivo los cuales son los canales
        print("Columnas Archivo" , columnas)
        labels_dict={}
        files_dataset = []
        skipped_files = []
        for file_name in csv_files:
            label = self.get_label(file_name)
            # print(f"Processing file: {path + f'/{file_name}'}")

            file = pd.read_csv(path + f'/{file_name}')
            if len(file) < sample_count:
                skipped_files.append((file_name, len(file)))
                continue

            file_channels =[]
            for channel in columnas[1:]: # no nos importa el Tm o si ???
                file_channels.append(np.array(file[channel].iloc[:sample_count]))
            file_channels = np.array(file_channels)
            files_dataset.append(file_channels)
            if label not in labels_dict:
                labels_dict[label]= [file_channels]
            else:
                labels_dict[label].append(file_channels)
            #print(label)
        print(f"Archivos validos: {len(files_dataset)}")
        if skipped_files:
            print(f"Archivos omitidos por tener menos de {sample_count} muestras: {len(skipped_files)}")
            for file_name, row_count in skipped_files:
                print(f"  {file_name}: {row_count} muestras")

        files_dataset = np.array(files_dataset)
        datasets_by_label = {}
        for idx, (label, label_files) in enumerate(labels_dict.items()):
            datasets_by_label[label] = np.array(label_files)
            print(f"idx {idx} : {label} -> {datasets_by_label[label].shape}")

        return datasets_by_label
        

    def create_df(self,path):

        """
        crear 3 datasets por cada clasificacion y uno general
        """
        directories = os.listdir(path)
        letters_path = path + "/Letters"
        numbers_path =path + "/Numbers"
        controls_path =path + "/Controls"
        print(letters_path)
        if os.path.exists(letters_path):
            letters_files = os.listdir(letters_path)
        if os.path.exists(numbers_path):
            numbers_files = os.listdir(numbers_path)
        if os.path.exists(controls_path):
            controls_files = os.listdir(controls_path)
        df_letters = self.generate_data_set_by_files(letters_path,letters_files)
        df_numbers = self.generate_data_set_by_files(numbers_path,numbers_files)
        df_controls = self.generate_data_set_by_files(controls_path,controls_files)
        df_general = {**df_letters, **df_numbers, **df_controls}
        print("DataFrames Generados:")
        print("Letters:", df_letters.keys(),df_letters.shape)
        print("Numbers:", df_numbers.keys(),df_numbers.shape)
        print("Controls:", df_controls.keys(),df_controls.shape)
        print("General:", df_general.keys(),df_general.shape)



if __name__ == "__main__":
    controller = controllerTraining()
    user = "UserMartinEpoc"
    path = "captures/" + user
    #controller.train_model(user, path, "LDA")
    controller.create_df(path)