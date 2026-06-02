import os
import pandas as pd
import numpy as np
from models.pipeline_completo_lda import PipelineCompletoLDA
class controllerTraining():
    def __init__(self):
        pass

    def train_model(self,user,dataPath,modelType):
        if not os.path.exists(dataPath):
            print("no existe la ruta")
            return
        if modelType== "LDA":
            self.pipeline_lda = PipelineCompletoLDA(base_output_dir="resultsALL/")
            self.pipeline_lda.separar_archivos_csv(usuario=user)
            print("SE SEPARARON LOS ARCHIVOS ")
            print("Obteniendo caracteristicas")
            self.pipeline_lda.obtener_caracteristicas_usuario(usuario=user)
            print("EXTRAIDAS TODAS LAS CARACTERISTICAS")
            print("Optimizando LDA")
            self.pipeline_lda.optimizacion_lda(user=user)
            print("Entrenado Modelo")
            #pipeline.entrenar_modelo(user, tipo='LDA_General', grupo='TODAS', modelo='ALL')
            self.pipeline_lda.entrenar_todos_los_modelos(usuario=user, save_plots=True, show_plots=True)
            print("LDA")
        if modelType == "CNN":
            print("cnn")
        df = self.create_df(dataPath)


    def save_model(self,path):
        pass

    def get_label(self,file_name:str):
        label =file_name.split('_')[1]
        return label

    def generate_data_set_by_files(self,path,files):
        """
            generamos el dataset de los archivos ,separados por canales y con su label
        """
        columnas = pd.read_csv(path + f'/{files[0]}').columns # obtenemos los headers del archivo los cuales son los canales 
        print("Columnas Archivo" , columnas)
        labels_dict={}
        files_dataset = []
        for file_name in files:
            label = self.get_label(file_name)

            file = pd.read_csv(path + f'/{file_name}')
            file_channels =[]
            for channel in columnas[1:]: # no nos importa el Tm o si ???
                file_channels.append(np.array(file[channel].iloc[:190])) # 190 muestras
            file_channels = np.array(file_channels)
            files_dataset.append(file_channels)
            if label not in labels_dict:
                labels_dict[label]= [file_channels]
            else:
                labels_dict[label].append(file_channels)
            #print(label)
        print(len(files_dataset))

        files_dataset = np.array(files_dataset)
        file_divided = []
        for idx,label in enumerate(labels_dict.keys()):
            print(f"idx {idx} : {label}")
            file_divided.append(labels_dict[label])
        file_divided = np.array(file_divided)
        print(file_divided.shape)
        

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
            for file in letters_files:
                self.get_label(file)
        if os.path.exists(numbers_path):
            numbers_files = os.listdir(numbers_path)
        if os.path.exists(controls_path):
            controls_files = os.listdir(controls_path)
        self.generate_data_set_by_files(letters_path,letters_files)
        



controller =controllerTraining()
user="Usermar"
path = "captures/" + user
controller.train_model(user,path,"lstm")