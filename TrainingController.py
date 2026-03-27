import os
import pandas as pd
import numpy as np
class controllerTraining():
    def __init__(self):
        pass

    def train_model(self,dataPath,modelType):
        if not os.path.exists(dataPath):
            print("no existe la ruta")
            return

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
        files_dataset =[]
        for file_name in files:
            file = pd.read_csv(path + f'/{file_name}')
            file_channels =[]
            for channel in columnas[1:]: # no nos importa el Tm o si ???
                file_channels.append(np.array(file[channel]))
            file_channels = np.array(file_channels)
            files_dataset.append(file_channels)
            print(file_channels.shape)
            label = self.get_label(file_name)
            print(label)
        files_dataset = np.array(files_dataset)
        print(files_dataset.shape)


    def create_df(self,path):

        """
        crear 3 datasets por cada clasificacion y uno general
        """
        directories = os.listdir(path)+
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
path = "captures/User0"
controller.train_model(path,"lstm")