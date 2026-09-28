from hierarchical_infer import HierarchicalBCIPredictor
import os


User="UserCMSM"
comand="7"
tip="Numbers"
file=[i for  i in range(1,29)]
predictor = HierarchicalBCIPredictor(usuario=User)
reCom=0
realTp=0
for num in file:
    csv_file = f"D:/APP_BCI_PYQUE/captures/{User}/{tip}/{User}_{comand}_{num}.csv"
    #checar si existe el archivo csv si no existe continuar
    if (os.path.exists(csv_file) == False):
        print("El archivo "+csv_file+" no existe")
        continue    
    comando, detalle = predictor.predict_from_csv(csv_file)
    
    
    if (comando==comand):
        reCom+=1
    if (tip==detalle['grupo_predicho']):
        realTp+=1
    # print("----------------------------------------------------------------")
    # print(detalle["ventanas"][0])
    # print(detalle["ventanas"][1])
    # print(detalle["ventanas"][2])
    # print(detalle["ventanas"][3])
    # print("----------------------------------------------------------------")
    # print("Comando: ",comando," Tipo:",detalle['grupo_predicho'],"del archivo ",csv_file)
print("Comando Reales Predichos, ",reCom)
print("Grupo Real Predicho, ",realTp)
#python validate_model.py UserCMSM D:/APP_BCI_PYQUE/captures/Speller/UserCMSM_UNKNOWN_2.csv O

##D:/APP_BCI_PYQUE/captures/UserCMSM/UNKNOWN/UserCMSM_UNKNOWN_2.csv
#Deltrador EEG Habla Imgainda del CIC  DEHI_CIC