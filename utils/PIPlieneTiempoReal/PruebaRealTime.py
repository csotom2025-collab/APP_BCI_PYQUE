from matplotlib._api import deprecation
from hierarchical_infer import HierarchicalBCIPredictor
import os
import logging

# Puedes poner este diccionario al inicio de tu script
MAPA_CONTROLES = {
    "\\u27f5": "BORRAR (⟵)",      
    "\u27f5":  "BORRAR (⟵)",      
    "\\u21a9": "ENTER (↩)",       
    "\u21a9":  "ENTER (↩)",       
    "\\u2500\\u2500\\u2500": "ESPACIO (───)",     
    "\u2500\u2500\u2500":  "ESPACIO (───)",
    "───":     "ESPACIO (───)"
}



User="UserCMSM"
comands=["A","E","I","O","U","S","R","L","D"]
tip="Letters"
file=[i for  i in range(30,60)]
predictor = HierarchicalBCIPredictor(usuario=User)
reCom=0
realTp=0
totlFile=0
logging.basicConfig(
    filename="RealTimeLOG.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
for comand in comands:
    logging.info("=======================================================================================")
    for num in file:
        csv_file = f"D:/APP_BCI_PYQUE/captures/{User}/{tip}/{User}_{comand}_{num}.csv"
        #checar si existe el archivo csv si no existe continuar
        if (os.path.exists(csv_file) == False):
            logging.info(f"El archivo {csv_file} no existe")
            continue    
        comando, detalle = predictor.predict_from_csv(csv_file)
        totlFile+=1

        if(detalle['grupo_predicho']== "Controls"):
            comando=MAPA_CONTROLES.get(comando, comando)

        if (comando==comand):
            reCom+=1
        if (tip==detalle['grupo_predicho']):
            realTp+=1
        
        logging.info(f"------------------>cvs name {csv_file}<----------------")
        for ventana in detalle["ventanas"]:
             logging.info(ventana)
        logging.info(f"Comando predicho {comando}")
        logging.info(f"Grupo predicho {detalle['grupo_predicho']}")
        logging.info("----------------------------------------------------------------")
    logging.info(f"Del Comando {comand} Total Archivos {totlFile}")
    logging.info(f"Comando Reales Predichos, {reCom}")
    logging.info(f"Grupo Real Predicho, {realTp}")
    reCom=0
    realTp=0
    totlFile=0
    logging.info("=======================================================================================")
    print("Siguiente comando",comand)
print("Fin del programa")
#python validate_model.py UserCMSM D:/APP_BCI_PYQUE/captures/Speller/UserCMSM_UNKNOWN_2.csv O

##D:/APP_BCI_PYQUE/captures/UserCMSM/UNKNOWN/UserCMSM_UNKNOWN_2.csv
#Deltrador EEG Habla Imgainda del CIC  DEHI_CIC