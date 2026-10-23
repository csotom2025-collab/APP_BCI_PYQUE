from matplotlib._api import deprecation
import os
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.newTraining.BCIpredictorhierarchical import (
    HierarchicalBCIPredictor,
    RecordingQualityError,
)

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



User="UserAlexJ"
modelo_path = os.path.join(
    PROJECT_ROOT,
    "trainingOutputs",
    "models",
    "UserAlexJ",
    "hierarchical_bundle_optimizado.joblib",
)
comands=["UNKNOWN"]
tip="recordings"
file=[i for  i in range(1,60)]
greenComd = ["A", "C", "F", "R", "V", "I", "M", "J", "L", "Q", "U", "B", "X",
             "4", "0", "2", "8", "W", "6", "⟵"]
blueComd = ["S", "G", "E", "T", "Z", "N", "Y", "O", "P", "Ñ", "D", "H", "1",
            "7", "K", "5", "───", "3", "9", "↩"]
colorByCommand = {
    **{command: "Green" for command in greenComd},
    **{command: "Blue" for command in blueComd},
}
reCom=0
realTp=0
reColor=0
totlFile=0
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RealTimeLOG.log")
logging.basicConfig(
    filename=log_path,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
predictor = HierarchicalBCIPredictor(usuario=User, model_path=modelo_path)
for comand in comands:
    logging.info("=======================================================================================")
    for num in file:
        csv_file = f"captures/{User}/{tip}/{User}_{comand}_{num}.csv"
        #checar si existe el archivo csv si no existe continuar
        if (os.path.exists(csv_file) == False):
            logging.info(f"El archivo {csv_file} no existe")
            continue    
        totlFile+=1
        try:
            comando, detalle = predictor.predict_from_csv(csv_file)
        except RecordingQualityError as error:
            logging.warning(
                "Grabación omitida por control de calidad: %s: %s",
                csv_file,
                error,
            )
            continue

        if comand in colorByCommand:
            color_real = colorByCommand[comand]
        else:
            color_real = "Desconocido"
        color_predicho = detalle["color_predicho"]
        confianza_color = float(detalle["color_probabilidades"][color_predicho])

        if(detalle['grupo_predicho']== "Controls"):
            comando=MAPA_CONTROLES.get(comando, comando)

        if (comando==comand):
            reCom+=1
        if (tip==detalle['grupo_predicho']):
            realTp+=1
        if (color_real==color_predicho):
            reColor+=1
        
        logging.info(f"------------------>cvs name {csv_file}<----------------")
        logging.info(f"Color real {color_real}")
        logging.info(
            f"Color predicho {color_predicho} "
            f"(confianza {confianza_color:.1%})"
        )
        logging.info(f"Probabilidades de color {detalle['color_probabilidades']}")
        for ventana, detalle_color in zip(
            detalle["ventanas"], detalle["detalle_ventanas_color"]
        ):
            color_ventana = detalle_color["prediccion"]
            confianza_ventana = float(
                detalle_color["probabilidades"][color_ventana]
            )
            logging.info(
                f"Ventana {ventana['ventana']}: color predicho {color_ventana} "
                f"(confianza {confianza_ventana:.1%}); "
                f"probabilidades {detalle_color['probabilidades']}"
            )
            logging.info(ventana)
        logging.info(f"Comando predicho {comando}")
        logging.info(f"Grupo predicho {detalle['grupo_predicho']}")
        logging.info("----------------------------------------------------------------")
    logging.info(f"Del Comando {comand} Total Archivos {totlFile}")
    logging.info(f"Comando Reales Predichos, {reCom}")
    logging.info(f"Grupo Real Predicho, {realTp}")
    logging.info(f"Color Real Predicho, {reColor}")
    reCom=0
    realTp=0
    reColor=0
    totlFile=0
    logging.info("=======================================================================================")
print("Fin del programa")
#python validate_model.py UserCMSM D:/APP_BCI_PYQUE/captures/Speller/UserCMSM_UNKNOWN_2.csv O

##D:/APP_BCI_PYQUE/captures/UserCMSM/UNKNOWN/UserCMSM_UNKNOWN_2.csv
#Deltrador EEG Habla Imgainda del CIC  DEHI_CIC