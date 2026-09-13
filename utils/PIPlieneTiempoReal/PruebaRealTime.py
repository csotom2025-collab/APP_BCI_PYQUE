from hierarchical_infer import HierarchicalBCIPredictor
from validate_model import validate_single

predictor = HierarchicalBCIPredictor(usuario="UserCMSM")
# Por defecto, use_flash_segmentation=True (automático)
comand="A"
tip="Letters"
file=[2,3,5,7,11,13,17,19,23,29]
for num in file:
    csv_file = f"D:/APP_BCI_PYQUE/captures/UserCMSM/{tip}/UserCMSM_{comand}_{num}.csv"
    comando, detalle = predictor.predict_from_csv(csv_file)
    print(f"Archivo: UserCMSM_{comand}_{num}.csv")
    print(f"Comando predicho: {comando}")
    print(f"Grupo: {detalle['grupo_predicho']}")
    print("-" * 40)


#python validate_model.py UserCMSM D:/APP_BCI_PYQUE/captures/Speller/UserCMSM_UNKNOWN_2.csv O

##D:/APP_BCI_PYQUE/captures/UserCMSM/UNKNOWN/UserCMSM_UNKNOWN_2.csv