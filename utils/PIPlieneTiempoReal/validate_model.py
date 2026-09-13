# -*- coding: utf-8 -*-
"""
Script de validación en tiempo real.

Uso:
    python validate_model.py <usuario> <ruta_archivo_csv> <comando_real_conocido>

Ejemplo:
    python validate_model.py UserCMSM UserCMSM_UNKNOWN_5.csv W

Predice usando el archivo (segmentando en 5 flashes, extrayendo + promediando),
compara contra el comando real, y muestra si fue correcto o no.
"""

import sys
from hierarchical_infer import HierarchicalBCIPredictor


def validate_single(usuario, csv_path, comando_real, use_flash_seg=True):
    """
    Predice sobre 1 archivo y compara contra comando_real.
    
    Retorna: (comando_predicho, es_correcto, detalle)
    """
    predictor = HierarchicalBCIPredictor(usuario=usuario, use_flash_segmentation=use_flash_seg)
    comando_pred, detalle = predictor.predict_from_csv(csv_path)
    es_correcto = comando_pred.upper() == comando_real.upper()
    return comando_pred, es_correcto, detalle


def main():
    if len(sys.argv) < 4:
        print("Uso: python validate_model.py <usuario> <csv_path> <comando_real_conocido>")
        print("Ejemplo: python validate_model.py UserCMSM UserCMSM_UNKNOWN_5.csv W")
        print()
        print("Opciones avanzadas:")
        print("  --no-segmentation : deshabilita segmentación de 5 flashes (prueba comparativa)")
        sys.exit(1)

    usuario = sys.argv[1]
    csv_path = sys.argv[2]
    comando_real = sys.argv[3]
    use_flash_seg = "--use-segmentation" in sys.argv

    print(f"Validando: usuario={usuario}, archivo={csv_path}, comando_real={comando_real}")
    if use_flash_seg:
        print("Modo: CON segmentación de 5 flashes (promediado múltiple)")
    else:
        print("Modo: SIN segmentación (como fue entrenado)")
    print()

    comando_pred, es_correcto, detalle = validate_single(usuario, csv_path, comando_real, use_flash_seg)

    print(f">>> Comando predicho: {comando_pred}")
    print(f">>> Comando real: {comando_real}")
    print(f">>> Resultado: {'✓ CORRECTO' if es_correcto else '✗ INCORRECTO'}")
    print()
    print("Probabilidades de grupo:")
    for g, p in detalle["grupo_probabilidades"].items():
        print(f"  {g}: {p:.4f}")
    print()
    print(f"Top 5 probabilidades de comando ({detalle['grupo_predicho']}):")
    for cmd, p in list(detalle["comando_probabilidades"].items())[:5]:
        marker = " <-- CORRECTO" if cmd.upper() == comando_real.upper() else ""
        print(f"  {cmd}: {p:.4f}{marker}")


if __name__ == "__main__":
    main()
