from pathlib import Path
import pandas as pd


BASE = Path(r"C:\tmp_bci\resultados_calidad")

ENTRADA = BASE / "pearson_region_simbolo.csv"

SALIDA = BASE / "consistencia_regional.csv"


df = pd.read_csv(ENTRADA)


# ============================================================
# PORCENTAJE DE SÍMBOLOS EN LOS QUE LA REGIÓN ESTÁ EN TOP 3
# ============================================================

total_simbolos = df["clase"].nunique()

top3 = df[df["ranking_region"] <= 3].copy()

consistencia = (
    top3.groupby("region")
        .agg(
            simbolos_top3=("clase", "nunique"),
            similitud_media_top3=(
                "abs_r_mediana_media",
                "mean"
            ),
            similitud_maxima=(
                "abs_r_mediana_max",
                "max"
            ),
        )
        .reset_index()
)

consistencia["total_simbolos"] = total_simbolos

consistencia["pct_simbolos_top3"] = (
    100 *
    consistencia["simbolos_top3"] /
    total_simbolos
)


# ============================================================
# ORDENAR
# ============================================================

consistencia = consistencia.sort_values(
    [
        "pct_simbolos_top3",
        "similitud_media_top3"
    ],
    ascending=False
)


# ============================================================
# GUARDAR
# ============================================================

consistencia.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR
# ============================================================

print("\n" + "=" * 80)
print("CONSISTENCIA DE LAS REGIONES")
print("=" * 80)

print(
    consistencia.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nArchivo:")
print(SALIDA)
