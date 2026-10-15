from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE = Path(r"C:\tmp_bci\resultados_calidad")

ENTRADA = BASE / "pearson_clase_canal_entre_usuarios.csv"

SALIDA_CANAL_REGION = BASE / "pearson_canal_region_simbolo.csv"
SALIDA_REGION = BASE / "pearson_region_simbolo.csv"
SALIDA_TOP = BASE / "top_regiones_por_simbolo.csv"


# ============================================================
# MAPA CANAL -> REGIÓN
# ============================================================

MAPA_REGION = {
    # Frontal anterior
    "AF3": "Frontal",
    "AF4": "Frontal",

    # Frontal
    "F3": "Frontal",
    "F4": "Frontal",
    "F7": "Frontal",
    "F8": "Frontal",

    # Fronto-central
    "FC5": "Fronto-central",
    "FC6": "Fronto-central",

    # Central
    "C3": "Central",
    "C4": "Central",
    "Cz": "Central",

    # Temporal
    "T7": "Temporal",
    "T8": "Temporal",

    # Parietal
    "P7": "Parietal",
    "P8": "Parietal",

    # Occipital
    "O1": "Occipital",
    "O2": "Occipital",
}


# ============================================================
# CARGAR
# ============================================================

print("=" * 80)
print("ANÁLISIS REGIÓN × SÍMBOLO")
print("=" * 80)

print(f"\nEntrada:")
print(ENTRADA)

if not ENTRADA.exists():
    raise FileNotFoundError(
        f"No existe:\n{ENTRADA}"
    )

df = pd.read_csv(ENTRADA)

print("\nColumnas encontradas:")
print(df.columns.tolist())


# ============================================================
# VALIDACIÓN
# ============================================================

necesarias = [
    "clase",
    "canal",
    "n_pares",
    "r_media",
    "r_mediana",
    "abs_r_media",
    "abs_r_mediana",
    "abs_r_p95",
    "pct_abs_r_ge_03",
    "pct_abs_r_ge_05",
    "pct_abs_r_ge_07",
    "pct_abs_r_ge_09",
]

faltantes = [
    c for c in necesarias
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"\nFaltan columnas:\n{faltantes}"
    )


# ============================================================
# LIMPIEZA
# ============================================================

df["clase"] = df["clase"].astype(str).str.strip()
df["canal"] = df["canal"].astype(str).str.strip()

metricas = [
    "n_pares",
    "r_media",
    "r_mediana",
    "abs_r_media",
    "abs_r_mediana",
    "abs_r_p95",
    "pct_abs_r_ge_03",
    "pct_abs_r_ge_05",
    "pct_abs_r_ge_07",
    "pct_abs_r_ge_09",
]

for col in metricas:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# ASIGNAR REGIÓN
# ============================================================

df["region"] = df["canal"].map(MAPA_REGION)

desconocidos = sorted(
    df.loc[
        df["region"].isna(),
        "canal"
    ].unique()
)

print("\n" + "=" * 80)
print("CANALES SIN REGIÓN ASIGNADA")
print("=" * 80)

if desconocidos:
    print(desconocidos)
    raise ValueError(
        "Hay canales sin región asignada. "
        "Añádelos al MAPA_REGION antes de continuar."
    )
else:
    print("Todos los canales tienen región.")


# ============================================================
# GUARDAR NIVEL CANAL + REGIÓN
# ============================================================

df.to_csv(
    SALIDA_CANAL_REGION,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# RESUMEN REGIONAL
#
# Para cada símbolo y región calculamos:
#
# - media de abs(r_mediana)
# - mediana de abs(r_mediana)
# - máximo de abs(r_mediana)
# - p95 de abs(r_mediana)
# - media de % de pares con |r| >= .5
# - media de % de pares con |r| >= .7
# - media de % de pares con |r| >= .9
#
# IMPORTANTE:
# La métrica principal será la MEDIANA regional de |r|.
# ============================================================

df["abs_r_mediana_directa"] = df["abs_r_mediana"]


region = (
    df
    .groupby(
        ["clase", "region"],
        as_index=False
    )
    .agg(
        n_canales=(
            "canal",
            "nunique"
        ),

        abs_r_mediana_media=(
            "abs_r_mediana",
            "mean"
        ),

        abs_r_mediana_mediana=(
            "abs_r_mediana",
            "median"
        ),

        abs_r_mediana_max=(
            "abs_r_mediana",
            "max"
        ),

        abs_r_p95_media=(
            "abs_r_p95",
            "mean"
        ),

        pct_abs_r_ge_05_media=(
            "pct_abs_r_ge_05",
            "mean"
        ),

        pct_abs_r_ge_07_media=(
            "pct_abs_r_ge_07",
            "mean"
        ),

        pct_abs_r_ge_09_media=(
            "pct_abs_r_ge_09",
            "mean"
        ),
    )
)


# ============================================================
# RANKING REGIONAL POR SÍMBOLO
# ============================================================

region["ranking_region"] = (
    region
    .groupby("clase")[
        "abs_r_mediana_mediana"
    ]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)

region = region.sort_values(
    [
        "clase",
        "ranking_region",
        "abs_r_mediana_mediana"
    ],
    ascending=[
        True,
        True,
        False
    ]
)


# ============================================================
# GUARDAR RESUMEN REGIONAL
# ============================================================

region.to_csv(
    SALIDA_REGION,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TOP 3 REGIONES POR SÍMBOLO
# ============================================================

top = region[
    region["ranking_region"] <= 3
].copy()

top.to_csv(
    SALIDA_TOP,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 100)
print("TOP 3 REGIONES POR SÍMBOLO")
print("=" * 100)

for clase, grupo in top.groupby(
    "clase",
    sort=False
):

    print(f"\nCLASE: {clase}")

    columnas = [
        "ranking_region",
        "region",
        "n_canales",
        "abs_r_mediana_mediana",
        "abs_r_mediana_media",
        "abs_r_mediana_max",
        "abs_r_p95_media",
        "pct_abs_r_ge_05_media",
        "pct_abs_r_ge_07_media",
        "pct_abs_r_ge_09_media",
    ]

    print(
        grupo[columnas].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# ARCHIVOS
# ============================================================

print("\n" + "=" * 100)
print("ARCHIVOS GENERADOS")
print("=" * 100)

print("\nNivel canal + región:")
print(SALIDA_CANAL_REGION)

print("\nResumen región × símbolo:")
print(SALIDA_REGION)

print("\nTop 3 regiones por símbolo:")
print(SALIDA_TOP)

print("\nANÁLISIS FINALIZADO")
