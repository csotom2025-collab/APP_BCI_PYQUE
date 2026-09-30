from pathlib import Path
import pandas as pd


BASE = Path(r"C:\tmp_bci\resultados_calidad")

ENTRADA = BASE / "pearson_clase_canal_entre_usuarios.csv"

SALIDA_CANAL = BASE / "matriz_simbolo_canal_abs_r_mediana.csv"
SALIDA_REGION = BASE / "matriz_simbolo_region_abs_r_mediana.csv"


# ============================================================
# CARGAR
# ============================================================

df = pd.read_csv(ENTRADA)

df["clase"] = df["clase"].astype(str).str.strip()
df["canal"] = df["canal"].astype(str).str.strip()

df["abs_r_mediana"] = pd.to_numeric(
    df["abs_r_mediana"],
    errors="coerce"
)


# ============================================================
# MATRIZ SÍMBOLO × CANAL
# ============================================================

matriz_canal = df.pivot_table(
    index="clase",
    columns="canal",
    values="abs_r_mediana",
    aggfunc="mean"
)


orden_canales = [
    "AF3",
    "AF4",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC5",
    "FC6",
    "C3",
    "C4",
    "Cz",
    "T7",
    "T8",
    "P7",
    "P8",
    "O1",
    "O2",
]

matriz_canal = matriz_canal.reindex(
    columns=[
        c for c in orden_canales
        if c in matriz_canal.columns
    ]
)


matriz_canal.to_csv(
    SALIDA_CANAL,
    encoding="utf-8-sig"
)


# ============================================================
# MAPA REGIONAL
# ============================================================

MAPA_REGION = {
    "AF3": "Frontal",
    "AF4": "Frontal",
    "F3": "Frontal",
    "F4": "Frontal",
    "F7": "Frontal",
    "F8": "Frontal",
    "FC5": "Fronto-central",
    "FC6": "Fronto-central",
    "C3": "Central",
    "C4": "Central",
    "Cz": "Central",
    "T7": "Temporal",
    "T8": "Temporal",
    "P7": "Parietal",
    "P8": "Parietal",
    "O1": "Occipital",
    "O2": "Occipital",
}


df["region"] = df["canal"].map(MAPA_REGION)


# ============================================================
# MATRIZ SÍMBOLO × REGIÓN
# ============================================================

matriz_region = df.pivot_table(
    index="clase",
    columns="region",
    values="abs_r_mediana",
    aggfunc="median"
)


orden_regiones = [
    "Frontal",
    "Fronto-central",
    "Central",
    "Temporal",
    "Parietal",
    "Occipital",
]

matriz_region = matriz_region.reindex(
    columns=[
        r for r in orden_regiones
        if r in matriz_region.columns
    ]
)


matriz_region.to_csv(
    SALIDA_REGION,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR
# ============================================================

print("\n" + "=" * 100)
print("MATRIZ SÍMBOLO × CANAL")
print("=" * 100)

print(
    matriz_canal.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 100)
print("MATRIZ SÍMBOLO × REGIÓN")
print("=" * 100)

print(
    matriz_region.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 100)
print("ARCHIVOS GENERADOS")
print("=" * 100)

print(SALIDA_CANAL)
print(SALIDA_REGION)

print("\nANÁLISIS FINALIZADO")
