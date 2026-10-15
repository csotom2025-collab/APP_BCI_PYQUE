from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE = Path(r"C:\tmp_bci\resultados_calidad")

ARCHIVO_ENTRADA = BASE / "pearson_clase_canal_entre_usuarios.csv"

SALIDA_RANKING = BASE / "ranking_canal_simbolo.csv"
SALIDA_TOP = BASE / "ranking_top_canales_por_simbolo.csv"


# ============================================================
# CARGAR
# ============================================================

df = pd.read_csv(ARCHIVO_ENTRADA)

print("Archivo:", ARCHIVO_ENTRADA)
print("\nColumnas encontradas:")
print(df.columns.tolist())

print("\nPrimeras filas:")
print(df.head())


# ============================================================
# VALIDAR COLUMNAS
# ============================================================

columnas_necesarias = [
    "clase",
    "canal",
    "n_pares",
    "r_mediana",
    "abs_r_mediana",
    "abs_r_p95",
    "pct_abs_r_ge_05",
    "pct_abs_r_ge_07",
    "pct_abs_r_ge_09",
]

faltantes = [c for c in columnas_necesarias if c not in df.columns]

if faltantes:
    raise ValueError(
        f"\nFaltan columnas necesarias: {faltantes}\n"
        f"Columnas disponibles: {df.columns.tolist()}"
    )


# ============================================================
# CONVERSIÓN NUMÉRICA
# ============================================================

metricas = [
    "n_pares",
    "r_mediana",
    "abs_r_mediana",
    "abs_r_p95",
    "pct_abs_r_ge_05",
    "pct_abs_r_ge_07",
    "pct_abs_r_ge_09",
]

for col in metricas:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# RANKING
#
# Criterio principal:
#   abs_r_mediana
#
# Es decir:
#   similitud interusuario típica por canal.
#
# No usamos r_mediana porque una correlación negativa
# también representa asociación, y aquí queremos magnitud.
# ============================================================

df = df.dropna(subset=["clase", "canal", "abs_r_mediana"])

df["ranking"] = (
    df.groupby("clase")["abs_r_mediana"]
      .rank(method="min", ascending=False)
      .astype(int)
)

df = df.sort_values(
    ["clase", "ranking", "abs_r_mediana"],
    ascending=[True, True, False]
)


# ============================================================
# GUARDAR RANKING COMPLETO
# ============================================================

df.to_csv(
    SALIDA_RANKING,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TOP 5 POR SÍMBOLO
# ============================================================

top5 = (
    df[df["ranking"] <= 5]
    .copy()
)

top5.to_csv(
    SALIDA_TOP,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 80)
print("TOP 5 CANALES POR SÍMBOLO")
print("=" * 80)

for clase, grupo in top5.groupby("clase", sort=False):

    print(f"\nCLASE: {clase}")

    mostrar = grupo[
        [
            "canal",
            "abs_r_mediana",
            "abs_r_p95",
            "pct_abs_r_ge_05",
            "pct_abs_r_ge_07",
            "pct_abs_r_ge_09",
        ]
    ]

    print(
        mostrar.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


print("\n" + "=" * 80)
print("ARCHIVOS GENERADOS")
print("=" * 80)

print("Ranking completo:")
print(SALIDA_RANKING)

print("\nTop 5 por símbolo:")
print(SALIDA_TOP)

print("\nANÁLISIS FINALIZADO")
