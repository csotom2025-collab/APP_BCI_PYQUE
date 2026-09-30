from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE = Path(r"C:\tmp_bci\resultados_calidad")

ENTRADA = BASE / "matriz_simbolo_region_abs_r_mediana.csv"

SALIDA_RANKING = BASE / "ranking_regiones_simbolo.csv"
SALIDA_RESUMEN = BASE / "resumen_regiones_dominantes_por_simbolo.csv"
SALIDA_VALIDACION = BASE / "validacion_regiones_simbolo.txt"

REGIONES_ESPERADAS = [
    "Frontal",
    "Fronto-central",
    "Temporal",
    "Parietal",
    "Occipital",
]

# Diferencia mínima entre 1.º y 2.º para considerar
# que existe una dominancia relativamente clara.
UMBRAL_DIFERENCIA = 0.01


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def error(mensaje):
    raise ValueError("\nERROR DE VALIDACIÓN:\n" + mensaje)


def limpiar_nombre_columna(col):
    return str(col).strip()


def es_empate(v1, v2, tolerancia=1e-6):
    return abs(v1 - v2) <= tolerancia


# ============================================================
# 1. COMPROBAR ARCHIVO
# ============================================================

print("=" * 90)
print("RANKING DE REGIONES POR SÍMBOLO")
print("=" * 90)

print(f"\nArchivo de entrada:")
print(ENTRADA)

if not ENTRADA.exists():
    error(
        f"No existe el archivo de entrada:\n{ENTRADA}\n\n"
        "Ejecuta primero matriz_simbolo_canal.py para generarlo."
    )


# ============================================================
# 2. CARGAR DATOS
# ============================================================

try:
    df = pd.read_csv(ENTRADA)
except Exception as e:
    error(f"No se pudo leer el CSV:\n{e}")

df.columns = [limpiar_nombre_columna(c) for c in df.columns]

print("\nColumnas encontradas:")
print(df.columns.tolist())


# ============================================================
# 3. VALIDAR COLUMNAS
# ============================================================

if "clase" not in df.columns:
    error(
        "Falta la columna obligatoria 'clase'.\n"
        f"Columnas disponibles: {df.columns.tolist()}"
    )

regiones_en_archivo = [c for c in df.columns if c != "clase"]

faltantes = [
    r for r in REGIONES_ESPERADAS
    if r not in regiones_en_archivo
]

extras = [
    r for r in regiones_en_archivo
    if r not in REGIONES_ESPERADAS
]

if faltantes:
    error(
        "Faltan regiones esperadas:\n"
        + "\n".join(f"  - {r}" for r in faltantes)
    )

if extras:
    error(
        "Se encontraron columnas que no corresponden a las regiones "
        "esperadas:\n"
        + "\n".join(f"  - {r}" for r in extras)
    )

# Ordenar explícitamente
df = df[["clase"] + REGIONES_ESPERADAS]


# ============================================================
# 4. VALIDAR SÍMBOLOS
# ============================================================

if df["clase"].isna().any():
    error("Hay filas con 'clase' vacía o NaN.")

duplicados = df["clase"].duplicated(keep=False)

if duplicados.any():
    valores = df.loc[duplicados, "clase"].tolist()

    error(
        "Hay símbolos/clases duplicados.\n"
        f"Valores duplicados: {valores}"
    )

if len(df) == 0:
    error("El archivo no contiene filas.")


# ============================================================
# 5. VALIDAR VALORES NUMÉRICOS
# ============================================================

for region in REGIONES_ESPERADAS:

    valores_originales = df[region]

    valores_numericos = pd.to_numeric(
        valores_originales,
        errors="coerce"
    )

    if valores_numericos.isna().any():
        filas = df.loc[
            valores_numericos.isna(),
            ["clase", region]
        ]

        error(
            f"La región '{region}' contiene valores no numéricos "
            "o vacíos:\n"
            f"{filas.to_string(index=False)}"
        )

    df[region] = valores_numericos

    if not np.isfinite(df[region]).all():
        error(
            f"La región '{region}' contiene valores infinitos."
        )


# ============================================================
# 6. VALIDAR RANGO DE CORRELACIÓN ABSOLUTA
# ============================================================

# abs(r) mediana debería estar entre 0 y 1.

for region in REGIONES_ESPERADAS:

    fuera_rango = (
        (df[region] < 0) |
        (df[region] > 1)
    )

    if fuera_rango.any():
        filas = df.loc[
            fuera_rango,
            ["clase", region]
        ]

        error(
            f"La región '{region}' tiene valores fuera del rango "
            "[0, 1]:\n"
            f"{filas.to_string(index=False)}"
        )


# ============================================================
# 7. VALIDACIÓN GENERAL
# ============================================================

print("\n" + "=" * 90)
print("VALIDACIÓN")
print("=" * 90)

print(f"OK - Archivo encontrado.")
print(f"OK - Filas/símbolos: {len(df)}")
print(f"OK - Columna 'clase' presente.")
print("OK - Regiones encontradas:")

for region in REGIONES_ESPERADAS:
    print(f"     - {region}")

print("OK - No hay regiones faltantes.")
print("OK - No hay columnas regionales extra.")
print("OK - No hay símbolos duplicados.")
print("OK - No hay valores NaN.")
print("OK - Todos los valores son numéricos.")
print("OK - Todos los valores están en [0, 1].")


# ============================================================
# 8. CALCULAR RANKING POR SÍMBOLO
# ============================================================

filas_ranking = []

for _, fila in df.iterrows():

    simbolo = fila["clase"]

    valores = {
        region: float(fila[region])
        for region in REGIONES_ESPERADAS
    }

    ordenados = sorted(
        valores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for posicion, (region, valor) in enumerate(
        ordenados,
        start=1
    ):

        filas_ranking.append({
            "clase": simbolo,
            "ranking": posicion,
            "region": region,
            "similitud_abs_r_mediana": valor,
        })


ranking = pd.DataFrame(filas_ranking)


# ============================================================
# 9. RESUMEN: DOMINANCIA POR SÍMBOLO
# ============================================================

resumen = []

for _, fila in df.iterrows():

    simbolo = fila["clase"]

    valores = {
        region: float(fila[region])
        for region in REGIONES_ESPERADAS
    }

    ordenados = sorted(
        valores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    region_1, valor_1 = ordenados[0]
    region_2, valor_2 = ordenados[1]
    region_3, valor_3 = ordenados[2]

    diferencia_1_2 = valor_1 - valor_2
    diferencia_2_3 = valor_2 - valor_3

    # Comprobar si hay empate práctico en primer lugar
    empate_1 = [
        region
        for region, valor in ordenados
        if es_empate(valor, valor_1)
    ]

    if len(empate_1) > 1:
        estado = "empate_practico"
    elif diferencia_1_2 >= UMBRAL_DIFERENCIA:
        estado = "dominancia_clara"
    else:
        estado = "dominancia_debil"

    resumen.append({
        "clase": simbolo,

        "region_1": region_1,
        "similitud_1": valor_1,

        "region_2": region_2,
        "similitud_2": valor_2,

        "region_3": region_3,
        "similitud_3": valor_3,

        "diferencia_1_vs_2": diferencia_1_2,
        "diferencia_2_vs_3": diferencia_2_3,

        "estado_dominancia": estado,

        "regiones_empate_primer_lugar":
            " | ".join(empate_1),
    })


resumen = pd.DataFrame(resumen)


# ============================================================
# 10. ESTADÍSTICAS DE ROBUSTEZ
# ============================================================

conteo_primer_lugar = (
    resumen["region_1"]
    .value_counts()
    .reindex(REGIONES_ESPERADAS, fill_value=0)
)

porcentaje_primer_lugar = (
    conteo_primer_lugar / len(resumen) * 100
)

resumen_regiones = pd.DataFrame({
    "region": REGIONES_ESPERADAS,
    "simbolos_primer_lugar":
        conteo_primer_lugar.values,
    "pct_simbolos_primer_lugar":
        porcentaje_primer_lugar.values,
})


# ============================================================
# 11. GUARDAR RESULTADOS
# ============================================================

ranking.to_csv(
    SALIDA_RANKING,
    index=False,
    encoding="utf-8-sig"
)

resumen.to_csv(
    SALIDA_RESUMEN,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. ARCHIVO DE VALIDACIÓN
# ============================================================

with open(
    SALIDA_VALIDACION,
    "w",
    encoding="utf-8"
) as f:

    f.write("VALIDACIÓN - RANKING REGIONES POR SÍMBOLO\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Archivo: {ENTRADA}\n")
    f.write(f"Símbolos: {len(df)}\n")
    f.write(
        "Regiones: "
        + ", ".join(REGIONES_ESPERADAS)
        + "\n\n"
    )

    f.write("VALIDACIONES CORRECTAS:\n")
    f.write("- Archivo existente\n")
    f.write("- Columna clase presente\n")
    f.write("- Todas las regiones presentes\n")
    f.write("- No hay regiones adicionales\n")
    f.write("- No hay símbolos duplicados\n")
    f.write("- No hay NaN\n")
    f.write("- Valores numéricos\n")
    f.write("- Valores dentro de [0, 1]\n")


# ============================================================
# 13. MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 90)
print("REGIÓN DOMINANTE POR SÍMBOLO")
print("=" * 90)

mostrar = resumen[
    [
        "clase",
        "region_1",
        "similitud_1",
        "region_2",
        "similitud_2",
        "diferencia_1_vs_2",
        "estado_dominancia",
    ]
].copy()

print(
    mostrar.to_string(
        index=False,
        formatters={
            "similitud_1": "{:.4f}".format,
            "similitud_2": "{:.4f}".format,
            "diferencia_1_vs_2": "{:.4f}".format,
        }
    )
)


print("\n" + "=" * 90)
print("ROBUSTEZ DEL PRIMER LUGAR")
print("=" * 90)

print(
    resumen_regiones.to_string(
        index=False,
        formatters={
            "pct_simbolos_primer_lugar":
                "{:.2f}%".format
        }
    )
)


# ============================================================
# 14. RESUMEN DE DOMINANCIAS
# ============================================================

print("\n" + "=" * 90)
print("RESUMEN DE DOMINANCIA")
print("=" * 90)

conteo_estado = (
    resumen["estado_dominancia"]
    .value_counts()
)

for estado in [
    "dominancia_clara",
    "dominancia_debil",
    "empate_practico",
]:

    print(
        f"{estado:22s}: "
        f"{conteo_estado.get(estado, 0)} símbolos"
    )


# ============================================================
# 15. ARCHIVOS
# ============================================================

print("\n" + "=" * 90)
print("ARCHIVOS GENERADOS")
print("=" * 90)

print(SALIDA_RANKING)
print(SALIDA_RESUMEN)
print(SALIDA_VALIDACION)

print("\nANÁLISIS FINALIZADO CORRECTAMENTE.")
