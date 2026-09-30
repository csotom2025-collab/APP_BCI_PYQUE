from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\tmp_bci\captures")
REPORTE = Path(r"C:\tmp_bci\resultados_calidad\reporte.csv")

CANALES = [
    "F3", "FC5", "AF3", "F7", "T7", "P7", "O1",
    "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"
]

UMBRALES = [120.0, 150.0]

CARPETAS_IGNORAR = {
    "ignorarSenales"
}

# ============================================================
# FUNCIONES
# ============================================================

def contar_episodios(mask):
    """
    Cuenta episodios consecutivos True y devuelve:
      - número de episodios
      - longitud máxima de un episodio
    """
    if len(mask) == 0:
        return 0, 0

    cambios = np.diff(np.concatenate(([False], mask, [False])).astype(int))

    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0]

    duraciones = finales - inicios

    if len(duraciones) == 0:
        return 0, 0

    return len(duraciones), int(np.max(duraciones))

def cargar_archivos_incluidos():
    """
    Carga desde reporte.csv el universo de archivos
    que el pipeline marcó como incluido=True.
    """

    reporte = pd.read_csv(REPORTE)

    if "ruta_relativa" not in reporte.columns:
        raise ValueError(
            "reporte.csv no contiene la columna 'ruta_relativa'"
        )

    if "incluido" not in reporte.columns:
        raise ValueError(
            "reporte.csv no contiene la columna 'incluido'"
        )

    incluidos = reporte[
        reporte["incluido"].astype(str).str.lower().isin(
            ["true", "1", "si", "sí"]
        )
    ].copy()

    rutas = []

    for ruta_relativa in incluidos["ruta_relativa"]:
        path = ROOT / Path(str(ruta_relativa))

        if path.exists():
            rutas.append(path)
        else:
            print(
                f"ADVERTENCIA: archivo incluido no encontrado: {path}"
            )

    return sorted(rutas)

def analizar_archivo(path):
    try:
        df = pd.read_csv(path)

        # Verificar canales
        if not all(canal in df.columns for canal in CANALES):
            return None, "faltan canales"

        valores = df[CANALES].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(dtype=float)

        # Eliminar filas sin ningún dato EEG válido
        filas_validas = np.isfinite(valores).any(axis=1)

        if not filas_validas.any():
            return None, "sin muestras EEG válidas"

        valores = valores[filas_validas]

        # --------------------------------------------------------
        # Centrado por mediana de cada canal
        # --------------------------------------------------------

        medianas = np.nanmedian(valores, axis=0)
        centrado = valores - medianas

        n_muestras = centrado.shape[0]

        # Máximo absoluto por canal
        max_por_canal = np.nanmax(
            np.abs(centrado),
            axis=0
        )

        resultado = {
            "ruta": str(path),
            "n_muestras": n_muestras,
        }

        # --------------------------------------------------------
        # Métricas por canal y umbral
        # --------------------------------------------------------

        for umbral in UMBRALES:

            sufijo = str(int(umbral))

            for i, canal in enumerate(CANALES):

                serie = np.abs(centrado[:, i])

                # Muestras que superan el umbral
                mask = serie > umbral

                n_superan = int(np.sum(mask))

                pct_superan = (
                    100.0 * n_superan / n_muestras
                    if n_muestras > 0
                    else np.nan
                )

                n_episodios, max_run = contar_episodios(mask)

                resultado[f"max_{canal}_uv_{sufijo}"] = float(
                    max_por_canal[i]
                )

                resultado[f"n_gt_{canal}_{sufijo}"] = n_superan

                resultado[f"pct_gt_{canal}_{sufijo}"] = pct_superan

                resultado[f"episodios_{canal}_{sufijo}"] = n_episodios

                resultado[f"max_run_{canal}_{sufijo}"] = max_run

            # ----------------------------------------------------
            # Número de canales afectados
            # ----------------------------------------------------

            n_canales_afectados = 0

            for i, canal in enumerate(CANALES):
                serie = np.abs(centrado[:, i])
                if np.any(serie > umbral):
                    n_canales_afectados += 1

            resultado[f"n_canales_gt_{sufijo}"] = (
                n_canales_afectados
            )

        # --------------------------------------------------------
        # Información del máximo global
        # --------------------------------------------------------

        indice = np.unravel_index(
            np.nanargmax(np.abs(centrado)),
            centrado.shape
        )

        fila_max = indice[0]
        canal_max_idx = indice[1]

        resultado["max_abs_centrado"] = float(
            np.abs(centrado[indice])
        )

        resultado["canal_max"] = CANALES[canal_max_idx]

        resultado["valor_max_centrado"] = float(
            centrado[indice]
        )

        resultado["muestra_max"] = fila_max

        return resultado, None

    except Exception as e:
        return None, str(e)


# ============================================================
# RECORRER ARCHIVOS
# ============================================================

resultados = []
omitidos = []

archivos = cargar_archivos_incluidos()

print()
print("=" * 70)
print("UNIVERSO DE AUDITORÍA")
print("=" * 70)
print(f"Archivos incluidos según reporte: {len(archivos)}")

for path in archivos:

    if any(
        parte in CARPETAS_IGNORAR
        for parte in path.parts
    ):
        continue

    resultado, error = analizar_archivo(path)

    if resultado is not None:
        resultados.append(resultado)
    else:
        omitidos.append((str(path), error))

# ============================================================
# DATAFRAME
# ============================================================

x = pd.DataFrame(resultados)

print()
print("=" * 70)
print("RESUMEN")
print("=" * 70)

print(f"Archivos evaluados: {len(x)}")
print(f"Archivos omitidos: {len(omitidos)}")


# ============================================================
# RESUMEN POR UMBRAL
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    print()
    print("=" * 70)
    print(f"UMBRAL: {umbral} µV")
    print("=" * 70)

    n_archivos = int(
        (x[f"n_canales_gt_{sufijo}"] > 0).sum()
    )

    print(
        f"Archivos con al menos un canal > {umbral} µV: "
        f"{n_archivos}"
    )

    print()

    print("Número de canales afectados por archivo:")

    print(
        x[f"n_canales_gt_{sufijo}"]
        .value_counts()
        .sort_index()
        .to_string()
    )

# ============================================================
# DISTRIBUCIÓN DE MUESTRAS > UMBRAL
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    print()
    print("=" * 70)
    print(
        f"MUESTRAS > {umbral} µV "
        "(porcentaje máximo entre canales)"
    )
    print("=" * 70)

    columnas_pct = [
        f"pct_gt_{canal}_{sufijo}"
        for canal in CANALES
    ]

    pct_max = x[columnas_pct].max(axis=1)

    print(
        pct_max.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
                0.995,
                0.999
            ]
        ).to_string()
    )


# ============================================================
# DISTRIBUCIÓN DE RACHA MÁXIMA
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    print()
    print("=" * 70)
    print(
        f"RACHA MÁXIMA > {umbral} µV "
        "(muestras consecutivas)"
    )
    print("=" * 70)

    columnas_run = [
        f"max_run_{canal}_{sufijo}"
        for canal in CANALES
    ]

    run_max = x[columnas_run].max(axis=1)

    print(
        run_max.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
                0.995,
                0.999
            ]
        ).to_string()
    )


# ============================================================
# DISTRIBUCIÓN DEL NÚMERO DE EPISODIOS
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    print()
    print("=" * 70)
    print(
        f"NÚMERO DE EPISODIOS > {umbral} µV"
    )
    print("=" * 70)

    columnas_episodios = [
        f"episodios_{canal}_{sufijo}"
        for canal in CANALES
    ]

    episodios_max = x[columnas_episodios].max(axis=1)

    print(
        episodios_max.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
                0.995,
                0.999
            ]
        ).to_string()
    )


# ============================================================
# ARCHIVOS MÁS EXTREMOS
# ============================================================

print()
print("=" * 70)
print("20 ARCHIVOS CON MAYOR PICO CENTRADO")
print("=" * 70)

columnas_mostrar = [
    "ruta",
    "n_muestras",
    "max_abs_centrado",
    "canal_max",
    "valor_max_centrado",
    "muestra_max",
    "n_canales_gt_120",
    "n_canales_gt_150",
]

print(
    x.sort_values(
        "max_abs_centrado",
        ascending=False
    )[columnas_mostrar]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# ARCHIVOS MÁS AFECTADOS POR DURACIÓN
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    columnas_run = [
        f"max_run_{canal}_{sufijo}"
        for canal in CANALES
    ]

    x["_run_max_temp"] = x[columnas_run].max(axis=1)

    print()
    print("=" * 70)
    print(
        f"20 MAYORES RACHAS > {umbral} µV"
    )
    print("=" * 70)

    print(
        x.sort_values(
            "_run_max_temp",
            ascending=False
        )[[
            "ruta",
            "n_muestras",
            "_run_max_temp",
            f"n_canales_gt_{sufijo}"
        ]]
        .head(20)
        .to_string(index=False)
    )

    x.drop(columns="_run_max_temp", inplace=True)


# ============================================================
# ARCHIVOS OMITIDOS
# ============================================================

print()
print("=" * 70)
print("ARCHIVOS OMITIDOS")
print("=" * 70)

for path, motivo in omitidos:
    print(f"{path} | {motivo}")

    # ============================================================
# CRUCE: EXTENSIÓN ESPACIAL VS DURACIÓN
# ============================================================

for umbral in UMBRALES:

    sufijo = str(int(umbral))

    columnas_run = [
        f"max_run_{canal}_{sufijo}"
        for canal in CANALES
    ]

    x["_run_max"] = x[columnas_run].max(axis=1)

    print()
    print("=" * 70)
    print(
        f"CRUCE CANALES AFECTADOS VS RACHA > {umbral} µV"
    )
    print("=" * 70)

    tabla = pd.crosstab(
        x[f"n_canales_gt_{sufijo}"],
        pd.cut(
            x["_run_max"],
            bins=[
                -1,
                0,
                1,
                5,
                10,
                20,
                50,
                100,
                150,
                1000
            ],
            labels=[
                "0",
                "1",
                "2-5",
                "6-10",
                "11-20",
                "21-50",
                "51-100",
                "101-150",
                ">150"
            ]
        )
    )

    print(tabla.to_string())

    x.drop(columns="_run_max", inplace=True)