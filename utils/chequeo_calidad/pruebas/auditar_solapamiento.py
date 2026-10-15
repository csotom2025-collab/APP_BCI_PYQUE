# ============================================================
# AUDITORÍA DE SOLAPAMIENTO TEMPORAL ENTRE CANALES
# ============================================================
#
# Objetivo:
#   Caracterizar cuántos canales superan simultáneamente
#   determinados niveles exploratorios de amplitud.
#
# IMPORTANTE:
#   Este script NO:
#   - elimina archivos
#   - modifica archivos fuente
#   - clasifica calidad
#   - asigna scores
#   - define umbrales de rechazo
#   - interpreta causalmente los eventos
#
# El análisis se realiza después de centrar cada canal
# restando su MEDIANA dentro del trial.
#
# Umbrales exploratorios:
#   120 uV
#   150 uV
#
# Las duraciones se expresan en MUESTRAS, no en ms,
# para no introducir todavía una conversión basada en fs.
#
# Salida:
#   C:\tmp_bci\resultados_calidad\auditoria_solapamiento.csv
#
# ============================================================

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\tmp_bci\captures")

REPORTE = Path(
    r"C:\tmp_bci\resultados_calidad\reporte.csv"
)

SALIDA = Path(
    r"C:\tmp_bci\resultados_calidad\auditoria_solapamiento.csv"
)

UMBRAL_AMPLITUD = [120.0, 150.0]

CANALES = [
    "F3",
    "FC5",
    "AF3",
    "F7",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "F8",
    "AF4",
    "FC6",
    "F4",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_ruta(ruta):
    """
    Convierte la ruta del reporte en una ruta absoluta
    respecto a ROOT cuando sea necesario.
    """

    ruta = Path(str(ruta))

    if ruta.is_absolute():
        return ruta

    return ROOT / ruta


def convertir_numerico(df, columnas):
    """
    Convierte las columnas EEG a valores numéricos.
    """

    salida = df[columnas].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return salida


def max_run_boolean(mask):
    """
    Devuelve la longitud máxima de una secuencia consecutiva
    de True.

    Ejemplo:

        [False, True, True, False, True]

    devuelve:

        2
    """

    if len(mask) == 0:
        return 0

    max_run = 0
    run_actual = 0

    for valor in mask:
        if valor:
            run_actual += 1

            if run_actual > max_run:
                max_run = run_actual

        else:
            run_actual = 0

    return max_run


def resumen_simultaneidad(
    n_simultaneos,
    n_muestras
):
    """
    Calcula todas las métricas descriptivas de simultaneidad
    para un trial y un umbral.

    n_simultaneos:
        array de longitud n_muestras.

        Cada posición contiene el número de canales que
        superan simultáneamente el umbral en esa muestra.
    """

    if n_muestras == 0:
        return {
            "max_canales_simultaneos": np.nan,

            "media_canales_simultaneos": np.nan,
            "mediana_canales_simultaneos": np.nan,
            "p95_canales_simultaneos": np.nan,

            "n_muestras_1plus": 0,
            "n_muestras_2plus": 0,
            "n_muestras_3plus": 0,
            "n_muestras_5plus": 0,
            "n_muestras_10plus": 0,
            "n_muestras_14": 0,

            "pct_muestras_1plus": np.nan,
            "pct_muestras_2plus": np.nan,
            "pct_muestras_3plus": np.nan,
            "pct_muestras_5plus": np.nan,
            "pct_muestras_10plus": np.nan,
            "pct_muestras_14": np.nan,

            "run_max_1plus": 0,
            "run_max_2plus": 0,
            "run_max_3plus": 0,
            "run_max_5plus": 0,
            "run_max_10plus": 0,
            "run_max_14": 0,
        }

    n_simultaneos = np.asarray(
        n_simultaneos,
        dtype=int
    )

    resultados = {}

    # --------------------------------------------------------
    # Distribución del número de canales simultáneos
    # --------------------------------------------------------

    resultados["max_canales_simultaneos"] = int(
        np.max(n_simultaneos)
    )

    resultados["media_canales_simultaneos"] = float(
        np.mean(n_simultaneos)
    )

    resultados["mediana_canales_simultaneos"] = float(
        np.median(n_simultaneos)
    )

    resultados["p95_canales_simultaneos"] = float(
        np.percentile(n_simultaneos, 95)
    )

    # --------------------------------------------------------
    # Conteos y proporciones
    # --------------------------------------------------------

    niveles = [1, 2, 3, 5, 10, 14]

    for nivel in niveles:

        n = int(
            np.sum(n_simultaneos >= nivel)
        )

        resultados[
            f"n_muestras_{nivel}plus"
        ] = n

        resultados[
            f"pct_muestras_{nivel}plus"
        ] = float(
            100.0 * n / n_muestras
        )

    # El nivel 14 se conserva explícitamente con nombre
    # propio además del esquema general.
    resultados["n_muestras_14"] = int(
        np.sum(n_simultaneos == 14)
    )

    resultados["pct_muestras_14"] = float(
        100.0 *
        resultados["n_muestras_14"] /
        n_muestras
    )

    # --------------------------------------------------------
    # Duraciones máximas consecutivas
    # --------------------------------------------------------

    for nivel in [1, 2, 3, 5, 10]:

        mask = n_simultaneos >= nivel

        resultados[
            f"run_max_{nivel}plus"
        ] = max_run_boolean(mask)

    # Exactamente 14 canales
    mask_14 = n_simultaneos == 14

    resultados["run_max_14"] = max_run_boolean(
        mask_14
    )

    return resultados


# ============================================================
# PROCESAMIENTO DE UN ARCHIVO
# ============================================================

def procesar_archivo(
    ruta,
    umbrales
):
    """
    Lee un archivo EEG y calcula la simultaneidad
    entre canales para cada umbral.
    """

    df = pd.read_csv(ruta)

    # --------------------------------------------------------
    # Verificación de canales
    # --------------------------------------------------------

    canales_faltantes = [
        canal
        for canal in CANALES
        if canal not in df.columns
    ]

    if canales_faltantes:
        raise ValueError(
            "Faltan canales: "
            + ", ".join(canales_faltantes)
        )

    # --------------------------------------------------------
    # Conversión numérica
    # --------------------------------------------------------

    X = convertir_numerico(
        df,
        CANALES
    )

    # --------------------------------------------------------
    # Eliminar filas con NaN
    #
    # No debería ocurrir en el universo incluido,
    # pero se verifica explícitamente.
    # --------------------------------------------------------

    mascara_validas = ~X.isna().any(axis=1)

    X = X.loc[
        mascara_validas
    ]

    if len(X) == 0:
        raise ValueError(
            "No quedan muestras válidas."
        )

    # --------------------------------------------------------
    # Convertir a numpy
    # --------------------------------------------------------

    X = X.to_numpy(
        dtype=float
    )

    n_muestras = X.shape[0]

    # --------------------------------------------------------
    # CENTRADO POR MEDIANA
    #
    # Cada canal se centra independientemente.
    #
    # Esto permite estudiar amplitud relativa al nivel
    # central del propio canal sin usar el offset DC.
    # --------------------------------------------------------

    medianas = np.median(
        X,
        axis=0
    )

    X_centered = X - medianas

    resultados = []

    # ========================================================
    # CADA UMBRAL
    # ========================================================

    for umbral in umbrales:

        # ----------------------------------------------------
        # Máscara:
        #
        # filas    = muestras
        # columnas = canales
        #
        # True significa que ese canal supera el umbral
        # en esa muestra.
        # ----------------------------------------------------

        mascara = (
            np.abs(X_centered)
            > umbral
        )

        # ----------------------------------------------------
        # Número de canales afectados simultáneamente
        # en cada muestra.
        # ----------------------------------------------------

        n_simultaneos = np.sum(
            mascara,
            axis=1
        )

        # ----------------------------------------------------
        # Resumen
        # ----------------------------------------------------

        resumen = resumen_simultaneidad(
            n_simultaneos,
            n_muestras
        )

        resultados.append(
            {
                "umbral_uv": umbral,
                "n_muestras_analizadas": n_muestras,
                **resumen,
            }
        )

    return resultados


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDITORÍA DE SOLAPAMIENTO TEMPORAL ENTRE CANALES")
    print("=" * 70)

    print()
    print(f"ROOT:     {ROOT}")
    print(f"REPORTE:  {REPORTE}")
    print(f"SALIDA:   {SALIDA}")
    print()
    print(
        "Umbrales exploratorios:",
        ", ".join(
            f"{x:g} uV"
            for x in UMBRAL_AMPLITUD
        )
    )
    print()

    # ========================================================
    # VERIFICACIONES INICIALES
    # ========================================================

    if not ROOT.exists():
        print(
            f"ERROR: no existe ROOT:\n{ROOT}"
        )
        sys.exit(1)

    if not REPORTE.exists():
        print(
            f"ERROR: no existe reporte:\n{REPORTE}"
        )
        sys.exit(1)

    # ========================================================
    # CARGAR REPORTE
    # ========================================================

    print("Leyendo reporte...")

    reporte = pd.read_csv(
        REPORTE
    )

    if "incluido" not in reporte.columns:
        print(
            "ERROR: el reporte no contiene "
            "la columna 'incluido'."
        )
        sys.exit(1)

    if "ruta_relativa" not in reporte.columns:
        print(
            "ERROR: el reporte no contiene "
            "la columna 'ruta_relativa'."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Mantener únicamente universo incluido
    # --------------------------------------------------------

    incluidos = reporte[
        reporte["incluido"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
            "si"
        ])
    ].copy()

    print(
        f"Archivos en reporte: "
        f"{len(reporte)}"
    )

    print(
        f"Archivos incluidos: "
        f"{len(incluidos)}"
    )

    print()

    # ========================================================
    # PROCESAMIENTO
    # ========================================================

    resultados = []

    n_ok = 0
    n_error = 0

    errores = []

    total = len(incluidos)

    for posicion, (_, fila) in enumerate(
        incluidos.iterrows(),
        start=1
    ):

        ruta_relativa = fila[
            "ruta_relativa"
        ]

        ruta = normalizar_ruta(
            ruta_relativa
        )

        if posicion == 1 or posicion % 250 == 0 or posicion == total:
            print(
                f"[{posicion:>5}/{total}] "
                f"{ruta_relativa}"
            )

        try:

            registros = procesar_archivo(
                ruta,
                UMBRAL_AMPLITUD
            )

            # ------------------------------------------------
            # Metadatos del reporte
            # ------------------------------------------------

            for registro in registros:

                resultado = {
                    "ruta_relativa": ruta_relativa,

                    "sha256": fila.get(
                        "sha256",
                        ""
                    ),

                    "usuario": fila.get(
                        "usuario",
                        ""
                    ),

                    "subcarpeta": fila.get(
                        "subcarpeta",
                        ""
                    ),

                    "letra": fila.get(
                        "letra",
                        ""
                    ),

                    "trial_nombre": fila.get(
                        "trial_nombre",
                        ""
                    ),

                    "session_id": fila.get(
                        "session_id",
                        ""
                    ),

                    "acquisition_order": fila.get(
                        "acquisition_order",
                        ""
                    ),

                    **registro,
                }

                resultados.append(
                    resultado
                )

            n_ok += 1

        except Exception as exc:

            n_error += 1

            errores.append(
                {
                    "ruta_relativa": ruta_relativa,
                    "error": repr(exc),
                }
            )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # Orden de columnas
    # --------------------------------------------------------

    columnas_preferidas = [
        "ruta_relativa",
        "sha256",
        "usuario",
        "subcarpeta",
        "letra",
        "trial_nombre",
        "session_id",
        "acquisition_order",

        "umbral_uv",
        "n_muestras_analizadas",

        "max_canales_simultaneos",

        "media_canales_simultaneos",
        "mediana_canales_simultaneos",
        "p95_canales_simultaneos",

        "n_muestras_1plus",
        "pct_muestras_1plus",

        "n_muestras_2plus",
        "pct_muestras_2plus",

        "n_muestras_3plus",
        "pct_muestras_3plus",

        "n_muestras_5plus",
        "pct_muestras_5plus",

        "n_muestras_10plus",
        "pct_muestras_10plus",

        "n_muestras_14",
        "pct_muestras_14",

        "run_max_1plus",
        "run_max_2plus",
        "run_max_3plus",
        "run_max_5plus",
        "run_max_10plus",
        "run_max_14",
    ]

    columnas_existentes = [
        columna
        for columna in columnas_preferidas
        if columna in df_resultados.columns
    ]

    columnas_restantes = [
        columna
        for columna in df_resultados.columns
        if columna not in columnas_existentes
    ]

    df_resultados = df_resultados[
        columnas_existentes
        + columnas_restantes
    ]

    # ========================================================
    # GUARDAR
    # ========================================================

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_resultados.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(
        f"Archivos procesados correctamente: "
        f"{n_ok}"
    )

    print(
        f"Archivos con error: "
        f"{n_error}"
    )

    print(
        f"Filas generadas: "
        f"{len(df_resultados)}"
    )

    print()
    print(
        f"Archivo generado:\n{SALIDA}"
    )

    # ========================================================
    # RESUMEN POR UMBRAL
    # ========================================================

    if not df_resultados.empty:

        print()
        print("-" * 70)
        print("RESUMEN POR UMBRAL")
        print("-" * 70)

        for umbral in UMBRAL_AMPLITUD:

            sub = df_resultados[
                df_resultados["umbral_uv"]
                == umbral
            ]

            if sub.empty:
                continue

            print()
            print(
                f"UMBRAL: {umbral:g} uV"
            )

            print(
                f"  Archivos: "
                f"{len(sub)}"
            )

            print(
                "  max_canales_simultaneos:"
            )

            print(
                f"    media   = "
                f"{sub['max_canales_simultaneos'].mean():.3f}"
            )

            print(
                f"    mediana = "
                f"{sub['max_canales_simultaneos'].median():.3f}"
            )

            print(
                f"    P95     = "
                f"{sub['max_canales_simultaneos'].quantile(0.95):.3f}"
            )

            print(
                f"    máximo  = "
                f"{sub['max_canales_simultaneos'].max():.0f}"
            )

            print()

            for nivel in [1, 2, 3, 5, 10, 14]:

                columna = (
                    f"pct_muestras_{nivel}plus"
                )

                if columna in sub.columns:

                    print(
                        f"  % muestras "
                        f">= {nivel} canales: "
                        f"media="
                        f"{sub[columna].mean():.4f}%"
                        f" | P95="
                        f"{sub[columna].quantile(0.95):.4f}%"
                        f" | max="
                        f"{sub[columna].max():.4f}%"
                    )

            print()

            for nivel in [2, 5, 10, 14]:

                columna = (
                    f"run_max_{nivel}plus"
                )

                if columna in sub.columns:

                    print(
                        f"  run máximo "
                        f">= {nivel} canales: "
                        f"mediana="
                        f"{sub[columna].median():.1f} "
                        f"| P95="
                        f"{sub[columna].quantile(0.95):.1f} "
                        f"| max="
                        f"{sub[columna].max():.0f} muestras"
                    )

    # ========================================================
    # ERRORES
    # ========================================================

    if errores:

        ruta_errores = (
            SALIDA.parent
            / "auditoria_solapamiento_errores.csv"
        )

        pd.DataFrame(
            errores
        ).to_csv(
            ruta_errores,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "Se generó también el archivo "
            "de errores:"
        )

        print(
            ruta_errores
        )

    print()
    print("=" * 70)
    print("AUDITORÍA FINALIZADA")
    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Proceso interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception:

        print()
        print(
            "ERROR NO CONTROLADO:"
        )

        traceback.print_exc()

        sys.exit(1)