# ============================================================
# AUDITORÍA DE CORRELACIONES DE PEARSON ENTRE LOS 14 CANALES
# ============================================================
#
# Objetivo:
#   Calcular las correlaciones de Pearson entre todos los pares
#   de los 14 canales EEG para cada trial.
#
# 14 canales -> 91 pares únicos
#
# Este script es DESCRIPTIVO.
#
# NO:
#   - elimina archivos
#   - modifica archivos fuente
#   - clasifica calidad
#   - asigna scores
#   - define umbrales de rechazo
#   - interpreta causalmente las correlaciones
#
# Se generan:
#
#   1) correlaciones_pearson_pares.csv
#      Una fila por archivo y par de canales.
#
#   2) correlaciones_pearson_resumen.csv
#      Una fila por archivo y etapa.
#
# Etapas:
#   crudo
#   cent
#
# cent:
#   cada canal se centra restando su mediana dentro del trial.
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

ROOT = Path(
    r"C:\tmp_bci\captures"
)

REPORTE = Path(
    r"C:\tmp_bci\resultados_calidad\reporte.csv"
)

SALIDA_PARES = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_pearson_pares.csv"
)

SALIDA_RESUMEN = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_pearson_resumen.csv"
)


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


ETAPAS = [
    "crudo",
    "cent",
]


# Umbrales descriptivos.
# NO son criterios de calidad.
UMBRALES_R = [
    0.3,
    0.5,
    0.7,
    0.9,
]


# ============================================================
# FUNCIONES
# ============================================================

def normalizar_ruta(ruta):
    """
    Convierte una ruta relativa del reporte
    en una ruta absoluta respecto a ROOT.
    """

    ruta = Path(str(ruta))

    if ruta.is_absolute():
        return ruta

    return ROOT / ruta


def convertir_numerico(df):
    """
    Convierte los 14 canales a valores numéricos.
    """

    return df[CANALES].apply(
        pd.to_numeric,
        errors="coerce"
    )


def calcular_matriz_pearson(X):
    """
    Calcula la matriz de correlación de Pearson.

    X:
        matriz [muestras x canales]

    Devuelve:
        matriz [canales x canales]
    """

    n_muestras = X.shape[0]

    if n_muestras < 2:
        return np.full(
            (len(CANALES), len(CANALES)),
            np.nan
        )

    # np.corrcoef:
    # rowvar=False -> cada columna es una variable/canal
    matriz = np.corrcoef(
        X,
        rowvar=False
    )

    return matriz


def extraer_pares(matriz):
    """
    Extrae solamente la parte superior de la matriz,
    evitando duplicar pares.

    Ejemplo:

        F3-FC5
        F3-AF3
        ...
        FC5-AF3
        ...

    No incluye:

        F3-F3

    ni:

        FC5-F3

    """

    registros = []

    for i in range(len(CANALES)):

        for j in range(i + 1, len(CANALES)):

            r = matriz[i, j]

            registros.append(
                {
                    "canal_1": CANALES[i],
                    "canal_2": CANALES[j],
                    "pearson_r": r,
                    "pearson_abs_r": (
                        abs(r)
                        if np.isfinite(r)
                        else np.nan
                    ),
                }
            )

    return registros


def calcular_resumen(matriz):
    """
    Calcula estadísticas descriptivas de los 91 pares.
    """

    # --------------------------------------------------------
    # Extraer triángulo superior sin diagonal
    # --------------------------------------------------------

    indices = np.triu_indices(
        len(CANALES),
        k=1
    )

    valores = matriz[indices]

    valores = valores[
        np.isfinite(valores)
    ]

    if len(valores) == 0:

        return {
            "n_pares_validos": 0,

            "pearson_media": np.nan,
            "pearson_mediana": np.nan,
            "pearson_p05": np.nan,
            "pearson_p25": np.nan,
            "pearson_p75": np.nan,
            "pearson_p95": np.nan,
            "pearson_min": np.nan,
            "pearson_max": np.nan,

            "abs_r_media": np.nan,
            "abs_r_mediana": np.nan,
            "abs_r_p05": np.nan,
            "abs_r_p25": np.nan,
            "abs_r_p75": np.nan,
            "abs_r_p95": np.nan,
            "abs_r_max": np.nan,

            "n_abs_r_ge_03": 0,
            "n_abs_r_ge_05": 0,
            "n_abs_r_ge_07": 0,
            "n_abs_r_ge_09": 0,

            "pct_abs_r_ge_03": np.nan,
            "pct_abs_r_ge_05": np.nan,
            "pct_abs_r_ge_07": np.nan,
            "pct_abs_r_ge_09": np.nan,
        }

    abs_valores = np.abs(
        valores
    )

    n = len(valores)

    resultado = {
        "n_pares_validos": n,

        "pearson_media": float(
            np.mean(valores)
        ),

        "pearson_mediana": float(
            np.median(valores)
        ),

        "pearson_p05": float(
            np.percentile(valores, 5)
        ),

        "pearson_p25": float(
            np.percentile(valores, 25)
        ),

        "pearson_p75": float(
            np.percentile(valores, 75)
        ),

        "pearson_p95": float(
            np.percentile(valores, 95)
        ),

        "pearson_min": float(
            np.min(valores)
        ),

        "pearson_max": float(
            np.max(valores)
        ),

        "abs_r_media": float(
            np.mean(abs_valores)
        ),

        "abs_r_mediana": float(
            np.median(abs_valores)
        ),

        "abs_r_p05": float(
            np.percentile(abs_valores, 5)
        ),

        "abs_r_p25": float(
            np.percentile(abs_valores, 25)
        ),

        "abs_r_p75": float(
            np.percentile(abs_valores, 75)
        ),

        "abs_r_p95": float(
            np.percentile(abs_valores, 95)
        ),

        "abs_r_max": float(
            np.max(abs_valores)
        ),
    }

    # --------------------------------------------------------
    # Conteos por magnitud absoluta de r
    # --------------------------------------------------------

    for umbral in UMBRALES_R:

        cantidad = int(
            np.sum(
                abs_valores >= umbral
            )
        )

        resultado[
            f"n_abs_r_ge_{str(umbral).replace('.', '')}"
        ] = cantidad

        resultado[
            f"pct_abs_r_ge_{str(umbral).replace('.', '')}"
        ] = float(
            100.0 * cantidad / n
        )

    return resultado


def procesar_archivo(ruta):
    """
    Procesa un archivo y devuelve:

        pares
        resumen

    para crudo y cent.
    """

    df = pd.read_csv(
        ruta
    )

    # --------------------------------------------------------
    # Verificar canales
    # --------------------------------------------------------

    faltantes = [
        canal
        for canal in CANALES
        if canal not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Faltan canales: "
            + ", ".join(faltantes)
        )

    # --------------------------------------------------------
    # Convertir a numérico
    # --------------------------------------------------------

    X_df = convertir_numerico(
        df
    )

    # --------------------------------------------------------
    # Mantener únicamente filas completas
    # --------------------------------------------------------

    mascara_validas = (
        ~X_df.isna().any(axis=1)
    )

    X_df = X_df.loc[
        mascara_validas
    ]

    if len(X_df) < 2:

        raise ValueError(
            "Menos de 2 muestras válidas."
        )

    # --------------------------------------------------------
    # Matriz cruda
    # --------------------------------------------------------

    X_crudo = X_df.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Matriz centrada
    # --------------------------------------------------------

    medianas = np.median(
        X_crudo,
        axis=0
    )

    X_cent = (
        X_crudo
        - medianas
    )

    datos = {}

    for nombre_etapa, X in [
        ("crudo", X_crudo),
        ("cent", X_cent),
    ]:

        matriz = calcular_matriz_pearson(
            X
        )

        pares = extraer_pares(
            matriz
        )

        resumen = calcular_resumen(
            matriz
        )

        datos[nombre_etapa] = {
            "matriz": matriz,
            "pares": pares,
            "resumen": resumen,
        }

    return {
        "n_muestras_analizadas": len(X_df),
        "datos": datos,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDITORÍA DE CORRELACIONES DE PEARSON")
    print("=" * 70)

    print()
    print(f"ROOT:    {ROOT}")
    print(f"REPORTE: {REPORTE}")
    print()
    print(
        "Canales:",
        len(CANALES)
    )

    print(
        "Pares por trial:",
        len(CANALES) * (
            len(CANALES) - 1
        ) // 2
    )

    print()

    # ========================================================
    # VERIFICACIONES
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

    reporte = pd.read_csv(
        REPORTE
    )

    if "incluido" not in reporte.columns:

        print(
            "ERROR: falta columna 'incluido'."
        )

        sys.exit(1)

    if "ruta_relativa" not in reporte.columns:

        print(
            "ERROR: falta columna "
            "'ruta_relativa'."
        )

        sys.exit(1)

    incluidos = reporte[
        reporte["incluido"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
            "si",
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
    # RESULTADOS
    # ========================================================

    resultados_pares = []
    resultados_resumen = []

    errores = []

    total = len(incluidos)

    # ========================================================
    # PROCESAR
    # ========================================================

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

        if (
            posicion == 1
            or posicion % 250 == 0
            or posicion == total
        ):

            print(
                f"[{posicion:>5}/{total}] "
                f"{ruta_relativa}"
            )

        try:

            resultado = procesar_archivo(
                ruta
            )

            n_muestras = resultado[
                "n_muestras_analizadas"
            ]

            datos = resultado[
                "datos"
            ]

            # =================================================
            # CADA ETAPA
            # =================================================

            for etapa in ETAPAS:

                matriz = datos[
                    etapa
                ]["matriz"]

                pares = datos[
                    etapa
                ]["pares"]

                resumen = datos[
                    etapa
                ]["resumen"]

                # -------------------------------------------------
                # Pares
                # -------------------------------------------------

                for par in pares:

                    resultados_pares.append(
                        {
                            "ruta_relativa":
                                ruta_relativa,

                            "sha256":
                                fila.get(
                                    "sha256",
                                    ""
                                ),

                            "usuario":
                                fila.get(
                                    "usuario",
                                    ""
                                ),

                            "subcarpeta":
                                fila.get(
                                    "subcarpeta",
                                    ""
                                ),

                            "letra":
                                fila.get(
                                    "letra",
                                    ""
                                ),

                            "trial_nombre":
                                fila.get(
                                    "trial_nombre",
                                    ""
                                ),

                            "session_id":
                                fila.get(
                                    "session_id",
                                    ""
                                ),

                            "acquisition_order":
                                fila.get(
                                    "acquisition_order",
                                    ""
                                ),

                            "etapa":
                                etapa,

                            "n_muestras_analizadas":
                                n_muestras,

                            **par,
                        }
                    )

                # -------------------------------------------------
                # Resumen
                # -------------------------------------------------

                resultados_resumen.append(
                    {
                        "ruta_relativa":
                            ruta_relativa,

                        "sha256":
                            fila.get(
                                "sha256",
                                ""
                            ),

                        "usuario":
                            fila.get(
                                "usuario",
                                ""
                            ),

                        "subcarpeta":
                            fila.get(
                                "subcarpeta",
                                ""
                            ),

                        "letra":
                            fila.get(
                                "letra",
                                ""
                            ),

                        "trial_nombre":
                            fila.get(
                                "trial_nombre",
                                ""
                            ),

                        "session_id":
                            fila.get(
                                "session_id",
                                ""
                            ),

                        "acquisition_order":
                            fila.get(
                                "acquisition_order",
                                ""
                            ),

                        "etapa":
                            etapa,

                        "n_muestras_analizadas":
                            n_muestras,

                        **resumen,
                    }
                )

        except Exception as exc:

            errores.append(
                {
                    "ruta_relativa":
                        ruta_relativa,

                    "error":
                        repr(exc),
                }
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    df_pares = pd.DataFrame(
        resultados_pares
    )

    df_resumen = pd.DataFrame(
        resultados_resumen
    )

    # ========================================================
    # ORDENAR
    # ========================================================

    if not df_pares.empty:

        df_pares = df_pares.sort_values(
            [
                "ruta_relativa",
                "etapa",
                "canal_1",
                "canal_2",
            ]
        )

    if not df_resumen.empty:

        df_resumen = df_resumen.sort_values(
            [
                "ruta_relativa",
                "etapa",
            ]
        )

    # ========================================================
    # GUARDAR
    # ========================================================

    SALIDA_PARES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_pares.to_csv(
        SALIDA_PARES,
        index=False,
        encoding="utf-8-sig"
    )

    df_resumen.to_csv(
        SALIDA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(
        f"Filas de pares: "
        f"{len(df_pares)}"
    )

    print(
        f"Filas de resumen: "
        f"{len(df_resumen)}"
    )

    print(
        f"Errores: "
        f"{len(errores)}"
    )

    print()
    print(
        "Archivo de pares:"
    )

    print(
        SALIDA_PARES
    )

    print()
    print(
        "Archivo de resumen:"
    )

    print(
        SALIDA_RESUMEN
    )

    # ========================================================
    # RESUMEN ESTADÍSTICO GLOBAL
    # ========================================================

    if not df_resumen.empty:

        print()
        print("-" * 70)
        print("DISTRIBUCIÓN GLOBAL")
        print("-" * 70)

        for etapa in ETAPAS:

            sub = df_resumen[
                df_resumen["etapa"]
                == etapa
            ]

            if sub.empty:
                continue

            print()
            print(
                f"ETAPA: {etapa}"
            )

            print(
                f"  Trials: {len(sub)}"
            )

            print()
            print(
                "  Pearson r:"
            )

            print(
                f"    media   = "
                f"{sub['pearson_media'].mean():.4f}"
            )

            print(
                f"    mediana = "
                f"{sub['pearson_mediana'].median():.4f}"
            )

            print(
                f"    P05     = "
                f"{sub['pearson_p05'].median():.4f}"
            )

            print(
                f"    P95     = "
                f"{sub['pearson_p95'].median():.4f}"
            )

            print(
                f"    mínimo  = "
                f"{sub['pearson_min'].min():.4f}"
            )

            print(
                f"    máximo  = "
                f"{sub['pearson_max'].max():.4f}"
            )

            print()
            print(
                "  |r|:"
            )

            print(
                f"    mediana = "
                f"{sub['abs_r_mediana'].median():.4f}"
            )

            print(
                f"    P95     = "
                f"{sub['abs_r_p95'].median():.4f}"
            )

            print(
                f"    máximo  = "
                f"{sub['abs_r_max'].max():.4f}"
            )

            print()

            for umbral in UMBRALES_R:

                sufijo = (
                    str(umbral)
                    .replace(".", "")
                )

                columna = (
                    f"pct_abs_r_ge_{sufijo}"
                )

                if columna in sub.columns:

                    print(
                        f"    |r| >= {umbral:.1f}: "
                        f"mediana por trial = "
                        f"{sub[columna].median():.2f}%"
                    )

    # ========================================================
    # ERRORES
    # ========================================================

    if errores:

        salida_errores = (
            SALIDA_RESUMEN.parent
            / "correlaciones_pearson_errores.csv"
        )

        pd.DataFrame(
            errores
        ).to_csv(
            salida_errores,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "Archivo de errores:"
        )

        print(
            salida_errores
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