# ================================================================
# AUDITORÍA PEARSON:
# CLASE × CANAL × RELACIÓN ENTRE TRIALS
# ================================================================
#
# OBJETIVO
#
# Determinar, para CADA CLASE y CADA CANAL, cómo se comporta la
# correlación temporal Pearson entre trials según:
#
#   1. mismo usuario + misma clase
#   2. mismo usuario + clase diferente
#   3. usuario diferente + misma clase
#   4. usuario diferente + clase diferente
#
# Esto permite separar:
#
#   - efecto individual del usuario
#   - efecto de la clase
#   - efecto del canal
#   - similitud general / artefactos comunes
#
# IMPORTANTE
#
# No se guardan los ~23 millones de pares como DataFrame.
# Las relaciones se calculan mediante máscaras vectorizadas.
#
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ================================================================
# CONFIGURACIÓN
# ================================================================

ROOT = Path(r"C:\tmp_bci\captures")

REPORTE = Path(
    r"C:\tmp_bci\resultados_calidad\reporte.csv"
)

SALIDA_RESUMEN = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\resumen_clase_canal_relacion.csv"
)

SALIDA_CLASE = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\resumen_clase_relacion.csv"
)

SALIDA_CANAL = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\resumen_canal_relacion.csv"
)

SALIDA_MATRICES = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_clase_canal_relacion.npz"
)


# ================================================================
# CANALES
# ================================================================

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


N_MUESTRAS = 256


# ================================================================
# COLUMNA DE CLASE
# ================================================================

COLUMNA_CLASE = "letra"


# ================================================================
# RELACIONES
# ================================================================

RELACIONES = [
    "mismo_usuario_misma_clase",
    "mismo_usuario_clase_diferente",
    "usuario_diferente_misma_clase",
    "usuario_diferente_clase_diferente",
]


# ================================================================
# LECTURA EEG
# ================================================================

def leer_csv_eeg(path):

    df = pd.read_csv(path)

    faltantes = [
        c for c in CANALES
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan canales: {faltantes}"
        )

    X = (
        df[CANALES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    if np.isnan(X).any():

        raise ValueError(
            "Se encontraron NaN."
        )

    return X


# ================================================================
# CENTRADO POR MEDIANA
# ================================================================

def centrar_por_mediana(X):

    medianas = np.median(
        X,
        axis=0
    )

    return X - medianas


# ================================================================
# CORRELACIÓN PEARSON
# ================================================================

def pearson_matrix(X):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )

        R = np.corrcoef(X)

    return R


# ================================================================
# ESTADÍSTICAS
# ================================================================

def calcular_estadisticas(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "n_pares": 0,
            "r_media": np.nan,
            "r_mediana": np.nan,
            "r_p05": np.nan,
            "r_p25": np.nan,
            "r_p75": np.nan,
            "r_p95": np.nan,
            "r_min": np.nan,
            "r_max": np.nan,

            "abs_r_media": np.nan,
            "abs_r_mediana": np.nan,
            "abs_r_p05": np.nan,
            "abs_r_p25": np.nan,
            "abs_r_p75": np.nan,
            "abs_r_p95": np.nan,
            "abs_r_max": np.nan,

            "pct_abs_r_ge_03": np.nan,
            "pct_abs_r_ge_05": np.nan,
            "pct_abs_r_ge_07": np.nan,
            "pct_abs_r_ge_09": np.nan,
        }

    abs_r = np.abs(values)

    return {

        "n_pares":
            len(values),

        # --------------------------------------------------------
        # r firmado
        # --------------------------------------------------------

        "r_media":
            float(np.mean(values)),

        "r_mediana":
            float(np.median(values)),

        "r_p05":
            float(np.percentile(values, 5)),

        "r_p25":
            float(np.percentile(values, 25)),

        "r_p75":
            float(np.percentile(values, 75)),

        "r_p95":
            float(np.percentile(values, 95)),

        "r_min":
            float(np.min(values)),

        "r_max":
            float(np.max(values)),

        # --------------------------------------------------------
        # |r|
        # --------------------------------------------------------

        "abs_r_media":
            float(np.mean(abs_r)),

        "abs_r_mediana":
            float(np.median(abs_r)),

        "abs_r_p05":
            float(np.percentile(abs_r, 5)),

        "abs_r_p25":
            float(np.percentile(abs_r, 25)),

        "abs_r_p75":
            float(np.percentile(abs_r, 75)),

        "abs_r_p95":
            float(np.percentile(abs_r, 95)),

        "abs_r_max":
            float(np.max(abs_r)),

        # --------------------------------------------------------
        # umbrales descriptivos
        # --------------------------------------------------------

        "pct_abs_r_ge_03":
            float(
                100 * np.mean(abs_r >= 0.3)
            ),

        "pct_abs_r_ge_05":
            float(
                100 * np.mean(abs_r >= 0.5)
            ),

        "pct_abs_r_ge_07":
            float(
                100 * np.mean(abs_r >= 0.7)
            ),

        "pct_abs_r_ge_09":
            float(
                100 * np.mean(abs_r >= 0.9)
            ),
    }


# ================================================================
# CONSTRUIR MÁSCARAS DE RELACIÓN
# ================================================================

def construir_mascaras(usuarios, clases):

    usuarios = np.asarray(
        usuarios,
        dtype=object
    )

    clases = np.asarray(
        clases,
        dtype=object
    )

    mismo_usuario = (
        usuarios[:, None]
        ==
        usuarios[None, :]
    )

    misma_clase = (
        clases[:, None]
        ==
        clases[None, :]
    )

    # ------------------------------------------------------------
    # Solo triángulo superior.
    #
    # Evita contar:
    #
    # A-B
    # B-A
    #
    # dos veces.
    # ------------------------------------------------------------

    superior = np.triu(
        np.ones(
            (len(usuarios), len(usuarios)),
            dtype=bool
        ),
        k=1
    )

    mascaras = {

        "mismo_usuario_misma_clase":
            superior
            & mismo_usuario
            & misma_clase,

        "mismo_usuario_clase_diferente":
            superior
            & mismo_usuario
            & (~misma_clase),

        "usuario_diferente_misma_clase":
            superior
            & (~mismo_usuario)
            & misma_clase,

        "usuario_diferente_clase_diferente":
            superior
            & (~mismo_usuario)
            & (~misma_clase),
    }

    return mascaras


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 75)
    print("AUDITORÍA PEARSON POR CLASE × CANAL × RELACIÓN")
    print("=" * 75)

    print()

    # ============================================================
    # REPORTE
    # ============================================================

    reporte = pd.read_csv(
        REPORTE
    )

    if "incluido" not in reporte.columns:

        raise ValueError(
            "reporte.csv no contiene 'incluido'."
        )

    incluidos = reporte[
        reporte["incluido"]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1", "yes", "si"]
        )
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

    if COLUMNA_CLASE not in incluidos.columns:

        raise ValueError(
            f"No existe la columna de clase: "
            f"{COLUMNA_CLASE}"
        )

    # ============================================================
    # CARGAR TRIALS
    # ============================================================

    X_trials = []

    usuarios = []

    clases = []

    rutas = []

    errores = []

    print("-" * 75)
    print("CARGANDO TRIALS")
    print("-" * 75)

    for contador, (_, row) in enumerate(
        incluidos.iterrows(),
        start=1
    ):

        ruta_relativa = str(
            row["ruta_relativa"]
        )

        path = ROOT / Path(
            ruta_relativa
        )

        if (
            contador == 1
            or contador % 250 == 0
        ):

            print(
                f"[{contador:5d}/"
                f"{len(incluidos)}] "
                f"{ruta_relativa}"
            )

        if not path.exists():

            errores.append({
                "ruta": ruta_relativa,
                "motivo": "no existe"
            })

            continue

        try:

            X = leer_csv_eeg(
                path
            )

            if X.shape[0] != N_MUESTRAS:

                continue

            X = centrar_por_mediana(
                X
            )

            clase = row[
                COLUMNA_CLASE
            ]

            usuario = row[
                "usuario"
            ]

            X_trials.append(
                X
            )

            usuarios.append(
                usuario
            )

            clases.append(
                clase
            )

            rutas.append(
                ruta_relativa
            )

        except Exception as e:

            errores.append({
                "ruta": ruta_relativa,
                "motivo": str(e)
            })

    # ============================================================
    # TENSOR
    # ============================================================

    X_all = np.asarray(
        X_trials,
        dtype=np.float32
    )

    usuarios = np.asarray(
        usuarios,
        dtype=object
    )

    clases = np.asarray(
        clases,
        dtype=object
    )

    n_trials = len(
        X_all
    )

    print()
    print("-" * 75)
    print("RESUMEN DE CARGA")
    print("-" * 75)

    print(
        f"Trials nominales: "
        f"{n_trials}"
    )

    print(
        f"Errores: "
        f"{len(errores)}"
    )

    print()

    print(
        "Tensor:"
    )

    print(
        f"  {n_trials} trials × "
        f"{N_MUESTRAS} muestras × "
        f"{len(CANALES)} canales"
    )

    # ============================================================
    # CLASES
    # ============================================================

    clases_unicas = sorted(
        pd.unique(clases),
        key=lambda x: str(x)
    )

    print()
    print(
        f"Clases: "
        f"{len(clases_unicas)}"
    )

    # ============================================================
    # MÁSCARAS
    # ============================================================

    print()
    print("-" * 75)
    print("CONSTRUYENDO RELACIONES")
    print("-" * 75)

    mascaras = construir_mascaras(
        usuarios,
        clases
    )

    for relacion in RELACIONES:

        n = int(
            np.sum(
                mascaras[relacion]
            )
        )

        print(
            f"{relacion:45s} {n:,}"
        )

    total_pares = (
        n_trials
        * (n_trials - 1)
        // 2
    )

    total_clasificado = sum(
        np.sum(m)
        for m in mascaras.values()
    )

    print()
    print(
        f"Total pares esperado: "
        f"{total_pares:,}"
    )

    print(
        f"Total clasificado: "
        f"{total_clasificado:,}"
    )

    if total_pares != total_clasificado:

        raise RuntimeError(
            "ERROR: no todos los pares fueron clasificados."
        )

    print(
        "OK: todos los pares están clasificados exactamente una vez."
    )

    # ============================================================
    # RESULTADOS
    # ============================================================

    resultados = []

    # ============================================================
    # MATRICES / VALORES POR CLASE
    #
    # Guardaremos solamente estadísticas, no los millones de
    # valores individuales.
    # ============================================================

    print()
    print("-" * 75)
    print("CALCULANDO CLASE × CANAL × RELACIÓN")
    print("-" * 75)

    for canal_idx, canal in enumerate(
        CANALES
    ):

        print(
            f"Canal: {canal}"
        )

        X_canal = X_all[
            :,
            :,
            canal_idx
        ]

        # --------------------------------------------------------
        # Pearson global del canal
        # --------------------------------------------------------

        R = pearson_matrix(
            X_canal
        )

        # --------------------------------------------------------
        # Para cada clase
        # --------------------------------------------------------

        for clase in clases_unicas:

            # trials pertenecientes a esta clase

            indices_clase = np.where(
                clases == clase
            )[0]

            if len(indices_clase) < 2:

                continue

            # ----------------------------------------------------
            # Mismo usuario + misma clase
            #
            # Como todos los trials aquí son de la misma clase,
            # solo quedan pares de mismo usuario.
            # ----------------------------------------------------

            for relacion in RELACIONES:

                mascara_global = mascaras[
                    relacion
                ]

                # Solo pares que:
                #
                # 1. pertenecen a esta clase
                # 2. cumplen la relación

                filas = indices_clase

                submascara = (
                    mascara_global[
                        np.ix_(
                            filas,
                            filas
                        )
                    ]
                )

                valores = R[
                    np.ix_(
                        filas,
                        filas
                    )
                ][
                    submascara
                ]

                stats = calcular_estadisticas(
                    valores
                )

                resultados.append({

                    "clase":
                        str(clase),

                    "canal":
                        canal,

                    "relacion":
                        relacion,

                    "n_trials_clase":
                        len(indices_clase),

                    **stats
                })

    # ============================================================
    # DATAFRAME
    # ============================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    # ============================================================
    # GUARDAR RESULTADO PRINCIPAL
    # ============================================================

    SALIDA_RESUMEN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados_df.to_csv(
        SALIDA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # RESUMEN CLASE × RELACIÓN
    #
    # Promedia descriptivamente los canales.
    #
    # IMPORTANTE:
    # Esto es solo resumen; el resultado principal es el CSV
    # clase × canal × relación.
    # ============================================================

    resumen_clase = (
        resultados_df
        .groupby(
            [
                "clase",
                "relacion"
            ],
            as_index=False
        )
        .agg({

            "n_pares": "sum",

            "r_mediana": "median",
            "r_p05": "median",
            "r_p95": "median",

            "abs_r_mediana": "median",
            "abs_r_p95": "median",

            "pct_abs_r_ge_03": "median",
            "pct_abs_r_ge_05": "median",
            "pct_abs_r_ge_07": "median",
            "pct_abs_r_ge_09": "median",
        })
    )

    resumen_clase.to_csv(
        SALIDA_CLASE,
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # RESUMEN CANAL × RELACIÓN
    # ============================================================

    resumen_canal = (
        resultados_df
        .groupby(
            [
                "canal",
                "relacion"
            ],
            as_index=False
        )
        .agg({

            "n_pares": "sum",

            "r_mediana": "median",
            "r_p05": "median",
            "r_p95": "median",

            "abs_r_mediana": "median",
            "abs_r_p95": "median",

            "pct_abs_r_ge_03": "median",
            "pct_abs_r_ge_05": "median",
            "pct_abs_r_ge_07": "median",
            "pct_abs_r_ge_09": "median",
        })
    )

    resumen_canal.to_csv(
        SALIDA_CANAL,
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # RESULTADO
    # ============================================================

    print()
    print("=" * 75)
    print("RESULTADO")
    print("=" * 75)

    print()
    print(
        "Archivo principal:"
    )

    print(
        f"  {SALIDA_RESUMEN}"
    )

    print()
    print(
        "Resumen por clase:"
    )

    print(
        f"  {SALIDA_CLASE}"
    )

    print()
    print(
        "Resumen por canal:"
    )

    print(
        f"  {SALIDA_CANAL}"
    )

    # ============================================================
    # MOSTRAR EJEMPLO DE LA CLASE A
    # ============================================================

    if "A" in resultados_df["clase"].values:

        print()
        print("-" * 75)
        print("EJEMPLO: CLASE A")
        print("-" * 75)

        ejemplo = resultados_df[
            resultados_df["clase"] == "A"
        ].copy()

        columnas = [
            "clase",
            "canal",
            "relacion",
            "n_pares",
            "r_mediana",
            "abs_r_mediana",
            "abs_r_p95",
            "pct_abs_r_ge_05",
            "pct_abs_r_ge_07",
            "pct_abs_r_ge_09",
        ]

        print(
            ejemplo[
                columnas
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 75)
    print("AUDITORÍA FINALIZADA")
    print("=" * 75)


# ================================================================
# EJECUCIÓN
# ================================================================

if __name__ == "__main__":
    main()
