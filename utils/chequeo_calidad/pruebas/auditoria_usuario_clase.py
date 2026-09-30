# ================================================================
# AUDITORÍA DE CORRELACIÓN DE PEARSON ENTRE TRIALS
# CLASIFICADA POR USUARIO Y CLASE
# ================================================================
#
# Objetivo:
#
# Determinar de dónde proviene la similitud entre trials:
#
#   A) mismo usuario + misma clase
#   B) mismo usuario + clase diferente
#   C) usuario diferente + misma clase
#   D) usuario diferente + clase diferente
#
# Y repetir el análisis para cada uno de los 14 canales.
#
# ================================================================
#
# IMPORTANTE:
#
# La correlación se calcula entre la forma temporal completa
# de dos trials:
#
#     canal_trial_A <-> canal_trial_B
#
# usando Pearson sobre las 256 muestras.
#
# Cada trial es centrado por su mediana por canal antes
# de calcular las correlaciones.
#
# ================================================================
#
# SALIDAS:
#
# 1. resumen_correlaciones_por_relacion.csv
#
#    Estadísticas agregadas por:
#       relacion × canal
#
# 2. pares_correlacion_trials.csv
#
#    Una fila por cada par de trials y canal.
#
#    Permite investigar exactamente qué pares producen
#    correlaciones altas.
#
# 3. correlaciones_clasificadas.npz
#
#    Matrices completas por canal.
#
# ================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd


# ================================================================
# CONFIGURACIÓN
# ================================================================

ROOT = Path(
    r"C:\tmp_bci\captures"
)

REPORTE = Path(
    r"C:\tmp_bci\resultados_calidad\reporte.csv"
)

SALIDA_RESUMEN = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\resumen_correlaciones_por_relacion.csv"
)

SALIDA_PARES = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\pares_correlacion_trials.csv"
)

SALIDA_MATRIZ = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_clasificadas.npz"
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


# ================================================================
# CONFIGURACIÓN
# ================================================================

N_MUESTRAS = 256


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

    if not np.isfinite(X).all():

        raise ValueError(
            "Se encontraron valores NaN/inf."
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
# CLASIFICACIÓN DEL PAR
# ================================================================

def clasificar_par(trial_a, trial_b):

    mismo_usuario = (
        str(trial_a["usuario"])
        ==
        str(trial_b["usuario"])
    )

    misma_clase = (
        str(trial_a["clase"])
        ==
        str(trial_b["clase"])
    )

    if mismo_usuario and misma_clase:

        return "mismo_usuario_misma_clase"

    elif mismo_usuario and not misma_clase:

        return "mismo_usuario_clase_diferente"

    elif not mismo_usuario and misma_clase:

        return "usuario_diferente_misma_clase"

    else:

        return "usuario_diferente_clase_diferente"


# ================================================================
# PEARSON ENTRE DOS TRIALS
# ================================================================

def pearson_dos_vectores(x, y):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.sqrt(
            np.sum(x * x)
        )
        *
        np.sqrt(
            np.sum(y * y)
        )
    )

    if denom == 0:

        return np.nan

    return float(
        np.sum(x * y) / denom
    )


# ================================================================
# ESTADÍSTICAS
# ================================================================

def estadisticas(values):

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

            "r_minimo": np.nan,
            "r_maximo": np.nan,

            "abs_r_media": np.nan,
            "abs_r_mediana": np.nan,

            "abs_r_p05": np.nan,
            "abs_r_p25": np.nan,
            "abs_r_p75": np.nan,
            "abs_r_p95": np.nan,

            "abs_r_maximo": np.nan,

            "pct_abs_r_ge_03": np.nan,
            "pct_abs_r_ge_05": np.nan,
            "pct_abs_r_ge_07": np.nan,
            "pct_abs_r_ge_09": np.nan,
        }

    abs_r = np.abs(values)

    return {

        "n_pares":
            len(values),

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

        "r_minimo":
            float(np.min(values)),

        "r_maximo":
            float(np.max(values)),

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

        "abs_r_maximo":
            float(np.max(abs_r)),

        "pct_abs_r_ge_03":
            float(
                100 *
                np.mean(abs_r >= 0.3)
            ),

        "pct_abs_r_ge_05":
            float(
                100 *
                np.mean(abs_r >= 0.5)
            ),

        "pct_abs_r_ge_07":
            float(
                100 *
                np.mean(abs_r >= 0.7)
            ),

        "pct_abs_r_ge_09":
            float(
                100 *
                np.mean(abs_r >= 0.9)
            ),
    }


# ================================================================
# CARGAR UNIVERSO
# ================================================================

def cargar_universo():

    if not REPORTE.exists():

        raise FileNotFoundError(
            f"No existe:\n{REPORTE}"
        )

    reporte = pd.read_csv(
        REPORTE
    )

    if "incluido" not in reporte.columns:

        raise ValueError(
            "reporte.csv no contiene "
            "la columna 'incluido'."
        )

    incluidos = reporte[
        reporte["incluido"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "si"
            ]
        )
    ].copy()

    return reporte, incluidos


# ================================================================
# DETERMINAR COLUMNA DE CLASE
# ================================================================

def obtener_columna_clase(df):

    candidatos = [

        "clase",
        "class",
        "label",
        "target",
        "letra",
        "numero",
        "trial_clase",
    ]

    for c in candidatos:

        if c in df.columns:

            return c

    raise ValueError(
        "\nNo encontré una columna de clase.\n\n"
        "Busqué:\n"
        + "\n".join(
            f"  - {c}"
            for c in candidatos
        )
        + "\n\n"
        "Agrega el nombre real de la columna "
        "a la lista 'candidatos' de "
        "obtener_columna_clase()."
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 75)
    print(
        "AUDITORÍA PEARSON ENTRE TRIALS "
        "POR USUARIO Y CLASE"
    )
    print("=" * 75)

    print()

    reporte, incluidos = (
        cargar_universo()
    )

    print(
        f"Archivos en reporte: "
        f"{len(reporte)}"
    )

    print(
        f"Archivos incluidos: "
        f"{len(incluidos)}"
    )

    # ------------------------------------------------------------
    # Verificar columna de clase
    # ------------------------------------------------------------

    columna_clase = (
        obtener_columna_clase(
            incluidos
        )
    )

    print()

    print(
        f"Columna utilizada como clase: "
        f"{columna_clase}"
    )

    # ------------------------------------------------------------
    # Mostrar clases
    # ------------------------------------------------------------

    print()

    print(
        "Clases encontradas:"
    )

    clases = (
        incluidos[
            columna_clase
        ]
        .astype(str)
        .value_counts()
    )

    print(
        clases.to_string()
    )

    # ------------------------------------------------------------
    # Cargar trials
    # ------------------------------------------------------------

    trials = []

    errores = []

    print()

    print(
        "-" * 75
    )

    print(
        "CARGANDO TRIALS"
    )

    print(
        "-" * 75
    )

    for contador, (_, row) in enumerate(
        incluidos.iterrows(),
        start=1
    ):

        ruta_relativa = str(
            row["ruta_relativa"]
        )

        path = (
            ROOT /
            Path(ruta_relativa)
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
                "ruta_relativa":
                    ruta_relativa,

                "motivo":
                    "archivo_no_encontrado",
            })

            continue

        try:

            X = leer_csv_eeg(
                path
            )

            n = X.shape[0]

            if n != N_MUESTRAS:

                continue

            X = (
                centrar_por_mediana(
                    X
                )
            )

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            usuario = str(
                row.get(
                    "usuario",
                    ""
                )
            )

            clase = str(
                row[
                    columna_clase
                ]
            )

            subcarpeta = str(
                row.get(
                    "subcarpeta",
                    ""
                )
            )

            trial_nombre = str(
                row.get(
                    "trial_nombre",
                    path.stem
                )
            )

            session_id = row.get(
                "session_id",
                None
            )

            trials.append({

                "ruta_relativa":
                    ruta_relativa,

                "path":
                    str(path),

                "usuario":
                    usuario,

                "clase":
                    clase,

                "subcarpeta":
                    subcarpeta,

                "trial_nombre":
                    trial_nombre,

                "session_id":
                    session_id,

                "X":
                    X.astype(
                        np.float32
                    ),
            })

        except Exception as e:

            errores.append({

                "ruta_relativa":
                    ruta_relativa,

                "motivo":
                    str(e),
            })

    # ------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------

    n_trials = len(
        trials
    )

    print()

    print(
        "-" * 75
    )

    print(
        "RESUMEN DE CARGA"
    )

    print(
        "-" * 75
    )

    print(
        f"Trials nominales: "
        f"{n_trials}"
    )

    print(
        f"Errores: "
        f"{len(errores)}"
    )

    if n_trials < 2:

        raise RuntimeError(
            "No hay suficientes trials."
        )

    # ============================================================
    # MATRIZ DE DATOS
    # ============================================================

    print()

    print(
        "-" * 75
    )

    print(
        "CONSTRUYENDO MATRIZ"
    )

    print(
        "-" * 75
    )

    X_all = np.stack(
        [
            t["X"]
            for t in trials
        ],
        axis=0
    )

    print(
        "Tensor:"
    )

    print(
        f"  {X_all.shape[0]} trials × "
        f"{X_all.shape[1]} muestras × "
        f"{X_all.shape[2]} canales"
    )

    # ============================================================
    # PRECALCULAR MATRICES PEARSON
    # ============================================================
    #
    # Esto evita calcular Pearson individualmente para cada par.
    #
    # ============================================================

    matrices = {}

    for canal_idx, canal in enumerate(
        CANALES
    ):

        print(
            f"[{canal_idx + 1:2d}/"
            f"{len(CANALES)}] "
            f"Calculando {canal}"
        )

        X_canal = X_all[
            :,
            :,
            canal_idx
        ]

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                RuntimeWarning
            )

            R = np.corrcoef(
                X_canal
            )

        matrices[
            canal
        ] = R.astype(
            np.float32
        )

        # # ================================================================
    # CLASIFICACIÓN VECTORIAL DE TODOS LOS PARES
    # ================================================================
    #
    # NO construimos un DataFrame de 23 millones de filas.
    #
    # Creamos directamente cuatro máscaras booleanas:
    #
    #   A = mismo usuario + misma clase
    #   B = mismo usuario + clase diferente
    #   C = usuario diferente + misma clase
    #   D = usuario diferente + clase diferente
    #
    # Cada máscara tiene tamaño:
    #
    #   n_trials × n_trials
    #
    # ================================================================

    print()
    print("-" * 75)
    print("CLASIFICANDO PARES DE FORMA VECTORIAL")
    print("-" * 75)


    # ------------------------------------------------------------
    # Arrays de metadata
    # ------------------------------------------------------------

    usuarios = np.array(
        [
            str(t["usuario"])
            for t in trials
        ],
        dtype=object
    )

    clases = np.array(
        [
            str(t["clase"])
            for t in trials
        ],
        dtype=object
    )


    # ------------------------------------------------------------
    # Matrices usuario / clase
    # ------------------------------------------------------------
    #
    # usuario_igual[i,j] = True si el trial i y j
    # pertenecen al mismo usuario.
    #
    # clase_igual[i,j] = True si tienen la misma clase.
    #
    # ------------------------------------------------------------

    usuario_igual = (
        usuarios[:, None]
        ==
        usuarios[None, :]
    )

    clase_igual = (
        clases[:, None]
        ==
        clases[None, :]
    )


    # ------------------------------------------------------------
    # Solamente nos interesan pares distintos.
    #
    # La diagonal i == j NO es un par entre trials.
    # ------------------------------------------------------------

    no_diagonal = ~np.eye(
        n_trials,
        dtype=bool
    )


    # ------------------------------------------------------------
    # Cuatro categorías
    # ------------------------------------------------------------

    mascaras = {

        "mismo_usuario_misma_clase":
            usuario_igual
            & clase_igual
            & no_diagonal,

        "mismo_usuario_clase_diferente":
            usuario_igual
            & ~clase_igual
            & no_diagonal,

        "usuario_diferente_misma_clase":
            ~usuario_igual
            & clase_igual
            & no_diagonal,

        "usuario_diferente_clase_diferente":
            ~usuario_igual
            & ~clase_igual
            & no_diagonal,
    }


    # ------------------------------------------------------------
    # Para evitar contar A-B y B-A dos veces,
    # utilizamos únicamente el triángulo superior.
    # ------------------------------------------------------------

    triangulo_superior = np.triu(
        np.ones(
            (n_trials, n_trials),
            dtype=bool
        ),
        k=1
    )


    for relacion in mascaras:

        mascaras[relacion] &= (
            triangulo_superior
        )


    # ------------------------------------------------------------
    # Verificación
    # ------------------------------------------------------------

    print()

    print(
        "Cantidad de pares por relación:"
    )

    for relacion, mascara in mascaras.items():

        cantidad = int(
            np.sum(mascara)
        )

        print(
            f"  {relacion:45s}"
            f"{cantidad:,}"
        )


    total_pares = sum(
        int(np.sum(m))
        for m in mascaras.values()
    )

    pares_teoricos = (
        n_trials * (n_trials - 1)
        // 2
    )

    print()

    print(
        f"Total pares clasificados: "
        f"{total_pares:,}"
    )

    print(
        f"Total pares esperado:     "
        f"{pares_teoricos:,}"
    )

    if total_pares != pares_teoricos:

        raise RuntimeError(
            "ERROR: las cuatro categorías "
            "no cubren todos los pares."
        )

    print(
        "OK: todos los pares están "
        "clasificados exactamente una vez."
    )


    # ================================================================
    # ESTADÍSTICAS POR RELACIÓN Y CANAL
    # ================================================================

    print()
    print("-" * 75)
    print("CALCULANDO ESTADÍSTICAS")
    print("-" * 75)


    resumen = []


    for canal in CANALES:

        print(
            f"Canal: {canal}"
        )

        R = matrices[
            canal
        ]

        for relacion, mascara in (
            mascaras.items()
        ):

            # --------------------------------------------------------
            # Extraer únicamente los valores de Pearson
            # pertenecientes a esta categoría.
            # --------------------------------------------------------

            r_values = R[
                mascara
            ]

            stats = estadisticas(
                r_values
            )

            registro = {

                "relacion":
                    relacion,

                "canal":
                    canal,

                "n_trials":
                    n_trials,

                "n_pares_teoricos":
                    int(
                        np.sum(mascara)
                    ),
            }

            registro.update(
                stats
            )

            resumen.append(
                registro
            )


    resumen_df = pd.DataFrame(
        resumen
    )

    # ============================================================
    # GUARDAR RESUMEN
    # ============================================================

    SALIDA_RESUMEN.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resumen_df.to_csv(
        SALIDA_RESUMEN,
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # GUARDAR NPZ
    # ============================================================

    guardar = {}

    for canal in CANALES:

        guardar[
            canal
        ] = matrices[
            canal
        ]

    guardar[
        "canales"
    ] = np.array(
        CANALES,
        dtype=object
    )

    guardar[
        "n_trials"
    ] = np.array(
        [n_trials],
        dtype=np.int64
    )

    guardar[
        "n_muestras"
    ] = np.array(
        [N_MUESTRAS],
        dtype=np.int64
    )

    np.savez_compressed(
        SALIDA_MATRIZ,
        **guardar
    )

    # ============================================================
    # MOSTRAR TABLA PRINCIPAL
    # ============================================================

    print()

    print(
        "=" * 75
    )

    print(
        "RESULTADO POR RELACIÓN Y CANAL"
    )

    print(
        "=" * 75
    )

    columnas_mostrar = [

        "relacion",
        "canal",

        "n_pares",

        "r_mediana",

        "r_p05",
        "r_p95",

        "abs_r_mediana",

        "abs_r_p95",

        "pct_abs_r_ge_03",
        "pct_abs_r_ge_05",
        "pct_abs_r_ge_07",
        "pct_abs_r_ge_09",
    ]

    print(
        resumen_df[
            columnas_mostrar
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ============================================================
    # ARCHIVOS
    # ============================================================

    print()

    print(
        "=" * 75
    )

    print(
        "ARCHIVOS GENERADOS"
    )

    print(
        "=" * 75
    )

    print()

    print(
        "Resumen:"
    )

    print(
        f"  {SALIDA_RESUMEN}"
    )

    print()

    print(
        "Pares individuales:"
    )

    print(
        f"  {SALIDA_PARES}"
    )

    print()

    print(
        "Matrices:"
    )

    print(
        f"  {SALIDA_MATRIZ}"
    )

    print()

    print(
        "=" * 75
    )

    print(
        "AUDITORÍA FINALIZADA"
    )

    print(
        "=" * 75
    )


# ================================================================
# EJECUCIÓN
# ================================================================

if __name__ == "__main__":

    main()
