
# ================================================================
# AUDITORÍA DE CORRELACIÓN DE PEARSON ENTRE TRIALS
# ================================================================
#
# Objetivo:
#   Responder:
#
#   "¿Las señales de EEG de los diferentes archivos se parecen
#    entre sí?"
#
#   Para cada uno de los 14 canales se compara la forma temporal
#   de cada trial contra los demás trials mediante Pearson.
#
# IMPORTANTE:
#   Esta auditoría es DIFERENTE de la correlación entre los
#   14 canales dentro de un mismo trial.
#
#   Aquí:
#
#       F3_trial_1  <->  F3_trial_2
#       F3_trial_1  <->  F3_trial_3
#       ...
#
#   y lo mismo para cada canal.
#
# Metodología:
#   - Usa únicamente archivos incluidos en reporte.csv.
#   - Usa la señal centrada por mediana.
#   - Para la matriz principal utiliza trials de 256 muestras.
#   - Los trials de 241-255 muestras se contabilizan aparte.
#   - No modifica los archivos originales.
#   - No establece umbrales de calidad.
#
# ================================================================

from pathlib import Path
import hashlib
import itertools
import math
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
    r"\correlaciones_entre_trials_resumen.csv"
)

SALIDA_MATRIZ = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_entre_trials_matrices.npz"
)

SALIDA_CENSO = Path(
    r"C:\tmp_bci\resultados_calidad"
    r"\correlaciones_entre_trials_censo.csv"
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
# CONFIGURACIÓN DE LONGITUD
# ================================================================

N_MUESTRAS_NOMINAL = 256

# Para esta primera auditoría NO se rellenan ni interpolan
# automáticamente los trials cortos.
#
# Se excluyen de la matriz principal porque Pearson entre dos
# vectores requiere que tengan la misma longitud.
#
# Se registran en el censo para mantener trazabilidad.


# ================================================================
# UTILIDADES
# ================================================================

def leer_csv_eeg(path):
    """
    Lee un CSV de EEG y devuelve:
        - dataframe
        - matriz numpy con los 14 canales
    """

    df = pd.read_csv(path)

    faltantes = [c for c in CANALES if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan canales: {faltantes}"
        )

    X = (
        df[CANALES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if np.isnan(X).any():
        raise ValueError(
            "Se encontraron NaN en los canales EEG."
        )

    return df, X


def centrar_por_mediana(X):
    """
    Centra cada canal restando su mediana temporal.

    X:
        shape = (muestras, 14)

    Retorna:
        señal centrada
    """

    medianas = np.median(X, axis=0)

    return X - medianas


def pearson_matrix(X):
    """
    Calcula la matriz de correlación Pearson entre canales
    temporales de un conjunto de trials.

    Entrada:
        X shape = (n_trials, n_muestras)

    Salida:
        matriz n_trials x n_trials
    """

    # np.corrcoef trabaja sobre filas cuando rowvar=True.
    #
    # Cada fila representa un trial.
    #
    # Resultado:
    #   matriz trial x trial

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        R = np.corrcoef(X)

    return R


def estadisticas_vector(values):
    """
    Calcula estadísticas descriptivas sobre un vector de
    correlaciones válidas.
    """

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "media": np.nan,
            "mediana": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "minimo": np.nan,
            "maximo": np.nan,
        }

    return {
        "n": len(values),
        "media": float(np.mean(values)),
        "mediana": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "minimo": float(np.min(values)),
        "maximo": float(np.max(values)),
    }


def estadisticas_correlacion(R):
    """
    Extrae la parte triangular superior de una matriz de
    correlación y calcula estadísticas de r y |r|.
    """

    n = R.shape[0]

    iu = np.triu_indices(n, k=1)

    r = R[iu]

    r_valid = r[np.isfinite(r)]

    abs_r = np.abs(r_valid)

    out = {}

    # ------------------------------------------------------------
    # r firmado
    # ------------------------------------------------------------

    s = estadisticas_vector(r_valid)

    out["n_pares_validos"] = s["n"]
    out["r_media"] = s["media"]
    out["r_mediana"] = s["mediana"]
    out["r_p05"] = s["p05"]
    out["r_p25"] = s["p25"]
    out["r_p75"] = s["p75"]
    out["r_p95"] = s["p95"]
    out["r_minimo"] = s["minimo"]
    out["r_maximo"] = s["maximo"]

    # ------------------------------------------------------------
    # |r|
    # ------------------------------------------------------------

    s_abs = estadisticas_vector(abs_r)

    out["abs_r_media"] = s_abs["media"]
    out["abs_r_mediana"] = s_abs["mediana"]
    out["abs_r_p05"] = s_abs["p05"]
    out["abs_r_p25"] = s_abs["p25"]
    out["abs_r_p75"] = s_abs["p75"]
    out["abs_r_p95"] = s_abs["p95"]
    out["abs_r_maximo"] = s_abs["maximo"]

    # ------------------------------------------------------------
    # Conteos descriptivos
    #
    # NO son reglas de calidad.
    # ------------------------------------------------------------

    if len(abs_r) > 0:

        for threshold in [0.3, 0.5, 0.7, 0.9]:

            out[
                f"n_abs_r_ge_{str(threshold).replace('.', '')}"
            ] = int(
                np.sum(abs_r >= threshold)
            )

            out[
                f"pct_abs_r_ge_{str(threshold).replace('.', '')}"
            ] = float(
                100.0 * np.mean(abs_r >= threshold)
            )

    else:

        for threshold in [0.3, 0.5, 0.7, 0.9]:

            out[
                f"n_abs_r_ge_{str(threshold).replace('.', '')}"
            ] = 0

            out[
                f"pct_abs_r_ge_{str(threshold).replace('.', '')}"
            ] = np.nan

    return out


def hash_archivo(path):
    """
    SHA256 opcional para trazabilidad.
    """

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ================================================================
# CARGAR UNIVERSO DE ARCHIVOS
# ================================================================

def cargar_universo():

    if not REPORTE.exists():

        raise FileNotFoundError(
            f"No existe:\n{REPORTE}"
        )

    reporte = pd.read_csv(REPORTE)

    if "incluido" not in reporte.columns:

        raise ValueError(
            "reporte.csv no contiene la columna 'incluido'."
        )

    incluidos = reporte[
        reporte["incluido"].astype(str).str.lower().isin(
            ["true", "1", "yes", "si"]
        )
    ].copy()

    return reporte, incluidos


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("AUDITORÍA DE CORRELACIÓN DE PEARSON ENTRE TRIALS")
    print("=" * 70)

    print()
    print(f"ROOT:    {ROOT}")
    print(f"REPORTE: {REPORTE}")
    print()
    print(f"Canales: {len(CANALES)}")
    print(f"Muestras nominales: {N_MUESTRAS_NOMINAL}")
    print()

    # ------------------------------------------------------------
    # Cargar reporte
    # ------------------------------------------------------------

    reporte, incluidos = cargar_universo()

    print(
        f"Archivos en reporte: {len(reporte)}"
    )

    print(
        f"Archivos incluidos: {len(incluidos)}"
    )

    print()

    # ------------------------------------------------------------
    # Preparar listas
    # ------------------------------------------------------------

    trials = []

    errores = []

    contador = 0

    # ------------------------------------------------------------
    # Leer todos los archivos
    # ------------------------------------------------------------

    for _, row in incluidos.iterrows():

        contador += 1

        ruta_relativa = str(
            row["ruta_relativa"]
        )

        path = ROOT / Path(ruta_relativa)

        if contador == 1 or contador % 250 == 0:

            print(
                f"[{contador:5d}/{len(incluidos)}] "
                f"{ruta_relativa}"
            )

        if not path.exists():

            errores.append({
                "ruta_relativa": ruta_relativa,
                "motivo": "archivo_no_encontrado",
            })

            continue

        try:

            df, X = leer_csv_eeg(path)

            n = X.shape[0]

            # ----------------------------------------------------
            # Centrado por mediana
            # ----------------------------------------------------

            X_cent = centrar_por_mediana(X)

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            usuario = row.get(
                "usuario",
                ""
            )

            subcarpeta = row.get(
                "subcarpeta",
                ""
            )

            trial_nombre = row.get(
                "trial_nombre",
                path.stem
            )

            session_id = row.get(
                "session_id",
                None
            )

            # ----------------------------------------------------
            # Guardar trial
            # ----------------------------------------------------

            trials.append({
                "ruta_relativa": ruta_relativa,
                "path": path,
                "usuario": usuario,
                "subcarpeta": subcarpeta,
                "trial_nombre": trial_nombre,
                "session_id": session_id,
                "n_muestras": n,
                "X": X_cent,
            })

        except Exception as e:

            errores.append({
                "ruta_relativa": ruta_relativa,
                "motivo": str(e),
            })

    # ------------------------------------------------------------
    # Resumen de carga
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("CARGA DE DATOS")
    print("-" * 70)

    print(
        f"Trials cargados: {len(trials)}"
    )

    print(
        f"Errores de lectura: {len(errores)}"
    )

    if errores:

        for e in errores[:20]:

            print(
                f"  ERROR: {e['ruta_relativa']} "
                f"-> {e['motivo']}"
            )

    # ------------------------------------------------------------
    # Censo de longitudes
    # ------------------------------------------------------------

    registros_censo = []

    for t in trials:

        registros_censo.append({

            "ruta_relativa":
                t["ruta_relativa"],

            "usuario":
                t["usuario"],

            "subcarpeta":
                t["subcarpeta"],

            "trial_nombre":
                t["trial_nombre"],

            "session_id":
                t["session_id"],

            "n_muestras":
                t["n_muestras"],

            "incluido_matriz_principal":
                t["n_muestras"] == N_MUESTRAS_NOMINAL,

        })

    censo_df = pd.DataFrame(
        registros_censo
    )

    SALIDA_CENSO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    censo_df.to_csv(
        SALIDA_CENSO,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # Seleccionar trials nominales
    # ------------------------------------------------------------

    trials_nominales = [
        t
        for t in trials
        if t["n_muestras"] == N_MUESTRAS_NOMINAL
    ]

    trials_cortos = [
        t
        for t in trials
        if t["n_muestras"] != N_MUESTRAS_NOMINAL
    ]

    print()
    print("-" * 70)
    print("LONGITUD")
    print("-" * 70)

    print(
        f"Trials nominales (256): "
        f"{len(trials_nominales)}"
    )

    print(
        f"Trials no nominales: "
        f"{len(trials_cortos)}"
    )

    if trials_cortos:

        print()
        print(
            "Los trials no nominales NO se interpolan ni "
            "rellenan automáticamente."
        )

        print(
            "Se excluyen únicamente de la matriz principal."
        )

    # ------------------------------------------------------------
    # Si no hay trials suficientes
    # ------------------------------------------------------------

    if len(trials_nominales) < 2:

        raise RuntimeError(
            "No hay suficientes trials de 256 muestras "
            "para calcular correlaciones."
        )

    # ------------------------------------------------------------
    # Construir tensor:
    #
    # trials × muestras × canales
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("CONSTRUCCIÓN DE MATRICES")
    print("-" * 70)

    n_trials = len(trials_nominales)

    n_muestras = N_MUESTRAS_NOMINAL

    n_canales = len(CANALES)

    X_all = np.empty(
        (
            n_trials,
            n_muestras,
            n_canales
        ),
        dtype=np.float32
    )

    for i, t in enumerate(trials_nominales):

        X_all[i] = t["X"].astype(
            np.float32
        )

    print(
        f"Tensor: "
        f"{X_all.shape[0]} trials × "
        f"{X_all.shape[1]} muestras × "
        f"{X_all.shape[2]} canales"
    )

    # ------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------

    matrices = {}

    resumen = []

    # ------------------------------------------------------------
    # Correlación por canal
    # ------------------------------------------------------------

    for canal_idx, canal in enumerate(CANALES):

        print()
        print(
            f"[{canal_idx + 1:2d}/{n_canales}] "
            f"Canal {canal}"
        )

        # --------------------------------------------------------
        # Extraer canal
        #
        # Resultado:
        #
        # trials × muestras
        # --------------------------------------------------------

        X_canal = X_all[
            :,
            :,
            canal_idx
        ]

        # --------------------------------------------------------
        # Verificar varianza
        # --------------------------------------------------------

        std_trials = np.std(
            X_canal,
            axis=1
        )

        n_varianza_cero = int(
            np.sum(std_trials == 0)
        )

        n_varianza_no_finita = int(
            np.sum(~np.isfinite(std_trials))
        )

        # --------------------------------------------------------
        # Pearson trial × trial
        # --------------------------------------------------------

        R = pearson_matrix(
            X_canal
        )

        matrices[canal] = R.astype(
            np.float32
        )

        # --------------------------------------------------------
        # Estadísticas globales
        # --------------------------------------------------------

        stats = estadisticas_correlacion(
            R
        )

        registro = {

            "nivel":
                "global",

            "canal":
                canal,

            "n_trials":
                n_trials,

            "n_muestras":
                N_MUESTRAS_NOMINAL,

            "n_trials_varianza_cero":
                n_varianza_cero,

            "n_trials_std_no_finita":
                n_varianza_no_finita,
        }

        registro.update(
            stats
        )

        resumen.append(
            registro
        )

        # --------------------------------------------------------
        # Estadísticas por subcarpeta
        # --------------------------------------------------------
        #
        # IMPORTANTE:
        #
        # Para evitar una interpretación incorrecta, aquí NO
        # calculamos correlación entre un trial de Letters y
        # otro de Numbers.
        #
        # Calculamos matrices independientes dentro de cada
        # condición.
        # --------------------------------------------------------

        grupos = {}

        for i, t in enumerate(
            trials_nominales
        ):

            grupo = str(
                t["subcarpeta"]
            )

            if grupo not in grupos:

                grupos[grupo] = []

            grupos[grupo].append(i)

        for grupo, indices in grupos.items():

            if len(indices) < 2:

                continue

            X_grupo = X_canal[
                indices,
                :
            ]

            R_grupo = pearson_matrix(
                X_grupo
            )

            stats_grupo = estadisticas_correlacion(
                R_grupo
            )

            registro = {

                "nivel":
                    "subcarpeta",

                "grupo":
                    grupo,

                "canal":
                    canal,

                "n_trials":
                    len(indices),

                "n_muestras":
                    N_MUESTRAS_NOMINAL,

                "n_trials_varianza_cero":
                    int(
                        np.sum(
                            np.std(
                                X_grupo,
                                axis=1
                            ) == 0
                        )
                    ),

                "n_trials_std_no_finita":
                    int(
                        np.sum(
                            ~np.isfinite(
                                np.std(
                                    X_grupo,
                                    axis=1
                                )
                            )
                        )
                    ),
            }

            registro.update(
                stats_grupo
            )

            resumen.append(
                registro
            )

        # --------------------------------------------------------
        # Estadísticas por usuario
        # --------------------------------------------------------

        usuarios = {}

        for i, t in enumerate(
            trials_nominales
        ):

            usuario = str(
                t["usuario"]
            )

            if usuario not in usuarios:

                usuarios[usuario] = []

            usuarios[usuario].append(i)

        for usuario, indices in usuarios.items():

            if len(indices) < 2:

                continue

            X_usuario = X_canal[
                indices,
                :
            ]

            R_usuario = pearson_matrix(
                X_usuario
            )

            stats_usuario = estadisticas_correlacion(
                R_usuario
            )

            registro = {

                "nivel":
                    "usuario",

                "grupo":
                    usuario,

                "canal":
                    canal,

                "n_trials":
                    len(indices),

                "n_muestras":
                    N_MUESTRAS_NOMINAL,

                "n_trials_varianza_cero":
                    int(
                        np.sum(
                            np.std(
                                X_usuario,
                                axis=1
                            ) == 0
                        )
                    ),

                "n_trials_std_no_finita":
                    int(
                        np.sum(
                            ~np.isfinite(
                                np.std(
                                    X_usuario,
                                    axis=1
                                )
                            )
                        )
                    ),
            }

            registro.update(
                stats_usuario
            )

            resumen.append(
                registro
            )

    # ============================================================
    # GUARDAR MATRICES
    # ============================================================

    print()
    print("-" * 70)
    print("GUARDANDO MATRICES")
    print("-" * 70)

    # ------------------------------------------------------------
    # Metadata para poder interpretar el NPZ
    # ------------------------------------------------------------

    metadata = {

        "canales":
            np.array(
                CANALES,
                dtype=object
            ),

        "n_trials":
            np.array(
                [n_trials],
                dtype=np.int64
            ),

        "n_muestras":
            np.array(
                [N_MUESTRAS_NOMINAL],
                dtype=np.int64
            ),

        "metodo":
            np.array(
                ["Pearson entre trials"],
                dtype=object
            ),

        "preprocesamiento":
            np.array(
                [
                    "centrado por mediana por canal"
                ],
                dtype=object
            ),
    }

    guardar = {}

    for canal in CANALES:

        guardar[
            canal
        ] = matrices[canal]

    guardar.update(
        metadata
    )

    np.savez_compressed(
        SALIDA_MATRIZ,
        **guardar
    )

    # ============================================================
    # GUARDAR RESUMEN
    # ============================================================

    resumen_df = pd.DataFrame(
        resumen
    )

    # Orden lógico
    columnas_preferidas = [

        "nivel",
        "grupo",
        "canal",

        "n_trials",
        "n_muestras",

        "n_trials_varianza_cero",
        "n_trials_std_no_finita",

        "n_pares_validos",

        "r_media",
        "r_mediana",
        "r_p05",
        "r_p25",
        "r_p75",
        "r_p95",
        "r_minimo",
        "r_maximo",

        "abs_r_media",
        "abs_r_mediana",
        "abs_r_p05",
        "abs_r_p25",
        "abs_r_p75",
        "abs_r_p95",
        "abs_r_maximo",

        "n_abs_r_ge_03",
        "pct_abs_r_ge_03",

        "n_abs_r_ge_05",
        "pct_abs_r_ge_05",

        "n_abs_r_ge_07",
        "pct_abs_r_ge_07",

        "n_abs_r_ge_09",
        "pct_abs_r_ge_09",
    ]

    columnas_finales = [
        c
        for c in columnas_preferidas
        if c in resumen_df.columns
    ]

    columnas_restantes = [
        c
        for c in resumen_df.columns
        if c not in columnas_finales
    ]

    resumen_df = resumen_df[
        columnas_finales +
        columnas_restantes
    ]

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
    # RESULTADO FINAL
    # ============================================================

    print()
    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(
        f"Trials cargados: "
        f"{len(trials)}"
    )

    print(
        f"Trials usados en matriz principal: "
        f"{len(trials_nominales)}"
    )

    print(
        f"Trials no nominales: "
        f"{len(trials_cortos)}"
    )

    print(
        f"Canales analizados: "
        f"{len(CANALES)}"
    )

    print()

    print(
        "Matriz por canal:"
    )

    print(
        f"  {n_trials} × {n_trials}"
    )

    print()

    print(
        "Archivo de resumen:"
    )

    print(
        f"  {SALIDA_RESUMEN}"
    )

    print()

    print(
        "Archivo de matrices:"
    )

    print(
        f"  {SALIDA_MATRIZ}"
    )

    print()

    print(
        "Censo de longitud:"
    )

    print(
        f"  {SALIDA_CENSO}"
    )

    # ============================================================
    # MOSTRAR RESUMEN GLOBAL
    # ============================================================

    print()
    print("-" * 70)
    print("DISTRIBUCIÓN GLOBAL POR CANAL")
    print("-" * 70)

    globales = resumen_df[
        resumen_df["nivel"] == "global"
    ].copy()

    columnas_mostrar = [
        "canal",
        "n_trials",
        "r_mediana",
        "r_p05",
        "r_p95",
        "abs_r_mediana",
        "abs_r_p95",
        "pct_abs_r_ge_05",
        "pct_abs_r_ge_07",
        "pct_abs_r_ge_09",
    ]

    print(
        globales[
            columnas_mostrar
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print("=" * 70)
    print("AUDITORÍA FINALIZADA")
    print("=" * 70)


# ================================================================
# EJECUCIÓN
# ================================================================

if __name__ == "__main__":
    main()