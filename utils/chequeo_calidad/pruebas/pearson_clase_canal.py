# ================================================================
# PEARSON POR CLASE × CANAL
# ENTRE USUARIOS DIFERENTES
#
# Objetivo:
#   Determinar en qué canales/regiones EEG existe mayor similitud
#   entre usuarios cuando realizan EXACTAMENTE LA MISMA CLASE.
#
# Ejemplo:
#
#   A de UserAxel    <-> A de UserJorge
#   A de UserAxel    <-> A de UserMar
#   A de UserJorge   <-> A de UserMar
#   ...
#
# NO compara:
#   A <-> B
#   A <-> C
#
# NO compara:
#   A de UserAxel <-> A de UserAxel
#
# Exclusiones:
#   - cualquier archivo dentro de "ignorarSenales"
#   - trials que no tengan exactamente 256 muestras
#   - trials sin alguno de los 14 canales requeridos
#
# ================================================================

from pathlib import Path
from itertools import combinations
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# CONFIGURACIÓN
# ================================================================

ROOT = Path(r"C:\tmp_bci\captures")

OUTPUT_DIR = Path(
    r"C:\tmp_bci\resultados_calidad"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Canales
# ------------------------------------------------

CHANNELS = [
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


# ------------------------------------------------
# Tamaño esperado
# ------------------------------------------------

N_SAMPLES = 256


# ------------------------------------------------
# Clases esperadas
# ------------------------------------------------

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUMBERS = list("0123456789")

COMMANDS = [
    "↩",
    "───",
    "⟵",
]

EXPECTED_CLASSES = LETTERS + NUMBERS + COMMANDS


# ================================================================
# FUNCIONES AUXILIARES
# ================================================================

def is_ignored(path: Path):
    """
    Devuelve True si el archivo está dentro de una carpeta
    llamada 'ignorarSenales' o cualquiera de sus subcarpetas.
    """

    for part in path.parts:
        if part.lower() == "ignorarsenales":
            return True

    return False


# ------------------------------------------------

def extract_metadata(path: Path):
    """
    Extrae:

        usuario
        clase
        trial

    desde nombres como:

        UserAxel_A_1.csv
        UserAxel_7_15.csv
        UserMar_↩_3.csv
        UserMar_───_26.csv

    La carpeta también ayuda a identificar el tipo de clase.
    """

    filename = path.stem

    # ------------------------------------------------------------
    # Primero intentamos usar el nombre del archivo.
    #
    # Patrón:
    #   UserAxel_A_1
    #   UserAxel_7_15
    # ------------------------------------------------------------

    match = re.match(
        r"^(?P<user>.+)_(?P<class>.+)_(?P<trial>\d+)$",
        filename
    )

    if not match:
        return None

    user = match.group("user")
    class_name = match.group("class")
    trial = int(match.group("trial"))

    return {
        "usuario": user,
        "clase": class_name,
        "trial": trial,
        "path": path,
    }


# ------------------------------------------------

def pearson_fast(x, y):
    """
    Pearson r calculado directamente con NumPy.

    x e y deben ser vectores de longitud 256.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = np.sqrt(
        np.sum(x * x) *
        np.sum(y * y)
    )

    if denominator == 0:
        return np.nan

    return np.sum(x * y) / denominator


# ------------------------------------------------

def load_trial(path):
    """
    Carga un CSV y devuelve:

        array [256, 14]

    o None si el trial no es válido.
    """

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR leyendo {path}: {e}")
        return None

    # ------------------------------------------------------------
    # Normalizamos nombres de columnas
    # ------------------------------------------------------------

    normalized = {
        str(c).strip().upper(): c
        for c in df.columns
    }

    # ------------------------------------------------------------
    # Verificar canales
    # ------------------------------------------------------------

    missing = [
        ch for ch in CHANNELS
        if ch.upper() not in normalized
    ]

    if missing:
        print(
            f"WARNING: faltan canales {missing}: {path}"
        )
        return None

    # ------------------------------------------------------------
    # Extraer canales
    # ------------------------------------------------------------

    data = []

    for ch in CHANNELS:

        original_column = normalized[ch.upper()]

        values = pd.to_numeric(
            df[original_column],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        data.append(values)

    # ------------------------------------------------------------
    # Convertimos:
    #
    #     [14, N]
    #
    # a:
    #
    #     [N, 14]
    # ------------------------------------------------------------

    lengths = [len(x) for x in data]

    if len(set(lengths)) != 1:
        print(
            f"WARNING: canales con longitudes diferentes: {path}"
        )
        return None

    n = lengths[0]

    if n != N_SAMPLES:
        print(
            f"WARNING: {path} tiene {n} muestras. "
            f"Se esperaban {N_SAMPLES}. Se ignora."
        )
        return None

    matrix = np.column_stack(data)

    # ------------------------------------------------------------
    # Verificar NaN
    # ------------------------------------------------------------

    if not np.isfinite(matrix).all():
        print(
            f"WARNING: NaN/Inf encontrado en {path}. Se ignora."
        )
        return None

    return matrix


# ================================================================
# CARGAR TODOS LOS TRIALS
# ================================================================

def load_all_trials():

    print("=" * 70)
    print("CARGANDO TRIALS")
    print("=" * 70)

    files = sorted(ROOT.rglob("*.csv"))

    print(f"Archivos CSV encontrados: {len(files)}")

    trials = []

    ignored_count = 0
    invalid_count = 0

    for i, path in enumerate(files, start=1):

        if i % 250 == 0 or i == 1:
            print(
                f"[{i:6d}/{len(files)}] {path}"
            )

        # --------------------------------------------------------
        # IGNORAR ignoraSenales
        # --------------------------------------------------------

        if is_ignored(path):
            ignored_count += 1
            continue

        metadata = extract_metadata(path)

        if metadata is None:
            invalid_count += 1
            continue

        matrix = load_trial(path)

        if matrix is None:
            invalid_count += 1
            continue

        metadata["data"] = matrix

        trials.append(metadata)

    print()
    print("=" * 70)
    print("RESUMEN DE CARGA")
    print("=" * 70)

    print(f"CSV encontrados:       {len(files):,}")
    print(f"Trials válidos:        {len(trials):,}")
    print(f"Ignorados:             {ignored_count:,}")
    print(f"Inválidos/excluidos:   {invalid_count:,}")

    return trials


# ================================================================
# ORGANIZAR POR CLASE
# ================================================================

def group_by_class(trials):

    groups = {}

    for t in trials:

        cls = t["clase"]

        if cls not in groups:
            groups[cls] = []

        groups[cls].append(t)

    return groups


# ================================================================
# CALCULAR ESTADÍSTICAS
# ================================================================

def calculate_statistics(values):

    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n_pares": 0,
            "r_media": np.nan,
            "r_mediana": np.nan,
            "r_std": np.nan,
            "r_p01": np.nan,
            "r_p05": np.nan,
            "r_p25": np.nan,
            "r_p75": np.nan,
            "r_p95": np.nan,
            "r_p99": np.nan,
            "abs_r_media": np.nan,
            "abs_r_mediana": np.nan,
            "abs_r_p95": np.nan,
            "pct_abs_r_ge_03": np.nan,
            "pct_abs_r_ge_05": np.nan,
            "pct_abs_r_ge_07": np.nan,
            "pct_abs_r_ge_09": np.nan,
        }

    abs_values = np.abs(values)

    return {
        "n_pares": len(values),

        "r_media":
            np.mean(values),

        "r_mediana":
            np.median(values),

        "r_std":
            np.std(values),

        "r_p01":
            np.percentile(values, 1),

        "r_p05":
            np.percentile(values, 5),

        "r_p25":
            np.percentile(values, 25),

        "r_p75":
            np.percentile(values, 75),

        "r_p95":
            np.percentile(values, 95),

        "r_p99":
            np.percentile(values, 99),

        "abs_r_media":
            np.mean(abs_values),

        "abs_r_mediana":
            np.median(abs_values),

        "abs_r_p95":
            np.percentile(abs_values, 95),

        "pct_abs_r_ge_03":
            100 * np.mean(abs_values >= 0.3),

        "pct_abs_r_ge_05":
            100 * np.mean(abs_values >= 0.5),

        "pct_abs_r_ge_07":
            100 * np.mean(abs_values >= 0.7),

        "pct_abs_r_ge_09":
            100 * np.mean(abs_values >= 0.9),
    }


# ================================================================
# ANÁLISIS PRINCIPAL
# ================================================================

def calculate_class_channel_statistics(trials):

    print()
    print("=" * 70)
    print("CALCULANDO PEARSON: CLASE × CANAL × USUARIOS DIFERENTES")
    print("=" * 70)

    groups = group_by_class(trials)

    rows = []

    # ------------------------------------------------------------
    # Procesamos cada clase
    # ------------------------------------------------------------

    for class_name in EXPECTED_CLASSES:

        class_trials = groups.get(class_name, [])

        if not class_trials:
            print(
                f"Clase {class_name}: sin trials válidos"
            )
            continue

        # --------------------------------------------------------
        # Solo necesitamos pares de USUARIOS DIFERENTES
        # --------------------------------------------------------

        valid_pairs = []

        for a, b in combinations(class_trials, 2):

            if a["usuario"] == b["usuario"]:
                continue

            valid_pairs.append((a, b))

        print(
            f"Clase {class_name:>3}: "
            f"{len(class_trials):5d} trials, "
            f"{len(valid_pairs):7d} pares entre usuarios"
        )

        # --------------------------------------------------------
        # Para cada canal
        # --------------------------------------------------------

        for channel_index, channel in enumerate(CHANNELS):

            correlations = []

            for a, b in valid_pairs:

                x = a["data"][:, channel_index]
                y = b["data"][:, channel_index]

                r = pearson_fast(x, y)

                if np.isfinite(r):
                    correlations.append(r)

            stats = calculate_statistics(correlations)

            row = {
                "clase": class_name,
                "canal": channel,
                **stats,
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# GUARDAR RESULTADOS
# ================================================================

def save_results(df):

    # ------------------------------------------------------------
    # 1. Resultado completo
    # ------------------------------------------------------------

    output_main = (
        OUTPUT_DIR /
        "pearson_clase_canal_entre_usuarios.csv"
    )

    df.to_csv(
        output_main,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 2. Ordenar por similitud absoluta
    # ------------------------------------------------------------

    df_sorted = df.sort_values(
        [
            "clase",
            "abs_r_mediana"
        ],
        ascending=[
            True,
            False
        ]
    )

    output_sorted = (
        OUTPUT_DIR /
        "pearson_clase_canal_mejores.csv"
    )

    df_sorted.to_csv(
        output_sorted,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3. Matriz:
    #
    #       clase
    # canal A B C ...
    #
    # valor = abs_r_mediana
    # ------------------------------------------------------------

    matrix = df.pivot(
        index="canal",
        columns="clase",
        values="abs_r_mediana"
    )

    output_matrix = (
        OUTPUT_DIR /
        "pearson_matriz_clase_canal.csv"
    )

    matrix.to_csv(
        output_matrix,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 4. TOP canal por clase
    # ------------------------------------------------------------

    top_rows = []

    for cls, group in df.groupby("clase"):

        group = group.dropna(
            subset=["abs_r_mediana"]
        )

        if group.empty:
            continue

        # Canal con mayor mediana de |r|
        best = group.loc[
            group["abs_r_mediana"].idxmax()
        ]

        top_rows.append(best)

    top_df = pd.DataFrame(top_rows)

    output_top = (
        OUTPUT_DIR /
        "pearson_top_por_clase.csv"
    )

    top_df.to_csv(
        output_top,
        index=False,
        encoding="utf-8-sig"
    )

    return (
        output_main,
        output_sorted,
        output_matrix,
        output_top,
    )


# ================================================================
# HEATMAP
# ================================================================

def create_heatmap(df):

    matrix = df.pivot(
        index="canal",
        columns="clase",
        values="abs_r_mediana"
    )

    # ------------------------------------------------------------
    # Ordenar clases:
    #
    # A-Z
    # 0-9
    # comandos
    # ------------------------------------------------------------

    ordered_classes = [
        c for c in EXPECTED_CLASSES
        if c in matrix.columns
    ]

    matrix = matrix[
        ordered_classes
    ]

    # ------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------

    plt.figure(
        figsize=(22, 9)
    )

    sns.heatmap(
        matrix,
        cmap="viridis",
        annot=False,
        vmin=0,
        vmax=np.nanpercentile(
            matrix.values,
            98
        ),
        cbar_kws={
            "label": "Mediana de |Pearson r|"
        }
    )

    plt.title(
        "Similitud interusuario por clase y canal\n"
        "Mediana de |Pearson r|"
    )

    plt.xlabel("Clase")
    plt.ylabel("Canal")

    plt.tight_layout()

    output = (
        OUTPUT_DIR /
        "pearson_heatmap_clase_canal.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output


# ================================================================
# RESUMEN EN CONSOLA
# ================================================================

def print_summary(df):

    print()
    print("=" * 70)
    print("TOP CANALES POR CLASE")
    print("=" * 70)

    for cls in EXPECTED_CLASSES:

        group = df[
            df["clase"] == cls
        ].copy()

        if group.empty:
            continue

        group = group.dropna(
            subset=["abs_r_mediana"]
        )

        if group.empty:
            continue

        group = group.sort_values(
            "abs_r_mediana",
            ascending=False
        )

        print()
        print(f"CLASE: {cls}")

        print(
            group[
                [
                    "canal",
                    "n_pares",
                    "r_mediana",
                    "abs_r_mediana",
                    "abs_r_p95",
                    "pct_abs_r_ge_05",
                    "pct_abs_r_ge_07",
                    "pct_abs_r_ge_09",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )


# ================================================================
# MAIN
# ================================================================

def main():


    print("AUDITORÍA PEARSON POR CLASE × CANAL")
    print("USUARIOS DIFERENTES")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Cargar
    # ------------------------------------------------------------

    trials = load_all_trials()

    if not trials:
        print(
            "ERROR: no se encontraron trials válidos."
        )
        return

    # ------------------------------------------------------------
    # 2. Calcular
    # ------------------------------------------------------------

    df = calculate_class_channel_statistics(
        trials
    )

    # ------------------------------------------------------------
    # 3. Guardar
    # ------------------------------------------------------------

    outputs = save_results(df)

    # ------------------------------------------------------------
    # 4. Heatmap
    # ------------------------------------------------------------

    heatmap = create_heatmap(df)

    # ------------------------------------------------------------
    # 5. Mostrar resumen
    # ------------------------------------------------------------

    print_summary(df)

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print()
    print("ARCHIVOS GENERADOS")

    for path in outputs:
        print(f"  {path}")

    print(f"  {heatmap}")

    print("AUDITORÍA FINALIZADA")



# ================================================================
# EJECUTAR
# ================================================================

if __name__ == "__main__":
    main()
