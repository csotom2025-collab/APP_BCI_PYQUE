import os
import re
import glob
import itertools
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = r"C:\tmp_bci\captures"

OUTPUT_DIR = r"C:\tmp_bci\resultados_calidad"

# Canales EEG a utilizar
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

N_SAMPLES = 256

# Ignorar completamente cualquier ruta que contenga esta carpeta
IGNORED_FOLDER = "ignorarSenales"

# Solo queremos comparar:
# MISMA CLASE + USUARIOS DIFERENTES
RELATION = "usuario_diferente_misma_clase"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def is_ignored(path):
    """
    Devuelve True si el archivo pertenece a una carpeta
    llamada ignorarSenales.
    """
    parts = os.path.normpath(path).split(os.sep)

    return any(
        part.lower() == IGNORED_FOLDER.lower()
        for part in parts
    )


def detectar_clase_desde_nombre(filename):
    """
    Intenta obtener la clase desde nombres como:

        UserPony_A_0.csv
        UserPony_7_15.csv
        UserPony_ENTER_3.csv
        UserPony_ESPACIO_2.csv

    Para letras/números toma el penúltimo elemento.

    Ejemplo:
        UserPony_A_0.csv -> A
        UserPony_7_15.csv -> 7
    """

    base = os.path.splitext(os.path.basename(filename))[0]

    partes = base.split("_")

    if len(partes) < 2:
        return None

    # El elemento anterior al índice del trial
    clase = partes[-2]

    return clase


def detectar_usuario_desde_nombre(filename):
    """
    Ejemplo:

        UserPony_A_0.csv -> UserPony
        UserAkatzin_A_15.csv -> UserAkatzin
    """

    base = os.path.splitext(os.path.basename(filename))[0]

    partes = base.split("_")

    if len(partes) < 3:
        return None

    # Todo excepto los dos últimos componentes
    usuario = "_".join(partes[:-2])

    return usuario


def normalizar_clase(clase):
    """
    Normaliza nombres de clases para facilitar el análisis.
    """

    if clase is None:
        return None

    clase = str(clase).strip()

    # Letras
    if len(clase) == 1 and clase.isalpha():
        return clase.upper()

    # Números
    if clase.isdigit():
        return clase

    # Comandos: conservar texto pero normalizar
    reemplazos = {
        "borrar": "BORRAR",
        "Borrar": "BORRAR",
        "BORRAR": "BORRAR",

        "enter": "ENTER",
        "Enter": "ENTER",
        "ENTER": "ENTER",

        "espacio": "ESPACIO",
        "Espacio": "ESPACIO",
        "ESPACIO": "ESPACIO",

        "space": "ESPACIO",
        "SPACE": "ESPACIO",

        "backspace": "BORRAR",
        "BACKSPACE": "BORRAR",
    }

    return reemplazos.get(clase, clase)


def cargar_trial(path):
    """
    Carga un CSV y devuelve una matriz:

        muestras × canales

    Devuelve None si:
      - faltan canales
      - no tiene 256 muestras
      - el CSV está vacío
      - ocurre algún error
    """

    try:
        df = pd.read_csv(path)

    except Exception as e:
        print(f"ERROR leyendo {path}: {e}")
        return None

    # --------------------------------------------------------
    # Comprobar canales
    # --------------------------------------------------------

    columnas = {str(c).strip(): c for c in df.columns}

    faltantes = [
        ch for ch in CHANNELS
        if ch not in columnas
    ]

    if faltantes:
        print(
            f"WARNING: {path} no tiene los canales: "
            f"{faltantes}"
        )
        return None

    # --------------------------------------------------------
    # Extraer canales
    # --------------------------------------------------------

    try:
        data = df[
            [columnas[ch] for ch in CHANNELS]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )

    except Exception as e:
        print(f"ERROR procesando {path}: {e}")
        return None

    # --------------------------------------------------------
    # Comprobar cantidad de muestras
    # --------------------------------------------------------

    if data.shape[0] != N_SAMPLES:

        print(
            f"WARNING: {path} tiene "
            f"{data.shape[0]} muestras. "
            f"Se esperaban {N_SAMPLES}."
        )

        return None

    # --------------------------------------------------------
    # Comprobar NaN
    # --------------------------------------------------------

    if not np.isfinite(data).all():

        print(
            f"WARNING: {path} contiene NaN/Inf."
        )

        return None

    return data


def pearson(x, y):
    """
    Correlación de Pearson entre dos señales.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        return np.nan

    # Evitar división por cero
    sx = np.std(x)
    sy = np.std(y)

    if sx == 0 or sy == 0:
        return np.nan

    r = np.corrcoef(x, y)[0, 1]

    return r


# ============================================================
# CARGAR ARCHIVOS
# ============================================================

def encontrar_archivos():

    patron = os.path.join(
        ROOT_DIR,
        "**",
        "*.csv"
    )

    archivos = glob.glob(
        patron,
        recursive=True
    )

    print(
        f"Archivos CSV encontrados originalmente: "
        f"{len(archivos):,}"
    )

    # --------------------------------------------------------
    # EXCLUSIÓN TOTAL DE ignorarSenales
    # --------------------------------------------------------

    archivos_validos = [
        p for p in archivos
        if not is_ignored(p)
    ]

    ignorados = len(archivos) - len(archivos_validos)

    print(
        f"Archivos excluidos por "
        f"'{IGNORED_FOLDER}': {ignorados:,}"
    )

    print(
        f"Archivos que serán analizados: "
        f"{len(archivos_validos):,}"
    )

    return sorted(archivos_validos)


# ============================================================
# CARGA DE TRIALS
# ============================================================

def cargar_trials():

    archivos = encontrar_archivos()

    trials = []

    print()
    print("=" * 70)
    print("CARGANDO TRIALS")
    print("=" * 70)

    for i, path in enumerate(archivos, start=1):

        if i == 1 or i % 250 == 0:
            print(
                f"[{i:6d}/{len(archivos)}] "
                f"{os.path.relpath(path, ROOT_DIR)}"
            )

        # Seguridad adicional
        if is_ignored(path):
            continue

        clase_raw = detectar_clase_desde_nombre(path)

        usuario = detectar_usuario_desde_nombre(path)

        clase = normalizar_clase(clase_raw)

        if clase is None or usuario is None:
            print(
                f"WARNING: no se pudo identificar "
                f"usuario/clase: {path}"
            )
            continue

        data = cargar_trial(path)

        if data is None:
            continue

        trials.append({
            "path": path,
            "usuario": usuario,
            "clase": clase,
            "data": data,
        })

    print()
    print("=" * 70)
    print("RESUMEN DE CARGA")
    print("=" * 70)

    print(f"Trials válidos: {len(trials):,}")

    return trials


# ============================================================
# AGRUPAR POR CLASE
# ============================================================

def agrupar_por_clase(trials):

    grupos = {}

    for trial in trials:

        clase = trial["clase"]

        if clase not in grupos:
            grupos[clase] = []

        grupos[clase].append(trial)

    return grupos


# ============================================================
# CORRELACIONES
# ============================================================

def calcular_correlaciones(trials):

    grupos = agrupar_por_clase(trials)

    resultados = []

    print()
    print("=" * 70)
    print("CALCULANDO PEARSON")
    print("MISMA CLASE + USUARIOS DIFERENTES")
    print("=" * 70)

    print()
    print(
        f"Clases encontradas: {len(grupos)}"
    )

    for clase in sorted(grupos.keys()):

        grupo = grupos[clase]

        usuarios = sorted(
            set(t["usuario"] for t in grupo)
        )

        print()
        print("-" * 70)
        print(
            f"CLASE: {clase} | "
            f"Trials: {len(grupo):,} | "
            f"Usuarios: {len(usuarios)}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Solo pares de usuarios diferentes
        # ----------------------------------------------------

        pares = []

        for i in range(len(grupo)):

            for j in range(i + 1, len(grupo)):

                t1 = grupo[i]
                t2 = grupo[j]

                # CRÍTICO:
                # solo usuarios diferentes
                if t1["usuario"] == t2["usuario"]:
                    continue

                pares.append((t1, t2))

        print(
            f"Pares entre usuarios diferentes: "
            f"{len(pares):,}"
        )

        # ----------------------------------------------------
        # Pearson por canal
        # ----------------------------------------------------

        for canal_idx, canal in enumerate(CHANNELS):

            print(
                f"  Canal: {canal}"
            )

            valores = []

            for t1, t2 in pares:

                x = t1["data"][:, canal_idx]
                y = t2["data"][:, canal_idx]

                r = pearson(x, y)

                if np.isfinite(r):
                    valores.append(r)

            valores = np.asarray(
                valores,
                dtype=np.float64
            )

            if len(valores) == 0:
                continue

            abs_values = np.abs(valores)

            resultados.append({

                "clase": clase,

                "canal": canal,

                "relacion": RELATION,

                "n_pares": len(valores),

                # Pearson firmado
                "r_media": np.mean(valores),
                "r_mediana": np.median(valores),

                # Percentiles
                "r_p05": np.percentile(valores, 5),
                "r_p25": np.percentile(valores, 25),
                "r_p75": np.percentile(valores, 75),
                "r_p95": np.percentile(valores, 95),

                # Magnitud absoluta
                "abs_r_media": np.mean(abs_values),
                "abs_r_mediana": np.median(abs_values),

                "abs_r_p95": np.percentile(
                    abs_values,
                    95
                ),

                # Porcentaje |r| >= umbrales
                "pct_abs_r_ge_03":
                    100 * np.mean(
                        abs_values >= 0.30
                    ),

                "pct_abs_r_ge_05":
                    100 * np.mean(
                        abs_values >= 0.50
                    ),

                "pct_abs_r_ge_07":
                    100 * np.mean(
                        abs_values >= 0.70
                    ),

                "pct_abs_r_ge_09":
                    100 * np.mean(
                        abs_values >= 0.90
                    ),
            })

    return pd.DataFrame(resultados)


# ============================================================
# RESUMEN POR CLASE
# ============================================================

def resumen_por_clase(df):

    if df.empty:
        return pd.DataFrame()

    columnas_numericas = [
        "n_pares",
        "r_media",
        "r_mediana",
        "r_p05",
        "r_p25",
        "r_p75",
        "r_p95",
        "abs_r_media",
        "abs_r_mediana",
        "abs_r_p95",
        "pct_abs_r_ge_03",
        "pct_abs_r_ge_05",
        "pct_abs_r_ge_07",
        "pct_abs_r_ge_09",
    ]

    resumen = (
        df.groupby("clase")[columnas_numericas]
        .mean()
        .reset_index()
    )

    return resumen


# ============================================================
# RESUMEN POR CANAL
# ============================================================

def resumen_por_canal(df):

    if df.empty:
        return pd.DataFrame()

    columnas_numericas = [
        "n_pares",
        "r_media",
        "r_mediana",
        "r_p05",
        "r_p25",
        "r_p75",
        "r_p95",
        "abs_r_media",
        "abs_r_mediana",
        "abs_r_p95",
        "pct_abs_r_ge_03",
        "pct_abs_r_ge_05",
        "pct_abs_r_ge_07",
        "pct_abs_r_ge_09",
    ]

    resumen = (
        df.groupby("canal")[columnas_numericas]
        .mean()
        .reset_index()
    )

    return resumen


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CARGAR
    # --------------------------------------------------------

    trials = cargar_trials()

    if len(trials) == 0:

        print()
        print("ERROR: no se encontraron trials válidos.")

        return

    # --------------------------------------------------------
    # CORRELACIONES
    # --------------------------------------------------------

    df = calcular_correlaciones(
        trials
    )

    if df.empty:

        print()
        print(
            "ERROR: no se pudieron calcular "
            "correlaciones."
        )

        return

    # --------------------------------------------------------
    # GUARDAR RESULTADO PRINCIPAL
    # --------------------------------------------------------

    archivo_principal = os.path.join(
        OUTPUT_DIR,
        "pearson_mismo_numero_letra_comando_entre_usuarios.csv"
    )

    df.to_csv(
        archivo_principal,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # RESUMEN POR CLASE
    # --------------------------------------------------------

    df_clase = resumen_por_clase(df)

    archivo_clase = os.path.join(
        OUTPUT_DIR,
        "pearson_resumen_por_clase.csv"
    )

    df_clase.to_csv(
        archivo_clase,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # RESUMEN POR CANAL
    # --------------------------------------------------------

    df_canal = resumen_por_canal(df)

    archivo_canal = os.path.join(
        OUTPUT_DIR,
        "pearson_resumen_por_canal.csv"
    )

    df_canal.to_csv(
        archivo_canal,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    print()
    print("RESUMEN POR CLASE")
    print()

    print(
        df_clase.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ARCHIVOS GENERADOS")
    print("=" * 70)

    print()
    print(
        f"Principal:\n  {archivo_principal}"
    )

    print(
        f"\nPor clase:\n  {archivo_clase}"
    )

    print(
        f"\nPor canal:\n  {archivo_canal}"
    )

    print()
    print("=" * 70)
    print("AUDITORÍA FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()