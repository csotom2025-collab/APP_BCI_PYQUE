# -*- coding: utf-8 -*-
"""
auditar_morfologia_eventos.py

Auditoría de morfología de eventos multicanal EEG.

Universo:
    - Se obtiene del reporte de archivos existente.
    - Solo se procesan archivos con incluido=True.
    - Los archivos EEG físicos están bajo:
          C:\\tmp_bci\\captures\\
    - Se verifica que existan físicamente.

Formato EEG esperado:
    Time,F3,FC5,AF3,F7,T7,P7,O1,O2,P8,T8,F8,AF4,FC6,F4

Definición de episodio:
    Para un umbral dado, cada muestra tiene un número de canales
    cuyo valor absoluto es >= umbral.

    Una muestra pertenece a un episodio si:
        número de canales simultáneos >= MIN_CANALES_SIMULTANEOS

    Un episodio es una secuencia contigua de esas muestras.

Para cada episodio se calculan:
    - umbral
    - ruta relativa
    - inicio
    - fin
    - n_muestras
    - max_canales_simultaneos
    - n_canales_participantes
    - max_abs
    - canales_en_pico

Salida:
    C:\\tmp_bci\\resultados_calidad\\auditoria_morfologia_eventos.csv

Importante:
    El índice de archivos físicos se construye una sola vez.
    No se ejecuta rglob() individualmente para cada archivo del reporte.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

BASE_DIR = Path(r"C:\tmp_bci")

# Directorio físico donde están los EEG.
CAPTURES_DIR = BASE_DIR / "captures"

RESULTADOS_DIR = BASE_DIR / "resultados_calidad"

SALIDA_CSV = (
    RESULTADOS_DIR
    / "auditoria_morfologia_eventos.csv"
)

SALIDA_ERRORES_CSV = (
    RESULTADOS_DIR
    / "auditoria_morfologia_eventos_errores.csv"
)

# Umbrales solicitados.
UMBRALES_UV = [
    120.0,
    150.0,
]

# Mínimo de canales que deben superar simultáneamente el umbral.
MIN_CANALES_SIMULTANEOS = 2

# Mostrar progreso cada N archivos.
PROGRESO_CADA = 250

# Canales EEG confirmados por la captura.
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

COL_TIME = "Time"


# ======================================================================
# UTILIDADES
# ======================================================================

def normalizar_nombre_columna(nombre) -> str:
    """
    Normaliza nombres de columnas para facilitar la detección.
    """

    if nombre is None:
        return ""

    s = str(nombre).strip()

    # Elimina BOM UTF-8 si aparece.
    s = s.replace("\ufeff", "")

    return s


def normalizar_ruta(s) -> str:
    """
    Convierte separadores a '/' y elimina detalles innecesarios.

    Ejemplo:
        UserX\\Letters\\archivo.csv
    ->
        UserX/Letters/archivo.csv
    """

    if s is None:
        return ""

    s = str(s).strip().strip('"').strip("'")

    s = s.replace("\\", "/")

    while "//" in s:
        s = s.replace("//", "/")

    return s.lstrip("./")


def bool_incluido(valor) -> bool:
    """
    Convierte distintas representaciones de True/False.
    """

    if isinstance(valor, bool):
        return valor

    if pd.isna(valor):
        return False

    s = str(valor).strip().lower()

    return s in {
        "true",
        "1",
        "yes",
        "y",
        "si",
        "sí",
        "included",
        "include",
        "incluido",
        "incluida",
    }


def encontrar_reporte() -> Path:
    """
    Busca automáticamente el reporte.

    Se priorizan CSV/XLSX/XLS dentro de resultados_calidad
    y luego dentro de C:\\tmp_bci.

    La búsqueda se hace solamente para localizar el reporte.
    No se utiliza para resolver los 6867 EEG.
    """

    candidatos: list[Path] = []

    extensiones = {
        ".csv",
        ".xlsx",
        ".xls",
    }

    carpetas = [
        RESULTADOS_DIR,
        BASE_DIR,
    ]

    vistas: set[str] = set()

    for carpeta in carpetas:

        if not carpeta.exists():
            continue

        try:

            for p in carpeta.rglob("*"):

                if not p.is_file():
                    continue

                if p.suffix.lower() not in extensiones:
                    continue

                try:
                    clave = str(
                        p.resolve()
                    ).lower()
                except Exception:
                    clave = str(p).lower()

                if clave in vistas:
                    continue

                vistas.add(clave)

                # No considerar nuestra salida.
                try:
                    if (
                        p.resolve()
                        == SALIDA_CSV.resolve()
                    ):
                        continue
                except Exception:
                    pass

                # Tampoco considerar el archivo de errores.
                try:
                    if (
                        p.resolve()
                        == SALIDA_ERRORES_CSV.resolve()
                    ):
                        continue
                except Exception:
                    pass

                candidatos.append(p)

        except Exception:
            pass

    if not candidatos:
        raise FileNotFoundError(
            "No se encontró ningún reporte CSV/XLS/XLSX "
            f"dentro de {BASE_DIR}"
        )

    # Prioridades por nombre.
    patrones_prioridad = [
        "reporte",
        "report",
        "archivos",
        "inclusion",
        "incluidos",
        "universo",
        "calidad",
    ]

    def score(p: Path) -> tuple[int, int, int]:

        nombre = p.stem.lower()

        prioridad = 0

        for i, patron in enumerate(
            patrones_prioridad
        ):
            if patron in nombre:
                prioridad += 100 - i

        # Los archivos de resultados_calidad tienen prioridad.
        try:
            if (
                p.resolve()
                .parent
                == RESULTADOS_DIR.resolve()
            ):
                prioridad += 50
        except Exception:
            pass

        # Preferir archivos relativamente grandes.
        try:
            tam = p.stat().st_size
        except Exception:
            tam = 0

        return (
            prioridad,
            min(tam, 10_000_000),
            -len(str(p)),
        )

    candidatos.sort(
        key=score,
        reverse=True,
    )

    return candidatos[0]


def leer_tabla_reporte(
    path: Path,
) -> pd.DataFrame:
    """
    Lee CSV/XLS/XLSX con tolerancia a distintos separadores
    y encoding.
    """

    suffix = path.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        return pd.read_excel(path)

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    ultimo_error = None

    for encoding in encodings:

        try:

            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
            )

        except Exception as e:

            ultimo_error = e

    raise RuntimeError(
        f"No se pudo leer el reporte {path}: "
        f"{ultimo_error}"
    )


def encontrar_columna(
    df: pd.DataFrame,
    candidatos: Iterable[str],
) -> str | None:
    """
    Busca una columna ignorando mayúsculas, espacios y separadores.
    """

    mapa = {}

    for c in df.columns:

        original = str(c)

        limpio = re.sub(
            r"[^a-z0-9áéíóúüñ]",
            "",
            original.lower(),
        )

        mapa[limpio] = original

    for candidato in candidatos:

        limpio = re.sub(
            r"[^a-z0-9áéíóúüñ]",
            "",
            candidato.lower(),
        )

        if limpio in mapa:
            return mapa[limpio]

    return None


def localizar_columna_ruta(
    df: pd.DataFrame,
) -> str:
    """
    Encuentra la columna que contiene la ruta relativa.
    """

    candidatos = [
        "ruta_relativa",
        "ruta relativa",
        "ruta",
        "archivo",
        "file",
        "filepath",
        "file_path",
        "path",
        "relative_path",
        "relativepath",
        "nombre_archivo",
        "nombre archivo",
    ]

    col = encontrar_columna(
        df,
        candidatos,
    )

    if col is None:

        raise RuntimeError(
            "No se encontró una columna de ruta "
            "en el reporte.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    return col


def localizar_columna_incluido(
    df: pd.DataFrame,
) -> str:
    """
    Encuentra la columna incluido=True.
    """

    candidatos = [
        "incluido",
        "incluida",
        "include",
        "included",
        "incluir",
        "usar",
        "seleccionado",
        "seleccionada",
    ]

    col = encontrar_columna(
        df,
        candidatos,
    )

    if col is None:

        raise RuntimeError(
            "No se encontró una columna "
            "'incluido' en el reporte.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    return col


# ======================================================================
# ÍNDICE FÍSICO DE ARCHIVOS
# ======================================================================

def construir_indice_archivos() -> dict[str, list[Path]]:
    """
    Construye una sola vez un índice de los CSV existentes.

    Los archivos EEG están normalmente bajo:

        C:\\tmp_bci\\captures\\UserX\\Letters\\archivo.csv

    El reporte contiene rutas relativas como:

        UserX/Letters/archivo.csv

    El índice permite resolver las rutas rápidamente sin ejecutar
    rglob() una vez por cada uno de los 6867 archivos.
    """

    if not CAPTURES_DIR.exists():

        raise FileNotFoundError(
            "No existe el directorio de capturas:\n"
            f"{CAPTURES_DIR}"
        )

    if not CAPTURES_DIR.is_dir():

        raise NotADirectoryError(
            "La ruta de capturas no es un directorio:\n"
            f"{CAPTURES_DIR}"
        )

    print()
    print(
        "Construyendo índice de archivos EEG..."
    )

    print(
        f"Directorio de capturas: "
        f"{CAPTURES_DIR}"
    )

    indice: dict[str, list[Path]] = {}

    contador = 0

    # Solamente recorremos captures UNA VEZ.
    for path in CAPTURES_DIR.rglob("*.csv"):

        if not path.is_file():
            continue

        contador += 1

        # Ruta relativa respecto a captures.
        #
        # UserAkatzin/Letters/UserAkatzin_A_0.csv
        relativa = path.relative_to(
            CAPTURES_DIR
        )

        clave_relativa = normalizar_ruta(
            relativa.as_posix()
        ).lower()

        indice.setdefault(
            clave_relativa,
            [],
        ).append(path)

        # Índice secundario por nombre.
        #
        # Solo se utilizará si el nombre es único.
        clave_nombre = path.name.lower()

        indice.setdefault(
            f"__NAME__:{clave_nombre}",
            [],
        ).append(path)

        # Mostrar avance del indexado para evitar sensación
        # de que el proceso está detenido.
        if (
            contador == 1
            or contador % 2000 == 0
        ):
            print(
                f"  CSV indexados: {contador}"
            )

    print(
        f"Archivos CSV indexados: {contador}"
    )

    return indice


def resolver_archivo(
    ruta_relativa: str,
    indice_archivos: dict[str, list[Path]],
) -> Path | None:
    """
    Resuelve una ruta del reporte.

    Prioridad:

    1. Ruta absoluta existente.
    2. C:\\tmp_bci\\captures\\<ruta>.
    3. C:\\tmp_bci\\<ruta>.
    4. Coincidencia exacta en el índice.
    5. Coincidencia única por nombre.

    No realiza rglob().
    """

    ruta = normalizar_ruta(
        ruta_relativa
    )

    if not ruta:
        return None

    # --------------------------------------------------------------
    # 1. Ruta absoluta
    # --------------------------------------------------------------

    p = Path(ruta)

    if p.is_absolute():

        if (
            p.exists()
            and p.is_file()
        ):
            return p

    # --------------------------------------------------------------
    # 2. Ubicación real esperada:
    #
    # C:\tmp_bci\captures\UserX\Letters\archivo.csv
    # --------------------------------------------------------------

    candidato = (
        CAPTURES_DIR
        / Path(ruta)
    )

    if (
        candidato.exists()
        and candidato.is_file()
    ):
        return candidato

    # --------------------------------------------------------------
    # 3. Compatibilidad:
    #
    # C:\tmp_bci\UserX\Letters\archivo.csv
    # --------------------------------------------------------------

    candidato = (
        BASE_DIR
        / Path(ruta)
    )

    if (
        candidato.exists()
        and candidato.is_file()
    ):
        return candidato

    # --------------------------------------------------------------
    # 4. Índice por ruta relativa
    # --------------------------------------------------------------

    encontrados = indice_archivos.get(
        ruta.lower(),
        [],
    )

    if len(encontrados) == 1:
        return encontrados[0]

    # --------------------------------------------------------------
    # 5. Fallback por nombre.
    #
    # Solamente se acepta si el nombre es único.
    # --------------------------------------------------------------

    nombre = Path(ruta).name.lower()

    encontrados = indice_archivos.get(
        f"__NAME__:{nombre}",
        [],
    )

    if len(encontrados) == 1:
        return encontrados[0]

    # 0 = no encontrado
    # >1 = ambiguo
    return None


# ======================================================================
# LECTURA EEG
# ======================================================================

def leer_eeg(
    path: Path,
) -> pd.DataFrame:
    """
    Lee un CSV EEG.

    Se utiliza pandas para conservar exactamente el índice
    de muestra.
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    ultimo_error = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
            )

            df.columns = [
                normalizar_nombre_columna(c)
                for c in df.columns
            ]

            return df

        except Exception as e:

            ultimo_error = e

    raise RuntimeError(
        f"No se pudo leer {path}: "
        f"{ultimo_error}"
    )


def preparar_matriz_eeg(
    df: pd.DataFrame,
    path: Path,
) -> np.ndarray:
    """
    Extrae los 14 canales y devuelve una matriz float64:

        filas    = muestras
        columnas = canales

    Los valores no numéricos se convierten en NaN.
    """

    columnas_presentes = {
        str(c): c
        for c in df.columns
    }

    faltantes = [
        canal
        for canal in CANALES
        if canal not in columnas_presentes
    ]

    if faltantes:

        raise ValueError(
            f"Faltan canales en {path}: "
            f"{faltantes}. "
            f"Columnas disponibles: "
            f"{list(df.columns)}"
        )

    matriz = []

    for canal in CANALES:

        serie = pd.to_numeric(
            df[columnas_presentes[canal]],
            errors="coerce",
        )

        matriz.append(
            serie.to_numpy(
                dtype=np.float64
            )
        )

    x = np.column_stack(matriz)

    return x


# ======================================================================
# DETECCIÓN DE EPISODIOS
# ======================================================================

def detectar_episodios(
    x: np.ndarray,
    umbral: float,
    min_canales: int,
) -> list[dict]:
    """
    Detecta episodios multicanal.

    Una muestra es válida si >= min_canales tienen:

        abs(valor) >= umbral

    Las muestras válidas contiguas forman un episodio.
    """

    if x.size == 0:
        return []

    # --------------------------------------------------------------
    # 1. Umbral por canal y muestra
    # --------------------------------------------------------------

    abs_x = np.abs(x)

    mascara = (
        np.isfinite(abs_x)
        & (abs_x >= umbral)
    )

    n_simultaneos = (
        mascara.sum(axis=1)
    )

    activa = (
        n_simultaneos
        >= min_canales
    )

    if not np.any(activa):
        return []

    # --------------------------------------------------------------
    # 2. Encontrar segmentos contiguos
    # --------------------------------------------------------------

    cambios = np.diff(
        activa.astype(np.int8)
    )

    inicios = list(
        np.flatnonzero(
            cambios == 1
        ) + 1
    )

    finales = list(
        np.flatnonzero(
            cambios == -1
        )
    )

    if activa[0]:
        inicios.insert(
            0,
            0,
        )

    if activa[-1]:
        finales.append(
            len(activa) - 1
        )

    episodios = []

    # --------------------------------------------------------------
    # 3. Métricas por episodio
    # --------------------------------------------------------------

    for inicio, fin in zip(
        inicios,
        finales,
    ):

        segmento_mascara = (
            mascara[
                inicio : fin + 1
            ]
        )

        segmento_abs = (
            abs_x[
                inicio : fin + 1
            ]
        )

        # Máximo de canales simultáneos
        # dentro del episodio.
        max_sim = int(
            n_simultaneos[
                inicio : fin + 1
            ].max()
        )

        # Canales que participaron al menos una vez.
        participantes = np.any(
            segmento_mascara,
            axis=0,
        )

        n_participantes = int(
            participantes.sum()
        )

        # Máximo absoluto de toda la ventana.
        if np.any(
            np.isfinite(
                segmento_abs
            )
        ):

            max_abs = float(
                np.nanmax(
                    segmento_abs
                )
            )

        else:

            max_abs = float("nan")

        # ----------------------------------------------------------
        # Canales en el punto de máxima simultaneidad.
        # ----------------------------------------------------------

        segmento_sim = (
            n_simultaneos[
                inicio : fin + 1
            ]
        )

        posiciones_max = (
            np.flatnonzero(
                segmento_sim == max_sim
            )
        )

        # Primer punto donde se alcanza
        # la máxima simultaneidad.
        pos_relativa = int(
            posiciones_max[0]
        )

        indice_pico = (
            inicio
            + pos_relativa
        )

        canales_en_pico = [
            CANALES[j]
            for j in np.flatnonzero(
                mascara[indice_pico]
            )
        ]

        episodios.append(
            {
                "umbral_uv": float(
                    umbral
                ),
                "inicio": int(
                    inicio
                ),
                "fin": int(
                    fin
                ),
                "n_muestras": int(
                    fin - inicio + 1
                ),
                "max_canales_simultaneos": (
                    max_sim
                ),
                "n_canales_participantes": (
                    n_participantes
                ),
                "max_abs": max_abs,
                "canales_en_pico": "|".join(
                    canales_en_pico
                ),
            }
        )

    return episodios


# ======================================================================
# ESTADÍSTICAS / REPORTE
# ======================================================================

def percentil(
    valores: list[float],
    q: float,
) -> float:
    """
    Percentil robusto.
    """

    if not valores:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(
                valores,
                dtype=float,
            ),
            q,
        )
    )


def mediana(
    valores: list[float],
) -> float:

    if not valores:
        return float("nan")

    return float(
        np.median(
            np.asarray(
                valores,
                dtype=float,
            )
        )
    )


def fmt_num(
    x: float,
    decimales: int = 2,
) -> str:

    if (
        x is None
        or not math.isfinite(
            float(x)
        )
    ):
        return "nan"

    return (
        f"{float(x):.{decimales}f}"
    )


def imprimir_estadisticas(
    df_eventos: pd.DataFrame,
    umbral: float,
) -> None:

    sub = df_eventos[
        df_eventos[
            "umbral_uv"
        ] == umbral
    ].copy()

    print()
    print(
        "=" * 70
    )
    print(
        f"UMBRAL: {umbral:.0f} uV"
    )
    print(
        "=" * 70
    )

    if sub.empty:

        print(
            "Episodios: 0"
        )

        print(
            "Archivos afectados: 0"
        )

        return

    n_episodios = len(
        sub
    )

    n_archivos = (
        sub[
            "ruta_relativa"
        ]
        .nunique()
    )

    print(
        f"Episodios: {n_episodios}"
    )

    print(
        f"Archivos afectados: "
        f"{n_archivos}"
    )

    # --------------------------------------------------------------
    # Duración
    # --------------------------------------------------------------

    duraciones = (
        sub[
            "n_muestras"
        ]
        .astype(float)
        .tolist()
    )

    print()
    print(
        "Duración (muestras):"
    )

    print(
        f"  mediana = "
        f"{fmt_num(mediana(duraciones), 2)}"
    )

    print(
        f"  P95     = "
        f"{fmt_num(percentil(duraciones, 95), 2)}"
    )

    print(
        f"  máximo  = "
        f"{int(max(duraciones))}"
    )

    # --------------------------------------------------------------
    # Simultaneidad
    # --------------------------------------------------------------

    simult = (
        sub[
            "max_canales_simultaneos"
        ]
        .astype(float)
        .tolist()
    )

    print()
    print(
        "Máximo de canales simultáneos:"
    )

    print(
        f"  mediana = "
        f"{fmt_num(mediana(simult), 2)}"
    )

    print(
        f"  P95     = "
        f"{fmt_num(percentil(simult, 95), 2)}"
    )

    print(
        f"  máximo  = "
        f"{int(max(simult))}"
    )

    # --------------------------------------------------------------
    # Participantes
    # --------------------------------------------------------------

    participantes = (
        sub[
            "n_canales_participantes"
        ]
        .astype(float)
        .tolist()
    )

    print()
    print(
        "Número de canales participantes:"
    )

    print(
        f"  mediana = "
        f"{fmt_num(mediana(participantes), 2)}"
    )

    print(
        f"  P95     = "
        f"{fmt_num(percentil(participantes, 95), 2)}"
    )

    print(
        f"  máximo  = "
        f"{int(max(participantes))}"
    )

    # --------------------------------------------------------------
    # Amplitud
    # --------------------------------------------------------------

    amplitudes = (
        sub[
            "max_abs"
        ]
        .astype(float)
        .dropna()
        .tolist()
    )

    print()
    print(
        "Amplitud máxima absoluta:"
    )

    if amplitudes:

        print(
            f"  mediana = "
            f"{fmt_num(mediana(amplitudes), 3)} uV"
        )

        print(
            f"  P95     = "
            f"{fmt_num(percentil(amplitudes, 95), 3)} uV"
        )

        print(
            f"  máximo  = "
            f"{fmt_num(max(amplitudes), 3)} uV"
        )

    else:

        print(
            "  sin valores numéricos"
        )

    # --------------------------------------------------------------
    # Episodios de 14 canales
    # --------------------------------------------------------------

    n_14 = int(
        (
            sub[
                "max_canales_simultaneos"
            ]
            == len(CANALES)
        ).sum()
    )

    porcentaje_14 = (
        100.0
        * n_14
        / n_episodios
        if n_episodios
        else 0.0
    )

    print()
    print(
        f"Episodios con "
        f"{len(CANALES)} "
        f"canales simultáneos:"
    )

    print(
        f"  {n_14} "
        f"({porcentaje_14:.3f}%)"
    )

    # --------------------------------------------------------------
    # Distribución
    # --------------------------------------------------------------

    print()
    print(
        "Distribución de máximo "
        "de canales simultáneos:"
    )

    distribucion = (
        sub[
            "max_canales_simultaneos"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "max_canales_simultaneos"
    )

    for (
        n_canales,
        cantidad,
    ) in distribucion.items():

        print(
            f"{int(n_canales):<2} "
            f"{int(cantidad):>7}"
        )

    # --------------------------------------------------------------
    # Top 10
    # --------------------------------------------------------------

    print()
    print(
        "Top 10 episodios por "
        "máxima simultaneidad:"
    )

    top = (
        sub.sort_values(
            by=[
                "max_canales_simultaneos",
                "n_muestras",
                "max_abs",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(10)
    )

    columnas = [
        "ruta_relativa",
        "inicio",
        "fin",
        "n_muestras",
        "max_canales_simultaneos",
        "n_canales_participantes",
        "max_abs",
        "canales_en_pico",
    ]

    with pd.option_context(
        "display.max_colwidth",
        100,
        "display.width",
        220,
        "display.max_columns",
        20,
    ):

        print(
            top[
                columnas
            ].to_string(
                index=False
            )
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> int:

    print()
    print(
        "=" * 70
    )
    print(
        "UNIVERSO DE AUDITORÍA"
    )
    print(
        "=" * 70
    )

    print(
        f"Directorio base: "
        f"{BASE_DIR}"
    )

    print(
        f"Directorio EEG: "
        f"{CAPTURES_DIR}"
    )

    if not BASE_DIR.exists():

        print(
            f"ERROR: no existe "
            f"{BASE_DIR}",
            file=sys.stderr,
        )

        return 1

    if not CAPTURES_DIR.exists():

        print(
            f"ERROR: no existe "
            f"{CAPTURES_DIR}",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------------
    # Buscar reporte
    # --------------------------------------------------------------

    try:

        reporte = encontrar_reporte()

    except Exception as e:

        print(
            f"ERROR buscando reporte: {e}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Reporte utilizado: "
        f"{reporte}"
    )

    # --------------------------------------------------------------
    # Leer reporte
    # --------------------------------------------------------------

    try:

        df_reporte = leer_tabla_reporte(
            reporte
        )

    except Exception as e:

        print(
            f"ERROR leyendo reporte: {e}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Archivos en reporte: "
        f"{len(df_reporte)}"
    )

    # --------------------------------------------------------------
    # Identificar columnas
    # --------------------------------------------------------------

    try:

        col_ruta = (
            localizar_columna_ruta(
                df_reporte
            )
        )

        col_incluido = (
            localizar_columna_incluido(
                df_reporte
            )
        )

    except Exception as e:

        print(
            "ERROR identificando "
            "columnas del reporte: "
            f"{e}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Columna ruta: "
        f"{col_ruta}"
    )

    print(
        f"Columna incluido: "
        f"{col_incluido}"
    )

    # --------------------------------------------------------------
    # Filtrar incluido=True
    # --------------------------------------------------------------

    mask_incluido = (
        df_reporte[
            col_incluido
        ]
        .apply(bool_incluido)
    )

    df_incluidos = (
        df_reporte.loc[
            mask_incluido
        ]
        .copy()
    )

    print(
        f"Archivos incluido=True: "
        f"{len(df_incluidos)}"
    )

    # --------------------------------------------------------------
    # Construir índice físico
    # --------------------------------------------------------------

    try:

        indice_archivos = (
            construir_indice_archivos()
        )

    except Exception as e:

        print(
            f"ERROR construyendo índice "
            f"de archivos: {e}",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------------
    # Resolver rutas
    # --------------------------------------------------------------

    entradas = []

    encontrados = 0
    no_encontrados = 0

    rutas_no_encontradas = []

    for _, row in (
        df_incluidos.iterrows()
    ):

        ruta_relativa = (
            normalizar_ruta(
                row[col_ruta]
            )
        )

        path = resolver_archivo(
            ruta_relativa,
            indice_archivos,
        )

        if path is not None:

            encontrados += 1

            entradas.append(
                (
                    ruta_relativa,
                    path,
                )
            )

        else:

            no_encontrados += 1

            if (
                len(
                    rutas_no_encontradas
                )
                < 20
            ):

                rutas_no_encontradas.append(
                    ruta_relativa
                )

    print()
    print(
        f"Archivos encontrados: "
        f"{encontrados}"
    )

    print(
        f"Archivos no encontrados: "
        f"{no_encontrados}"
    )

    if rutas_no_encontradas:

        print()
        print(
            "Primeros archivos "
            "no encontrados:"
        )

        for ruta in (
            rutas_no_encontradas
        ):

            print(
                f"  {ruta}"
            )

        if (
            no_encontrados
            > len(
                rutas_no_encontradas
            )
        ):

            print(
                f"  ... y "
                f"{no_encontrados - len(rutas_no_encontradas)} "
                f"adicionales."
            )

    if no_encontrados:

        print()
        print(
            "AVISO: existen archivos "
            "incluidos en el reporte "
            "que no se encontraron "
            "físicamente."
        )

    if not entradas:

        print(
            "ERROR: no hay archivos "
            "para procesar.",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------------
    # Cabecera auditoría
    # --------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "AUDITORÍA DE MORFOLOGÍA "
        "DE EVENTOS MULTICANAL"
    )
    print(
        "=" * 70
    )

    print(
        f"Umbrales: "
        f"{UMBRALES_UV}"
    )

    print(
        f"Mínimo de canales simultáneos: "
        f"{MIN_CANALES_SIMULTANEOS}"
    )

    # --------------------------------------------------------------
    # Procesamiento
    # --------------------------------------------------------------

    eventos = []

    errores = []

    total = len(
        entradas
    )

    for (
        i,
        (
            ruta_relativa,
            path,
        ),
    ) in enumerate(
        entradas,
        start=1,
    ):

        if (
            i == 1
            or i % PROGRESO_CADA == 0
            or i == total
        ):

            print(
                f"[{i:5d}/{total}] "
                f"{ruta_relativa}"
            )

        try:

            df = leer_eeg(
                path
            )

            x = preparar_matriz_eeg(
                df,
                path,
            )

            for umbral in (
                UMBRALES_UV
            ):

                episodios = (
                    detectar_episodios(
                        x=x,
                        umbral=umbral,
                        min_canales=(
                            MIN_CANALES_SIMULTANEOS
                        ),
                    )
                )

                for evento in (
                    episodios
                ):

                    evento[
                        "ruta_relativa"
                    ] = ruta_relativa

                    eventos.append(
                        evento
                    )

        except Exception as e:

            errores.append(
                {
                    "ruta_relativa": (
                        ruta_relativa
                    ),
                    "archivo_fisico": (
                        str(path)
                    ),
                    "error": str(e),
                }
            )

    # --------------------------------------------------------------
    # DataFrame final
    # --------------------------------------------------------------

    columnas_salida = [
        "umbral_uv",
        "ruta_relativa",
        "inicio",
        "fin",
        "n_muestras",
        "max_canales_simultaneos",
        "n_canales_participantes",
        "max_abs",
        "canales_en_pico",
    ]

    df_eventos = pd.DataFrame(
        eventos,
        columns=columnas_salida,
    )

    if not df_eventos.empty:

        df_eventos = (
            df_eventos.sort_values(
                by=[
                    "umbral_uv",
                    "ruta_relativa",
                    "inicio",
                    "fin",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------------
    # Crear directorio salida
    # --------------------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------------

    df_eventos.to_csv(
        SALIDA_CSV,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    # --------------------------------------------------------------
    # Guardar errores si existen
    # --------------------------------------------------------------

    if errores:

        pd.DataFrame(
            errores
        ).to_csv(
            SALIDA_ERRORES_CSV,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )

    else:

        # Si existe un archivo viejo de errores,
        # lo eliminamos para que no quede información obsoleta.
        try:

            if SALIDA_ERRORES_CSV.exists():
                SALIDA_ERRORES_CSV.unlink()

        except Exception:
            pass

    # --------------------------------------------------------------
    # Resultado general
    # --------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "RESULTADO"
    )
    print(
        "=" * 70
    )

    print(
        f"Archivos encontrados: "
        f"{encontrados}"
    )

    print(
        f"Archivos procesados "
        f"correctamente: "
        f"{encontrados - len(errores)}"
    )

    print(
        f"Archivos con error: "
        f"{len(errores)}"
    )

    print(
        f"Episodios multicanal "
        f"encontrados: "
        f"{len(df_eventos)}"
    )

    print()
    print(
        "Archivo generado:"
    )

    print(
        SALIDA_CSV
    )

    # --------------------------------------------------------------
    # Estadísticas por umbral
    # --------------------------------------------------------------

    for umbral in (
        UMBRALES_UV
    ):

        imprimir_estadisticas(
            df_eventos,
            umbral,
        )

    # --------------------------------------------------------------
    # Errores
    # --------------------------------------------------------------

    if errores:

        print()
        print(
            "=" * 70
        )
        print(
            "ERRORES DE PROCESAMIENTO"
        )
        print(
            "=" * 70
        )

        for error in errores[:50]:

            print(
                f"{error['ruta_relativa']}: "
                f"{error['error']}"
            )

        if len(errores) > 50:

            print(
                f"... y "
                f"{len(errores) - 50} "
                f"errores adicionales."
            )

        print()
        print(
            "Errores guardados en:"
        )

        print(
            SALIDA_ERRORES_CSV
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "AUDITORÍA FINALIZADA"
    )
    print(
        "=" * 70
    )

    return 0


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
