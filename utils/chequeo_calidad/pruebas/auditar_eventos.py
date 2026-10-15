from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"C:\tmp_bci\captures")
REPORTE = Path(r"C:\tmp_bci\resultados_calidad\reporte.csv")

SALIDA = Path(r"C:\tmp_bci\resultados_calidad\auditoria_eventos.csv")

UMBRALES = [120.0, 150.0]

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
# CARGAR UNIVERSO DE AUDITORÍA
# ============================================================

def cargar_archivos_incluidos():
    """
    Carga exclusivamente los archivos marcados como incluidos
    en resultados_calidad/reporte.csv.

    No realiza una nueva decisión de inclusión/exclusión.
    """

    if not REPORTE.exists():
        raise FileNotFoundError(
            f"No existe el reporte:\n{REPORTE}"
        )

    reporte = pd.read_csv(REPORTE)

    columnas_requeridas = {
        "ruta_relativa",
        "incluido",
    }

    faltantes = columnas_requeridas - set(reporte.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas en reporte.csv: "
            + ", ".join(sorted(faltantes))
        )

    incluidos = reporte[
        reporte["incluido"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "si", "sí"])
    ].copy()

    rutas = []

    for ruta_relativa in incluidos["ruta_relativa"]:

        path = ROOT / Path(str(ruta_relativa))

        if not path.exists():

            print(
                "ADVERTENCIA: archivo incluido no encontrado:"
            )
            print(path)

            continue

        rutas.append(path)

    print("=" * 70)
    print("UNIVERSO DE AUDITORÍA")
    print("=" * 70)

    print(
        f"Archivos incluidos según reporte: "
        f"{len(incluidos)}"
    )

    print(
        f"Archivos encontrados físicamente: "
        f"{len(rutas)}"
    )

    if len(rutas) != len(incluidos):

        print()
        print(
            "ADVERTENCIA: el número de archivos encontrados "
            "no coincide con el número de archivos incluidos."
        )

    print()

    return sorted(rutas)


# ============================================================
# DETECCIÓN DE EPISODIOS
# ============================================================

def detectar_episodios(mask):
    """
    Recibe un vector booleano.

    True = muestra que supera el umbral.

    Devuelve una lista de episodios consecutivos:

        [
            (inicio, fin),
            ...
        ]

    donde fin es inclusivo.
    """

    mask = np.asarray(mask, dtype=bool)

    episodios = []

    inicio = None

    for i, activo in enumerate(mask):

        if activo and inicio is None:

            inicio = i

        elif not activo and inicio is not None:

            episodios.append(
                (inicio, i - 1)
            )

            inicio = None

    # Episodio que llega hasta la última muestra
    if inicio is not None:

        episodios.append(
            (inicio, len(mask) - 1)
        )

    return episodios


# ============================================================
# ANALIZAR UN CANAL
# ============================================================

def analizar_canal(
    valores,
    umbral,
):
    """
    Analiza un único canal para un único umbral.

    La detección se realiza sobre:

        abs(valor - mediana_del_canal) > umbral

    Devuelve los episodios encontrados.
    """

    valores = np.asarray(
        valores,
        dtype=float,
    )

    if len(valores) == 0:
        return []

    mediana = np.median(valores)

    centrado = valores - mediana

    magnitud = np.abs(centrado)

    mask = magnitud > umbral

    episodios = detectar_episodios(mask)

    resultados = []

    for inicio, fin in episodios:

        segmento = centrado[inicio:fin + 1]

        abs_segmento = np.abs(segmento)

        indice_local_max = int(
            np.argmax(abs_segmento)
        )

        indice_max = inicio + indice_local_max

        valor_max_abs = float(
            abs_segmento[indice_local_max]
        )

        valor_en_max = float(
            segmento[indice_local_max]
        )

        valor_max = float(
            np.max(segmento)
        )

        valor_min = float(
            np.min(segmento)
        )

        n_muestras = fin - inicio + 1

        resultados.append(
            {
                "inicio": inicio,
                "fin": fin,
                "n_muestras": n_muestras,

                "max_abs": valor_max_abs,
                "valor_en_max_abs": valor_en_max,

                "max_valor": valor_max,
                "min_valor": valor_min,

                "muestra_max_abs": indice_max,
            }
        )

    return resultados


# ============================================================
# ANALIZAR ARCHIVO
# ============================================================

def analizar_archivo(path):
    """
    Analiza un archivo incluido.

    No modifica el archivo original.

    Devuelve una lista de eventos.
    """

    try:

        df = pd.read_csv(path)

    except Exception as e:

        print()
        print("ERROR leyendo:")
        print(path)
        print(e)

        return [], "error_lectura"

    # --------------------------------------------------------
    # Verificar canales
    # --------------------------------------------------------

    canales_faltantes = [
        canal
        for canal in CANALES
        if canal not in df.columns
    ]

    if canales_faltantes:

        return [], (
            "canales_faltantes:"
            + ",".join(canales_faltantes)
        )

    # --------------------------------------------------------
    # Extraer EEG
    # --------------------------------------------------------

    X = df[CANALES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Eliminar filas donde no existe ningún canal válido
    # --------------------------------------------------------

    validas = X.notna().any(axis=1)

    X = X.loc[validas].reset_index(drop=True)

    if len(X) == 0:

        return [], "sin_muestras_eeg_validas"

    # --------------------------------------------------------
    # Convertir a numpy
    # --------------------------------------------------------

    X = X.to_numpy(dtype=float)

    n_muestras = X.shape[0]

    # --------------------------------------------------------
    # Medianas por canal
    # --------------------------------------------------------

    medianas = np.nanmedian(
        X,
        axis=0,
    )

    # --------------------------------------------------------
    # Centrado por mediana
    # --------------------------------------------------------

    X_centrado = X - medianas

    # --------------------------------------------------------
    # Procesar umbrales
    # --------------------------------------------------------

    eventos = []

    for indice_canal, canal in enumerate(CANALES):

        valores = X_centrado[:, indice_canal]

        for umbral in UMBRALES:

            episodios = analizar_canal(
                valores,
                umbral,
            )

            for episodio in episodios:

                inicio = episodio["inicio"]
                fin = episodio["fin"]

                # ------------------------------------------------
                # Posición relativa del episodio
                # ------------------------------------------------

                posicion_inicio = (
                    inicio / (n_muestras - 1)
                    if n_muestras > 1
                    else 0.0
                )

                posicion_fin = (
                    fin / (n_muestras - 1)
                    if n_muestras > 1
                    else 0.0
                )

                # ------------------------------------------------
                # Duración relativa
                # ------------------------------------------------

                proporcion_trial = (
                    episodio["n_muestras"]
                    / n_muestras
                )

                eventos.append(
                    {
                        "ruta": str(path),

                        "canal": canal,

                        "umbral_uv": umbral,

                        "n_muestras_trial": n_muestras,

                        "inicio": inicio,

                        "fin": fin,

                        "n_muestras": episodio[
                            "n_muestras"
                        ],

                        "proporcion_trial": (
                            proporcion_trial
                        ),

                        "posicion_inicio": (
                            posicion_inicio
                        ),

                        "posicion_fin": (
                            posicion_fin
                        ),

                        "max_abs": episodio[
                            "max_abs"
                        ],

                        "valor_en_max_abs": episodio[
                            "valor_en_max_abs"
                        ],

                        "max_valor": episodio[
                            "max_valor"
                        ],

                        "min_valor": episodio[
                            "min_valor"
                        ],

                        "muestra_max_abs": episodio[
                            "muestra_max_abs"
                        ],

                        "mediana_canal": float(
                            medianas[indice_canal]
                        ),
                    }
                )

    # ========================================================
    # Calcular número de canales afectados por archivo/umbral
    # ========================================================

    if eventos:

        temp = pd.DataFrame(eventos)

        afectados = (
            temp.groupby("umbral_uv")["canal"]
            .nunique()
            .to_dict()
        )

        for evento in eventos:

            evento["n_canales_afectados_archivo"] = int(
                afectados.get(
                    evento["umbral_uv"],
                    0,
                )
            )

    return eventos, None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    archivos = cargar_archivos_incluidos()

    resultados = []

    omitidos = []

    print("=" * 70)
    print("INICIO DE AUDITORÍA DE EVENTOS")
    print("=" * 70)

    print(
        f"Archivos a analizar: {len(archivos)}"
    )

    print(
        f"Umbrales exploratorios: {UMBRALES}"
    )

    print()

    # --------------------------------------------------------
    # Procesar archivos
    # --------------------------------------------------------

    for i, path in enumerate(archivos, start=1):

        eventos, motivo = analizar_archivo(path)

        if motivo is not None:

            omitidos.append(
                {
                    "ruta": str(path),
                    "motivo": motivo,
                }
            )

        else:

            resultados.extend(eventos)

        # ----------------------------------------------------
        # Progreso
        # ----------------------------------------------------

        if i % 500 == 0 or i == len(archivos):

            print(
                f"Procesados: {i}/{len(archivos)}"
            )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    columnas = [
        "ruta",
        "canal",
        "umbral_uv",
        "n_muestras_trial",
        "inicio",
        "fin",
        "n_muestras",
        "proporcion_trial",
        "posicion_inicio",
        "posicion_fin",
        "max_abs",
        "valor_en_max_abs",
        "max_valor",
        "min_valor",
        "muestra_max_abs",
        "mediana_canal",
        "n_canales_afectados_archivo",
    ]

    if resultados:

        eventos_df = pd.DataFrame(
            resultados
        )

        # Garantizar orden de columnas
        eventos_df = eventos_df[
            columnas
        ]

        # Orden lógico
        eventos_df = eventos_df.sort_values(
            [
                "umbral_uv",
                "ruta",
                "canal",
                "inicio",
            ]
        ).reset_index(drop=True)

    else:

        eventos_df = pd.DataFrame(
            columns=columnas
        )

    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    eventos_df.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)

    print(
        f"Archivos evaluados: {len(archivos)}"
    )

    print(
        f"Archivos omitidos: {len(omitidos)}"
    )

    print(
        f"Eventos encontrados: {len(eventos_df)}"
    )

    print(
        f"Archivo generado:\n{SALIDA}"
    )

    # ========================================================
    # RESUMEN POR UMBRAL
    # ========================================================

    for umbral in UMBRALES:

        print()
        print("=" * 70)
        print(
            f"UMBRAL: {umbral} µV"
        )
        print("=" * 70)

        if eventos_df.empty:

            print(
                "No se encontraron eventos."
            )

            continue

        e = eventos_df[
            eventos_df["umbral_uv"] == umbral
        ]

        print(
            f"Eventos: {len(e)}"
        )

        print(
            f"Archivos afectados: "
            f"{e['ruta'].nunique()}"
        )

        print(
            f"Canales afectados: "
            f"{e['canal'].nunique()}"
        )

        print(
            f"Duración del episodio "
            f"(mediana): "
            f"{e['n_muestras'].median():.2f} muestras"
        )

        print(
            f"Duración del episodio "
            f"(P95): "
            f"{e['n_muestras'].quantile(0.95):.2f} muestras"
        )

        print(
            f"Duración máxima: "
            f"{e['n_muestras'].max()} muestras"
        )

        print(
            f"Max_abs del episodio "
            f"(mediana): "
            f"{e['max_abs'].median():.3f} µV"
        )

        print(
            f"Max_abs del episodio "
            f"(P95): "
            f"{e['max_abs'].quantile(0.95):.3f} µV"
        )

        print(
            f"Max_abs máximo: "
            f"{e['max_abs'].max():.3f} µV"
        )

        print()
        print(
            "Número de canales afectados "
            "por archivo:"
        )

        por_archivo = (
            e.groupby("ruta")[
                "canal"
            ]
            .nunique()
            .value_counts()
            .sort_index()
        )

        print(
            por_archivo.to_string()
        )

    # ========================================================
    # ARCHIVOS OMITIDOS
    # ========================================================

    if omitidos:

        print()
        print("=" * 70)
        print("ARCHIVOS OMITIDOS")
        print("=" * 70)

        for item in omitidos:

            print(
                f"{item['ruta']} "
                f"--> {item['motivo']}"
            )

    print()
    print("=" * 70)
    print("AUDITORÍA TERMINADA")
    print("=" * 70)