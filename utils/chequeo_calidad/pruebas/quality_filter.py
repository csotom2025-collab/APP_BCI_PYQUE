"""
quality_filter.py — Caracterizador de integridad y artefactos por canal.
Modo --report-only. Version 0.1.0.

FRONTERA DE ALCANCE (congelada por especificacion):

    ESCANEO -> INTEGRIDAD -> 3 PREPROCESAMIENTOS -> METRICAS -> PARQUET/REPORTES
    ===========================================================================
    CALIBRACION / DECISION  -> NO IMPLEMENTADO

Este modulo NO emite veredictos, NO aplica umbrales de calidad, NO calcula
scores y NO usa R para ninguna decision. Solo mide y registra.

Toda constante numerica que aparece aqui es un PARAMETRO DECLARADO, no un
umbral de calidad aprobado. Ninguna se usa para aceptar o rechazar nada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

VERSION = "0.1.0-report-only"

# ---------------------------------------------------------------------------
# PARAMETROS DECLARADOS
# Ninguno es un umbral de calidad. Ninguno decide nada.
# ---------------------------------------------------------------------------

# 128 Hz es una frecuencia de muestreo asumida/parametrizable,
# no una frecuencia verificada de esta captura.
FS_ASUMIDA = 128.0

# Escala inferida del decodificador actual; pendiente de validar
# contra el código/documentación de captura.
LSB_EMOTIV = 0.5151515151
ADC_BITS = 14
RIEL_INF_UV = 0.0
RIEL_SUP_UV = (2**14 - 1) * LSB_EMOTIV # 8439.72 uV

CANALES_ESPERADOS = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1",
                     "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]
N_CANALES_ESPERADOS = len(CANALES_ESPERADOS)

# Criterios ESTRUCTURALES de inclusion (no son juicios de calidad de senal).
N_MUESTRAS_ESPERADAS = 256
N_MUESTRAS_MIN = 240

# Bandas para potencia relativa. Parametros declarados.
BANDA_BAJA_HZ = (0.5, 4.0)
BANDA_ALTA_HZ = (20.0, 45.0)

# Barrido de z para episodios. NO es un umbral elegido: se reportan TODOS los
# valores del barrido. La eleccion de un punto de operacion es REQUIERE DECISION.
Z_BARRIDO = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

# Etapas de preprocesamiento. Las tres se conservan; ninguna sustituye a otra.
ETAPAS = ("crudo", "cent", "detr", "hp05")
ETAPAS_TRANSFORMADAS = ("cent", "detr", "hp05")

HP_FC_HZ = 0.5
HP_ORDEN = 4
HP_PADTYPE = "odd"

CARPETAS_EXCLUIDAS = ("ignorarsenales",)

# Diagnostico auxiliar de sincronizacion (R). AISLADO del pipeline de calidad.
R_N_PERM = 1000
R_SEMILLA = 31415
R_N_MIN_TRIALS = 20


# ---------------------------------------------------------------------------
# REQUIERE DECISION — puntos abiertos que este modulo NO resuelve
# ---------------------------------------------------------------------------
REQUIERE_DECISION: list[dict] = []


def _requiere_decision(clave: str, descripcion: str, resuelto_provisionalmente: str) -> None:
    """Registra un punto que exige una decision metodologica no especificada.

    El modulo no inventa la regla: aplica una alternativa puramente descriptiva
    (que no decide nada) y deja constancia de que la decision sigue abierta.
    """
    if not any(d["clave"] == clave for d in REQUIERE_DECISION):
        REQUIERE_DECISION.append({
            "clave": clave,
            "descripcion": descripcion,
            "tratamiento_provisional": resuelto_provisionalmente,
        })


# ---------------------------------------------------------------------------
# ESTRUCTURAS
# ---------------------------------------------------------------------------

@dataclass
class RegistroArchivo:
    ruta_relativa: str
    sha256: str
    usuario: str
    subcarpeta: str
    letra: str | None
    trial_nombre: int | None
    n_filas: int | None = None
    n_columnas: int | None = None
    n_canales: int | None = None
    canales_encontrados: str | None = None
    n_nan: int | None = None
    tm_monotonico: bool | None = None
    tm_inicio: float | None = None
    tm_fin: float | None = None
    tm_span_ms: float | None = None
    n_muestras_analizadas: int | None = None
    incluido: bool = False
    motivos_exclusion: list[str] = field(default_factory=list)
    session_id: int | None = None
    acquisition_order: int | None = None


# ---------------------------------------------------------------------------
# 1. ESCANEO
# ---------------------------------------------------------------------------

def sha256_archivo(ruta: str, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def _ruta_excluida(ruta_rel: str) -> bool:
    partes = ruta_rel.replace("\\", "/").lower().split("/")
    return any(p in CARPETAS_EXCLUIDAS for p in partes)


def descubrir_archivos(raiz_captures: str) -> list[str]:
    """Descubre CSV bajo `raiz_captures`, excluyendo carpetas prohibidas.

    Devuelve rutas relativas a `raiz_captures`, ORDENADAS de forma determinista
    para que el resultado no dependa del orden que entregue el sistema de
    archivos (os.walk no garantiza orden).
    """
    encontrados: list[str] = []
    for dirpath, dirnames, filenames in os.walk(raiz_captures):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, raiz_captures)
        if rel_dir == ".":
            rel_dir = ""
        if rel_dir and _ruta_excluida(rel_dir):
            dirnames[:] = []
            continue
        # Poda: no descender en carpetas excluidas
        dirnames[:] = [d for d in dirnames if d.lower() not in CARPETAS_EXCLUIDAS]
        for nombre in sorted(filenames):
            if not nombre.lower().endswith(".csv"):
                continue
            rel = os.path.join(rel_dir, nombre) if rel_dir else nombre
            if _ruta_excluida(rel):
                continue
            encontrados.append(rel.replace("\\", "/"))
    return sorted(encontrados)


def parsear_nombre(ruta_rel: str) -> tuple[str, str, str | None, int | None]:
    """usuario, subcarpeta, letra, trial_nombre a partir de la ruta relativa.

    `trial_nombre` es el sufijo numerico del archivo. NO se usa como orden de
    adquisicion (solo es correlativo dentro de cada caracter).
    """
    partes = ruta_rel.split("/")
    usuario = partes[0] if len(partes) > 1 else ""
    subcarpeta = partes[1] if len(partes) > 2 else ""
    base = os.path.splitext(partes[-1])[0]
    trozos = base.split("_")
    letra, trial = None, None
    if len(trozos) >= 3:
        letra = "_".join(trozos[1:-1])
        try:
            trial = int(trozos[-1])
        except ValueError:
            trial = None
    return usuario, subcarpeta, letra, trial


# ---------------------------------------------------------------------------
# 2. INTEGRIDAD
# ---------------------------------------------------------------------------

def leer_y_validar(ruta_abs: str, reg: RegistroArchivo) -> np.ndarray | None:
    """Lee el CSV y aplica SOLO criterios estructurales.

    No juzga la calidad de la senal. Devuelve la matriz (n_muestras, 14) en el
    orden de CANALES_ESPERADOS, o None si el archivo no es estructuralmente apto.
    """
    try:
        df = pd.read_csv(ruta_abs)
    except Exception as exc:  # archivo ilegible o vacio
        reg.motivos_exclusion.append(f"lectura_fallida:{type(exc).__name__}")
        return None

    reg.n_filas = int(df.shape[0])
    reg.n_columnas = int(df.shape[1])
    canales = [c for c in df.columns if c != "Tm"]
    reg.n_canales = len(canales)
    reg.canales_encontrados = ",".join(canales)

    if "Tm" not in df.columns:
        reg.motivos_exclusion.append("sin_columna_Tm")
    if reg.n_canales != N_CANALES_ESPERADOS:
        reg.motivos_exclusion.append(
            f"n_canales_incorrecto:{reg.n_canales}!={N_CANALES_ESPERADOS}")
    elif canales != CANALES_ESPERADOS:
        if sorted(canales) == sorted(CANALES_ESPERADOS):
            reg.motivos_exclusion.append("orden_de_canales_distinto")
        else:
            reg.motivos_exclusion.append("nombres_de_canal_distintos")
    if reg.n_filas < N_MUESTRAS_MIN:
        reg.motivos_exclusion.append(
            f"n_muestras_insuficiente:{reg.n_filas}<{N_MUESTRAS_MIN}")

    if reg.motivos_exclusion:
        return None

    tm = pd.to_numeric(df["Tm"], errors="coerce").to_numpy(dtype=np.float64)
    X = df[CANALES_ESPERADOS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    n_nan = int(np.isnan(X).sum() + np.isnan(tm).sum())
    reg.n_nan = n_nan
    if n_nan > 0:
        reg.motivos_exclusion.append(f"contiene_nan:{n_nan}")

    if np.isnan(tm).any():
        reg.tm_monotonico = None
    else:
        reg.tm_monotonico = bool(np.all(np.diff(tm) > 0))
        reg.tm_inicio = float(tm[0])
        reg.tm_fin = float(tm[-1])
        reg.tm_span_ms = float(tm[-1] - tm[0])
        if not reg.tm_monotonico:
            reg.motivos_exclusion.append("Tm_no_monotonico")

    if reg.motivos_exclusion:
        return None

    # Recorte estructural a la longitud nominal, para que todas las metricas se
    # calculen sobre ventanas de la misma longitud. Se registra si hubo recorte.
    if X.shape[0] > N_MUESTRAS_ESPERADAS:
        X = X[:N_MUESTRAS_ESPERADAS, :]
    elif X.shape[0] < N_MUESTRAS_ESPERADAS:
        _requiere_decision(
            "longitud_no_nominal",
            "Archivos con entre N_MUESTRAS_MIN y N_MUESTRAS_ESPERADAS muestras: "
            "no esta especificado si deben rellenarse, recortarse a la longitud "
            "minima comun, o analizarse con su longitud propia.",
            "Se analizan con su longitud propia y se registra n_filas.")

    reg.incluido = True
    return X


# ---------------------------------------------------------------------------
# 1b. ORDEN DETERMINISTA
# ---------------------------------------------------------------------------

def asignar_orden(registros: list[RegistroArchivo]) -> dict:
    """Asigna acquisition_order por usuario. NO asigna session_id.

    `acquisition_order` es el rango por Tm[0] ascendente dentro del usuario, con
    desempate por ruta relativa. Es determinista y no necesita ningun orden de
    arranque, asi que no depende del sistema de archivos ni de los nombres.

    `session_id` se deja NULO deliberadamente. Los CSV no contienen identificador
    de sesion y reconstruirlo desde Tm exige un orden de arranque arbitrario: con
    un arranque por nombre de archivo, cada cambio de caracter produce un falso
    descenso de Tm y se detectan decenas de "sesiones" inexistentes. Emitir ese
    numero seria inventar una regla, asi que se detiene esa parte.

    Devuelve un diagnostico que cuantifica la ambiguedad, sin resolverla.
    """
    _requiere_decision(
        "reconstruccion_de_sesiones",
        "Los CSV no contienen identificador de sesion. Tm es un reloj de host "
        "que se reinicia con la aplicacion, y reconstruir sesiones a partir de "
        "el exige fijar un orden de arranque que los datos no determinan: "
        "ordenando por nombre, cada cambio de caracter simula un reinicio.",
        "session_id se emite NULO. Se emite acquisition_order (rango por Tm[0] "
        "dentro del usuario), que si es determinista. Se reporta el diagnostico "
        "de ambiguedad por usuario.")

    por_usuario: dict[str, list[RegistroArchivo]] = {}
    for r in registros:
        if r.incluido and r.tm_inicio is not None:
            por_usuario.setdefault(r.usuario, []).append(r)

    diagnostico: dict[str, dict] = {}
    for usuario, lista in por_usuario.items():
        ordenados = sorted(lista, key=lambda r: (r.tm_inicio, r.ruta_relativa))
        for i, r in enumerate(ordenados):
            r.acquisition_order = i
            r.session_id = None

        # Diagnostico de ambiguedad: cuantos descensos de Tm[0] aparecerian bajo
        # un arranque por nombre de archivo. Es evidencia de por que no se puede
        # reconstruir la sesion, no una estimacion del numero de sesiones.
        por_nombre = sorted(lista, key=lambda r: (r.subcarpeta, r.letra or "",
                                                  r.trial_nombre if r.trial_nombre is not None else -1,
                                                  r.ruta_relativa))
        descensos = sum(1 for a, b in zip(por_nombre, por_nombre[1:])
                        if b.tm_inicio < a.tm_inicio)
        tms = [r.tm_inicio for r in ordenados]
        diagnostico[usuario] = {
            "n_archivos": len(lista),
            "tm_inicio_min_ms": float(min(tms)),
            "tm_inicio_max_ms": float(max(tms)),
            "descensos_de_Tm_bajo_arranque_por_nombre": int(descensos),
            "nota": "Los descensos NO son un conteo de sesiones: bajo un arranque "
                    "por nombre, cada cambio de caracter produce descensos espurios.",
        }
    return diagnostico


# ---------------------------------------------------------------------------
# 3. PREPROCESAMIENTOS (las tres se conservan; ninguna sustituye a otra)
# ---------------------------------------------------------------------------

_HP_SOS = sp_signal.butter(HP_ORDEN, HP_FC_HZ / (FS_ASUMIDA / 2.0),
                           btype="highpass", output="ba")


def aplicar_etapa(X: np.ndarray, etapa: str) -> np.ndarray:
    """X: (n_muestras, n_canales). NUNCA aplica re-referencia (CAR/Laplaciano).

    Cada canal se procesa de forma independiente: un canal no puede influir en
    las metricas de otro.
    """
    if etapa == "crudo":
        return X
    if etapa == "cent":
        return X - X.mean(axis=0, keepdims=True)
    if etapa == "detr":
        return sp_signal.detrend(X, axis=0, type="linear")
    if etapa == "hp05":
        b, a = _HP_SOS
        Xc = X - X.mean(axis=0, keepdims=True)
        return sp_signal.filtfilt(b, a, Xc, axis=0, padtype=HP_PADTYPE)
    raise ValueError(f"etapa desconocida: {etapa}")


# ---------------------------------------------------------------------------
# 4. METRICAS
# ---------------------------------------------------------------------------

MAD_PISO = LSB_EMOTIV / 1.4826  # piso fisico: la escala no puede ser menor que
                                # la resolucion del convertidor.


def _rachas_identicas(col: np.ndarray) -> int:
    """Longitud de la racha maxima de muestras consecutivas identicas."""
    if col.size < 2:
        return int(col.size)
    iguales = np.diff(col) == 0
    if not iguales.any():
        return 1
    mejor = actual = 1
    for v in iguales:
        if v:
            actual += 1
            if actual > mejor:
                mejor = actual
        else:
            actual = 1
    return mejor


def metricas_crudas(X: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Metricas que solo tienen sentido sobre la senal SIN transformar."""
    n, c = X.shape
    t = np.arange(n) / fs
    # pendiente por canal (ajuste lineal); polyfit vectorizado
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    pendiente = coef[0, :]

    run_max = np.array([_rachas_identicas(X[:, j]) for j in range(c)], dtype=float)
    difs = np.diff(X, axis=0)
    pct_rep = (difs == 0).mean(axis=0)
    n_niveles = np.array([np.unique(X[:, j]).size for j in range(c)], dtype=float)

    return {
        "dc_medio": X.mean(axis=0),
        "dc_mediana": np.median(X, axis=0),
        "val_min": X.min(axis=0),
        "val_max": X.max(axis=0),
        "dist_riel_inf": X.min(axis=0) - RIEL_INF_UV,
        "dist_riel_sup": RIEL_SUP_UV - X.max(axis=0),
        "pendiente_uV_s": pendiente,
        "run_max_identicos": run_max,
        "pct_repetidos": pct_rep,
        "n_niveles_unicos": n_niveles,
    }


def metricas_transformadas(Y: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Metricas sobre una senal ya transformada. Por canal, independientes."""
    n, c = Y.shape
    mediana = np.median(Y, axis=0)
    mad_bruto = np.median(np.abs(Y - mediana), axis=0)
    piso_activado = (mad_bruto < MAD_PISO).astype(float)
    mad = np.maximum(mad_bruto, MAD_PISO)

    z = np.abs(Y - mediana) / (1.4826 * mad)
    max_z = z.max(axis=0)

    p2p = Y.max(axis=0) - Y.min(axis=0)

    # Espectro (ventana de Hann). fs es fs_asumida: parametro declarado.
    win = np.hanning(n)[:, None]
    P = np.abs(np.fft.rfft(Y * win, axis=0)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    total = P.sum(axis=0)
    total_seg = np.where(total > 0, total, np.nan)
    m_baja = (fr >= BANDA_BAJA_HZ[0]) & (fr < BANDA_BAJA_HZ[1])
    m_alta = (fr >= BANDA_ALTA_HZ[0]) & (fr < BANDA_ALTA_HZ[1])

    # Correlacion entre canales: se CALCULA, no se aplica como re-referencia.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Y.T)
    C = np.asarray(C, dtype=float)
    if C.ndim == 0:
        C = np.array([[1.0]])
    np.fill_diagonal(C, np.nan)
    # Un canal constante no tiene correlacion definida: su fila queda toda NaN y
    # corr_mediana_otros debe ser NaN. Se silencia el aviso porque el NaN es el
    # resultado correcto, no un error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr_med = np.nanmedian(C, axis=1)

    out = {
        "mad": mad,
        "mad_bruto": mad_bruto,
        "mad_piso_activado": piso_activado,
        "max_z_robusto": max_z,
        "p2p": p2p,
        "pot_total": total,
        "pot_rel_baja": P[m_baja].sum(axis=0) / total_seg,
        "pot_rel_alta": P[m_alta].sum(axis=0) / total_seg,
        "corr_mediana_otros": corr_med,
    }

    # Episodios: barrido completo, sin elegir punto de operacion.
    for zt in Z_BARRIDO:
        sobre = z > zt
        n_epis = np.zeros(c)
        dur_max = np.zeros(c)
        for j in range(c):
            s = sobre[:, j]
            if not s.any():
                continue
            bordes = np.diff(np.concatenate(([0], s.view(np.int8), [0])))
            ini = np.where(bordes == 1)[0]
            fin = np.where(bordes == -1)[0]
            n_epis[j] = len(ini)
            dur_max[j] = (fin - ini).max() / fs * 1000.0
        clave = f"{zt:g}".replace(".", "p")
        out[f"n_episodios_z{clave}"] = n_epis
        out[f"dur_max_episodio_ms_z{clave}"] = dur_max

    return out


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL (etapas 1-4)
# ---------------------------------------------------------------------------

def procesar(raiz_captures: str, dir_salida: str, fs_asumida: float = FS_ASUMIDA,
             limite: int | None = None, verbose: bool = True) -> tuple[pd.DataFrame, list[RegistroArchivo], dict]:
    rutas = descubrir_archivos(raiz_captures)
    if limite is not None:
        rutas = rutas[:limite]
    if verbose:
        print(f"[escaneo] {len(rutas)} CSV candidatos (ignorarSenales excluida)")

    registros: list[RegistroArchivo] = []
    matrices: dict[str, np.ndarray] = {}

    for k, rel in enumerate(rutas):
        absruta = os.path.join(raiz_captures, rel)
        usuario, subcarpeta, letra, trial = parsear_nombre(rel)
        reg = RegistroArchivo(ruta_relativa=rel, sha256=sha256_archivo(absruta),
                              usuario=usuario, subcarpeta=subcarpeta,
                              letra=letra, trial_nombre=trial)
        X = leer_y_validar(absruta, reg)
        if X is not None:
            matrices[rel] = X
        registros.append(reg)
        if verbose and (k + 1) % 500 == 0:
            print(f"[integridad] {k + 1}/{len(rutas)}")

    diag_orden = asignar_orden(registros)

    filas: list[dict] = []
    for reg in registros:
        if not reg.incluido:
            continue
        X = matrices[reg.ruta_relativa]
        base = {
            "ruta_relativa": reg.ruta_relativa,
            "sha256": reg.sha256,
            "usuario": reg.usuario,
            "subcarpeta": reg.subcarpeta,
            "letra": reg.letra,
            "trial_nombre": reg.trial_nombre,
            "session_id": reg.session_id,
            "acquisition_order": reg.acquisition_order,
            "n_filas": reg.n_filas,
            "fs_asumida": fs_asumida,
            "version_pipeline": VERSION,
        }
        for etapa in ETAPAS:
            Y = aplicar_etapa(X, etapa)
            met = metricas_crudas(X, fs_asumida) if etapa == "crudo" \
                else metricas_transformadas(Y, fs_asumida)
            for j, canal in enumerate(CANALES_ESPERADOS):
                fila = dict(base)
                fila["canal"] = canal
                fila["etapa"] = etapa
                for nombre, vec in met.items():
                    fila[nombre] = float(vec[j])
                filas.append(fila)

    df = pd.DataFrame(filas)
    os.makedirs(dir_salida, exist_ok=True)
    if verbose:
        print(f"[metricas] {len(df)} filas = "
              f"{df['ruta_relativa'].nunique() if len(df) else 0} archivos "
              f"x {N_CANALES_ESPERADOS} canales x {len(ETAPAS)} etapas")
    return df, registros, diag_orden


# ---------------------------------------------------------------------------
# DIAGNOSTICO AUXILIAR DE SINCRONIZACION (R)
# AISLADO: no toca metricas, no excluye nada, no clasifica nada.
# ---------------------------------------------------------------------------

def _R(M: np.ndarray) -> float:
    """M: (n_trials, n_muestras, n_canales)."""
    n = M.shape[0]
    return float(M.mean(axis=0).var(axis=0).mean() / (M.var(axis=1).mean() / n))


def _permutar_circular(M: np.ndarray, desplaz: np.ndarray) -> np.ndarray:
    """Desplazamiento circular independiente por trial, IDENTICO para sus canales."""
    n, T, _ = M.shape
    idx = (np.arange(T)[None, :] - desplaz[:, None]) % T
    return M[np.arange(n)[:, None], idx]


def diagnostico_sincronizacion(raiz_captures: str, registros: list[RegistroArchivo],
                               fs_asumida: float = FS_ASUMIDA,
                               n_perm: int = R_N_PERM, semilla: int = R_SEMILLA,
                               verbose: bool = True) -> dict:
    _requiere_decision(
        "agrupacion_para_R",
        "No esta especificado si R debe calcularse por (usuario, subcarpeta), "
        "por usuario con todas las subcarpetas juntas, o de ambas formas.",
        "Se calculan y reportan AMBAS agrupaciones, sin elegir una.")

    incluidos = [r for r in registros if r.incluido and r.acquisition_order is not None]
    grupos: dict[tuple[str, str], list[RegistroArchivo]] = {}
    for r in incluidos:
        grupos.setdefault((r.usuario, r.subcarpeta), []).append(r)
        grupos.setdefault((r.usuario, "__TODAS__"), []).append(r)

    rng = np.random.default_rng(semilla)
    resultados = []
    for (usuario, sub), lista in sorted(grupos.items()):
        lista = sorted(lista, key=lambda r: r.acquisition_order)
        lista = [r for r in lista if r.n_filas >= N_MUESTRAS_ESPERADAS]
        if len(lista) < R_N_MIN_TRIALS:
            continue
        X = np.stack([
            pd.read_csv(os.path.join(raiz_captures, r.ruta_relativa))[CANALES_ESPERADOS]
              .to_numpy(dtype=np.float64)[:N_MUESTRAS_ESPERADAS, :]
            for r in lista])
        n = X.shape[0]
        for etapa in ETAPAS_TRANSFORMADAS:
            Y = np.stack([aplicar_etapa(X[i], etapa) for i in range(n)])
            r_obs = _R(Y)
            nulos = np.empty(n_perm)
            for p in range(n_perm):
                nulos[p] = _R(_permutar_circular(Y, rng.integers(0, Y.shape[1], size=n)))
            k = int((nulos >= r_obs).sum())
            p_val = (1 + k) / (n_perm + 1)
            resultados.append({
                "usuario": usuario,
                "subcarpeta": sub,
                "etapa": etapa,
                "n_trials": n,
                "R_observado": r_obs,
                "nula_media": float(nulos.mean()),
                "nula_sd": float(nulos.std()),
                "nula_p95": float(np.percentile(nulos, 95)),
                "p_valor": p_val,
                "n_permutaciones": n_perm,
            })
        if verbose:
            print(f"[R] {usuario}/{sub}: n={n}")

    # Correcciones por multiplicidad, POR etapa y POR agrupacion (familias separadas)
    for sub_filtro in {r["subcarpeta"] for r in resultados}:
        for etapa in ETAPAS_TRANSFORMADAS:
            fam = [r for r in resultados
                   if r["subcarpeta"] == sub_filtro and r["etapa"] == etapa]
            if not fam:
                continue
            kfam = len(fam)
            ps = np.array([r["p_valor"] for r in fam])
            orden = np.argsort(ps)
            bh = np.empty(kfam)
            acc = 1.0
            for rank in range(kfam - 1, -1, -1):
                i = orden[rank]
                acc = min(acc, ps[i] * kfam / (rank + 1))
                bh[i] = acc
            for i, r in enumerate(fam):
                r["k_familia"] = kfam
                r["umbral_bonferroni"] = 0.05 / kfam
                r["significativo_bonferroni"] = bool(ps[i] < 0.05 / kfam)
                r["p_benjamini_hochberg"] = float(bh[i])
                r["significativo_bh_005"] = bool(bh[i] < 0.05)

    return {
        "aviso": ("DIAGNOSTICO EXPERIMENTAL / AUXILIAR. R no esta validado como "
                  "metrica. No se usa para excluir, rechazar, clasificar ni "
                  "modificar ninguna metrica de calidad. R no es comparable "
                  "entre grupos con distinto n_trials: solo lo son los p-valores."),
        "protocolo": {
            "seleccion": "todos los trials estructuralmente validos del grupo",
            "orden": "acquisition_order (rango por Tm[0] dentro del usuario)",
            "etapas": list(ETAPAS_TRANSFORMADAS),
            "n_permutaciones": n_perm,
            "semilla": semilla,
            "metodo_permutacion": "desplazamiento circular (indices mod T)",
            "desplazamiento": "independiente por trial, identico para sus 14 canales",
            "p_valor": "(1 + k) / (n_perm + 1)",
            "correccion_multiplicidad": "Bonferroni y Benjamini-Hochberg por familia",
        },
        "resultados": resultados,
    }


# ---------------------------------------------------------------------------
# REPORTES
# ---------------------------------------------------------------------------

def _versiones_dependencias() -> dict:
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "pyarrow"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    return vers


def escribir_salidas(df: pd.DataFrame, registros: list[RegistroArchivo],
                     dir_salida: str, raiz_captures: str, fs_asumida: float,
                     diag_R: dict | None, diag_orden: dict | None = None) -> dict:
    diag_orden = diag_orden or {}
    os.makedirs(dir_salida, exist_ok=True)

    ruta_parquet = os.path.join(dir_salida, "metricas_calidad.parquet")
    df.to_parquet(ruta_parquet, index=False)

    # reporte.csv: una fila por archivo. SIN veredicto, SIN score.
    filas = []
    for r in registros:
        d = asdict(r)
        d["motivos_exclusion"] = "|".join(r.motivos_exclusion)
        filas.append(d)
    df_arch = pd.DataFrame(filas)
    # erp_readiness: existe, siempre nulo, con motivo.
    df_arch["erp_readiness"] = pd.Series([None] * len(df_arch), dtype="object")
    df_arch["erp_readiness_motivo"] = "sin_trigger"
    ruta_csv = os.path.join(dir_salida, "reporte.csv")
    df_arch.to_csv(ruta_csv, index=False, encoding="utf-8")

    incluidos = [r for r in registros if r.incluido]
    motivos: dict[str, int] = {}
    for r in registros:
        for m in r.motivos_exclusion:
            clave = m.split(":")[0]
            motivos[clave] = motivos.get(clave, 0) + 1


    reporte = {
        "version_pipeline": VERSION,
        "modo": "report-only",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "raiz_captures": os.path.abspath(raiz_captures),
        "dir_salida": os.path.abspath(dir_salida),
        "carpetas_excluidas": list(CARPETAS_EXCLUIDAS),
        "orden_utilizado": {
            "criterio": "acquisition_order",
            "acquisition_order": "rango por Tm[0] ascendente dentro del usuario; "
                                 "desempate por ruta relativa",
            "session_id": "NULO — no determinable desde los CSV (ver requiere_decision)",
            "nota": "Tm es un reloj de host, no del dispositivo. Se usa SOLO para "
                    "ordenar y como verificacion de integridad, nunca como eje temporal.",
        },
        "parametros": {
            "fs_asumida": fs_asumida,
            "fs_asumida_estado": "NOMINAL / PROVISIONAL — no verificada. "
                                 "REQUIERE VERIFICACION DOCUMENTAL.",
            "canales_esperados": CANALES_ESPERADOS,
            "n_canales": N_CANALES_ESPERADOS,
            "n_muestras_esperadas": N_MUESTRAS_ESPERADAS,
            "n_muestras_min_estructural": N_MUESTRAS_MIN,
            "etapas_preprocesamiento": list(ETAPAS),
            "hp": {"fc_hz": HP_FC_HZ, "orden": HP_ORDEN, "padtype": HP_PADTYPE},
            "banda_baja_hz": list(BANDA_BAJA_HZ),
            "banda_alta_hz": list(BANDA_ALTA_HZ),
            "z_barrido_episodios": list(Z_BARRIDO),
            "lsb_uV": LSB_EMOTIV,
            "lsb_estado": "del codigo del capturador; escala fisica REQUIERE "
                          "VERIFICACION DOCUMENTAL",
            "mad_piso": MAD_PISO,
            "riel_inf_uV": RIEL_INF_UV,
            "riel_sup_uV": RIEL_SUP_UV,
            "re_referencia_aplicada": False,
        },
        "conteos": {
            "archivos_descubiertos": len(registros),
            "archivos_incluidos": len(incluidos),
            "archivos_excluidos": len(registros) - len(incluidos),
            "usuarios": sorted({r.usuario for r in incluidos}),
            "n_usuarios": len({r.usuario for r in incluidos}),
            "session_id": None,
            "session_id_motivo": "no determinable desde los CSV (ver requiere_decision)",
            "diagnostico_de_orden": diag_orden,
            "motivos_exclusion": motivos,
            "filas_parquet": int(len(df)),
        },
        "erp_readiness": {"value": None, "reason": "sin_trigger"},
        "no_implementado": ["calibracion_de_umbrales", "capa_de_decision",
                            "veredictos", "scores"],
        "requiere_decision": REQUIERE_DECISION,
        "semilla_diagnostico_R": R_SEMILLA,
        "dependencias": _versiones_dependencias(),
        "hashes": {r.ruta_relativa: r.sha256 for r in registros},
    }
    with open(os.path.join(dir_salida, "reporte.json"), "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    if diag_R is not None:
        with open(os.path.join(dir_salida, "diagnostico_sincronizacion.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(diag_R, fh, indent=2, ensure_ascii=False)

    return reporte


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Caracterizador de integridad y artefactos EEG (report-only).")
    ap.add_argument("--captures", default="captures", help="directorio de entrada")
    ap.add_argument("--out", default="resultados_calidad", help="directorio de salida")
    ap.add_argument("--report-only", action="store_true", required=True,
                    help="unico modo implementado; obligatorio")
    ap.add_argument("--fs-asumida", type=float, default=FS_ASUMIDA)
    ap.add_argument("--limite", type=int, default=None, help="solo para pruebas")
    ap.add_argument("--con-diagnostico-R", action="store_true",
                    help="calcula el diagnostico auxiliar de sincronizacion")
    ap.add_argument("--n-perm", type=int, default=R_N_PERM)
    ap.add_argument("--silencioso", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.silencioso
    salida = os.path.abspath(args.out)
    entrada = os.path.abspath(args.captures)
    if salida == entrada or salida.startswith(entrada + os.sep):
        raise SystemExit("ERROR: el directorio de salida no puede estar dentro "
                         "del directorio de entrada (modo report-only).")

    df, registros, diag_orden = procesar(entrada, salida, args.fs_asumida, args.limite, verbose)

    diag = None
    if args.con_diagnostico_R:
        diag = diagnostico_sincronizacion(entrada, registros, args.fs_asumida,
                                          args.n_perm, R_SEMILLA, verbose)

    rep = escribir_salidas(df, registros, salida, entrada, args.fs_asumida, diag, diag_orden)

    if verbose:
        print("\n--- RESUMEN ---")
        print(f"descubiertos : {rep['conteos']['archivos_descubiertos']}")
        print(f"incluidos    : {rep['conteos']['archivos_incluidos']}")
        print(f"excluidos    : {rep['conteos']['archivos_excluidos']}")
        for m, n in sorted(rep["conteos"]["motivos_exclusion"].items()):
            print(f"   {m}: {n}")
        print(f"filas parquet: {rep['conteos']['filas_parquet']}")
        print(f"REQUIERE DECISION: {len(REQUIERE_DECISION)}")
        for d in REQUIERE_DECISION:
            print(f"   - {d['clave']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
