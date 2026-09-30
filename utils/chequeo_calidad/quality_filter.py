#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_filter.py  --  Filtro de calidad de capturas EEG (BCI speller).  v1.0.0

Clasifica CADA archivo de captura en uno de tres estados:

    A  UTILIZABLE     Ningun nodo presenta anomalias. El archivo se copia
                      byte a byte, sin ninguna modificacion.
    B  MODIFICADO     Uno o mas nodos (canales) presentaron una anomalia
                      transitoria (pico, salto, tramo plano, saturacion...) o
                      quedaron inservibles, pero quedan suficientes nodos sanos.
                      SOLO se recortan los tramos anomalos de ESE nodo; los
                      demas nodos y la columna de tiempo quedan identicos.
    C  A REGRABAR     El archivo no sirve: estructura invalida, muestras
                      insuficientes, etiqueta invalida, duplicado exacto, o
                      demasiados nodos fuera de norma. Hay que volver a
                      grabar ese simbolo de ese usuario (p. ej. "el 9 de Jorge").

Que significa "recortar un tramo"
---------------------------------
Las muestras anomalas de ese nodo se dejan VACIAS (NaN) en el CSV de salida.
No se borran filas: asi todos los nodos conservan la misma longitud y el mismo
eje temporal, y el resto del archivo sigue exactamente igual. Cualquier
calculo posterior (Pearson, etc.) debe usar correlacion "pairwise complete"
(pandas .corr() ya lo hace; np.corrcoef NO).

Criterios (todos son parametros declarados, ver seccion PARAMETROS)
-------------------------------------------------------------------
Nivel archivo  -> C:  lectura fallida, falta columna de tiempo, canales
                      distintos de los 14 esperados, < N_MUESTRAS_MIN filas,
                      tiempo no monotono, etiqueta/carpeta invalida, duplicado
                      exacto (sha256), o nodos utiles < MIN_NODOS_UTILES.
Nivel nodo     -> DESCARTADO (todo el nodo a NaN): DC fuera de rango (electrodo
                      desconectado), nodo plano/casi constante, ruido de fondo
                      excesivo, o mas de MAX_FRAC_RECORTE_NODO de sus muestras
                      resultaron anomalas.
Nivel muestra  -> RECORTADO: (1) transitorio: |x - mediana movil| supera
                      max(A_SEG_UV, Z_SEG * sigma_robusta); (2) desvio grueso
                      respecto a la mediana del nodo > A_DERIVA_UV;
                      (3) racha de valores identicos >= RUN_PLANO_MIN;
                      (4) muestra en el riel del ADC; (5) valor no numerico.
                      Cada tramo se extiende por histeresis, se le agrega un
                      margen y los tramos cercanos se fusionan.

El directorio de salida NO puede estar dentro del de entrada; los originales
nunca se tocan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import ndimage

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
# Estructura del dispositivo / captura (Emotiv EPOC de 14 canales)
FS = 128.0                          # Hz nominal
N_MUESTRAS_ESPERADAS = 256          # 2 s
N_MUESTRAS_MIN = 256                # < esto -> C (registros insuficientes)
CANALES = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1",
           "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]
ALIAS_TIEMPO = ("Tm", "Time", "time", "TIME")

LSB_UV = 0.5151515151               # escala heredada del capturador
RIEL_SUP_UV = (2 ** 14 - 1) * LSB_UV
RIEL_MARGEN_UV = 5 * LSB_UV         # muestra a <= 5 LSB del riel = saturada

# Nivel de continua (DC) tipico por nodo: mediana de los 6867 archivos ya
# validados del proyecto. Un nodo cuyo DC se aleja mas de DC_TOL_UV de su
# referencia suele ser un electrodo desconectado (DC observados: 646, 711,
# 1263, 1809, 2402 uV...).
DC_REF_UV = {"AF3": 4156.2, "AF4": 4235.1, "F3": 4269.3, "F4": 4134.1,
             "F7": 4416.1, "F8": 4436.5, "FC5": 4227.1, "FC6": 4267.5,
             "O1": 4390.1, "O2": 4301.0, "P7": 4541.6, "P8": 4486.5,
             "T7": 4252.1, "T8": 3758.8}
DC_TOL_UV = 600.0

# Deteccion de tramos anomalos (nivel muestra)
VENTANA_BASE = 97                   # mediana movil, ~0.76 s (impar)
A_SEG_UV = 120.0                    # piso absoluto (nivel bajo de las auditorias)
Z_SEG = 8.0                         # multiplo de sigma robusta
HISTERESIS = 0.5                    # el tramo crece mientras |d| > 0.5*umbral
MARGEN_MUESTRAS = 3                 # ~23 ms a cada lado
FUSION_HUECO = 6                    # une tramos separados <= 6 muestras (~47 ms)
A_DERIVA_UV = 300.0                 # desvio grueso lento respecto a la mediana
RUN_PLANO_MIN = 26                  # ~200 ms de valores identicos

# Nivel nodo
RUN_PLANO_NODO = 128                # >= 1 s identico -> nodo muerto
NIVELES_MIN_NODO = 8                # <= 8 valores distintos -> nodo muerto
SIGMA_NODO_MAX_UV = 100.0           # sigma robusta de fondo -> nodo ruidoso
MAX_FRAC_RECORTE_NODO = 0.50        # > 50 % recortado -> nodo descartado

# Nivel archivo
MIN_NODOS_UTILES = 7                # < 7 de 14 nodos utiles -> C
TRIALS_OBJETIVO_DEFECTO = 30        # trials por (usuario, clase) en el protocolo

CARPETAS_EXCLUIDAS = ("ignorarsenales",)
LETRAS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Ñ"]
DIGITOS = list("0123456789")
COMANDOS = ["↩", "───", "⟵"]
CLASES_POR_CARPETA = {"letters": set(LETRAS), "numbers": set(DIGITOS),
                      "controls": set(COMANDOS)}

ESTADO_A, ESTADO_B, ESTADO_C = "A", "B", "C"


# ---------------------------------------------------------------------------
# ESTRUCTURAS
# ---------------------------------------------------------------------------
@dataclass
class Segmento:
    canal: str
    inicio: int          # indice de muestra, inclusive
    fin: int             # indice de muestra, inclusive
    causas: list

    @property
    def n(self) -> int:
        return self.fin - self.inicio + 1


@dataclass
class ResultadoNodo:
    canal: str
    estado: str = "OK"                    # OK | RECORTADO | DESCARTADO
    motivos: list = field(default_factory=list)
    segmentos: list = field(default_factory=list)
    n_recortadas: int = 0
    frac_recortada: float = 0.0
    dc_mediana: float | None = None
    sigma_robusta: float | None = None


@dataclass
class ResultadoArchivo:
    ruta_relativa: str
    usuario: str
    subcarpeta: str
    clase: str | None
    trial: int | None
    sha256: str = ""
    n_filas: int | None = None
    estado: str = ESTADO_C
    motivos: list = field(default_factory=list)
    nodos: list = field(default_factory=list)
    n_nodos_utiles: int | None = None
    n_muestras_recortadas: int = 0
    columna_tiempo: str | None = None


# ---------------------------------------------------------------------------
# ESCANEO Y NOMBRES
# ---------------------------------------------------------------------------
def sha256_archivo(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def descubrir_archivos(raiz: str) -> list[str]:
    """CSV bajo `raiz` (rutas relativas, '/' , orden determinista)."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(raiz):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in CARPETAS_EXCLUIDAS)
        rel_dir = os.path.relpath(dirpath, raiz)
        rel_dir = "" if rel_dir == "." else rel_dir
        for nombre in sorted(filenames):
            if nombre.lower().endswith(".csv"):
                out.append(os.path.join(rel_dir, nombre).replace("\\", "/"))
    return sorted(out)


def parsear_nombre(rel: str):
    """usuario, subcarpeta, clase, trial  desde  Usuario/Sub/Usuario_CLASE_N.csv"""
    partes = rel.split("/")
    usuario = partes[0] if len(partes) > 1 else ""
    subcarpeta = partes[1] if len(partes) > 2 else ""
    trozos = os.path.splitext(partes[-1])[0].split("_")
    clase, trial = None, None
    if len(trozos) >= 3:
        clase = "_".join(trozos[1:-1])
        try:
            trial = int(trozos[-1])
        except ValueError:
            trial = None
    return usuario, subcarpeta, clase, trial


def validar_etiqueta(subcarpeta: str, clase: str | None) -> str | None:
    """Devuelve el motivo de rechazo o None si la etiqueta es coherente."""
    todas = set().union(*CLASES_POR_CARPETA.values())
    if clase is None or clase not in todas:
        return f"etiqueta_invalida:{clase}"
    esperadas = CLASES_POR_CARPETA.get(subcarpeta.lower())
    if esperadas is None:
        return f"carpeta_desconocida:{subcarpeta}"
    if clase not in esperadas:
        return f"clase_incoherente_con_carpeta:{clase}@{subcarpeta}"
    return None


# ---------------------------------------------------------------------------
# DETECCION DE ANOMALIAS EN UN NODO
# ---------------------------------------------------------------------------
def _rachas(mask: np.ndarray):
    """(inicio, fin_inclusive) de cada racha contigua de True."""
    if not mask.any():
        return []
    b = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    ini, fin = np.where(b == 1)[0], np.where(b == -1)[0] - 1
    return list(zip(ini.tolist(), fin.tolist()))


def _rachas_identicas(x: np.ndarray, minimo: int) -> np.ndarray:
    """Marca las muestras que pertenecen a una racha de valores identicos >= minimo."""
    marca = np.zeros(x.size, dtype=bool)
    if x.size < 2:
        return marca
    iguales = np.concatenate(([False], np.diff(x) == 0))   # x[i]==x[i-1]
    inicio = None
    for i in range(x.size):
        if iguales[i]:
            if inicio is None:
                inicio = i - 1
        else:
            if inicio is not None and (i - inicio) >= minimo:
                marca[inicio:i] = True
            inicio = None
    if inicio is not None and (x.size - inicio) >= minimo:
        marca[inicio:] = True
    return marca


def _cerrar(mask: np.ndarray, margen: int, hueco: int) -> np.ndarray:
    """Agrega margen a cada tramo y fusiona tramos separados <= hueco muestras."""
    if not mask.any():
        return mask
    out = mask.copy()
    if hueco > 0:
        for (i0, f0), (i1, _) in zip(_rachas(mask), _rachas(mask)[1:]):
            if i1 - f0 - 1 <= hueco:
                out[f0:i1] = True
    if margen > 0:
        out = ndimage.binary_dilation(out, structure=np.ones(2 * margen + 1, dtype=bool))
    return out


def analizar_nodo(x: np.ndarray, canal: str) -> tuple[ResultadoNodo, np.ndarray]:
    """Analiza un nodo. Devuelve (resultado, mascara_de_muestras_a_recortar).

    `x` puede contener NaN (valores no numericos del CSV original).
    """
    n = x.size
    res = ResultadoNodo(canal=canal)
    nan_mask = ~np.isfinite(x)
    validos = x[~nan_mask]

    # --- nodo inservible de entrada -------------------------------------
    if validos.size < max(8, int(0.5 * n)):
        res.estado, res.motivos = "DESCARTADO", ["menos_de_50pct_de_muestras_numericas"]
        res.n_recortadas, res.frac_recortada = n, 1.0
        return res, np.ones(n, dtype=bool)

    med = float(np.median(validos))
    res.dc_mediana = med
    ref = DC_REF_UV.get(canal)
    motivos_nodo: list[str] = []
    if ref is not None and abs(med - ref) > DC_TOL_UV:
        motivos_nodo.append(f"dc_fuera_de_rango:{med:.0f}uV(ref {ref:.0f}+-{DC_TOL_UV:.0f})")
    if np.unique(validos).size <= NIVELES_MIN_NODO:
        motivos_nodo.append(f"nodo_casi_constante:{np.unique(validos).size}_niveles")
    if _rachas_identicas(np.where(nan_mask, np.nan, x), RUN_PLANO_NODO).any():
        motivos_nodo.append(f"racha_plana>={RUN_PLANO_NODO}_muestras")

    # --- linea base robusta y umbral ----------------------------------------
    xi = np.where(nan_mask, med, x)                          # NaN -> mediana (solo para filtrar)
    base = ndimage.median_filter(xi, size=min(VENTANA_BASE, n | 1), mode="reflect")
    d = xi - base
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = max(1.4826 * mad, LSB_UV)
    res.sigma_robusta = sigma
    if sigma > SIGMA_NODO_MAX_UV:
        motivos_nodo.append(f"ruido_de_fondo_excesivo:sigma={sigma:.0f}uV")

    if motivos_nodo:
        res.estado, res.motivos = "DESCARTADO", motivos_nodo
        res.n_recortadas, res.frac_recortada = n, 1.0
        return res, np.ones(n, dtype=bool)

    thr = max(A_SEG_UV, Z_SEG * sigma)
    absd = np.abs(d)

    # (1) transitorios con histeresis
    core = absd > thr
    transit = np.zeros(n, dtype=bool)
    if core.any():
        etiquetas, k = ndimage.label(absd > HISTERESIS * thr)
        for lab in np.unique(etiquetas[core]):
            transit |= etiquetas == lab
    # (2) desvio grueso lento
    deriva = np.abs(xi - med) > A_DERIVA_UV
    # (3) tramo plano
    plano = _rachas_identicas(np.where(nan_mask, np.nan, x), RUN_PLANO_MIN)
    # (4) saturacion en el riel
    riel = (x >= RIEL_SUP_UV - RIEL_MARGEN_UV) | (x <= RIEL_MARGEN_UV)
    riel &= ~nan_mask

    causas = {"transitorio": transit, "deriva": deriva, "plano": plano,
              "riel": riel, "no_numerico": nan_mask}
    crudo = np.zeros(n, dtype=bool)
    for m in causas.values():
        crudo |= m
    mascara = _cerrar(crudo, MARGEN_MUESTRAS, FUSION_HUECO)

    for ini, fin in _rachas(mascara):
        cs = [nom for nom, m in causas.items() if m[ini:fin + 1].any()] or ["margen"]
        res.segmentos.append(Segmento(canal, ini, fin, cs))
    res.n_recortadas = int(mascara.sum())
    res.frac_recortada = res.n_recortadas / n

    if res.frac_recortada > MAX_FRAC_RECORTE_NODO:
        res.estado = "DESCARTADO"
        res.motivos = [f"recorte_excesivo:{res.frac_recortada:.0%}>{MAX_FRAC_RECORTE_NODO:.0%}"]
        res.n_recortadas, res.frac_recortada = n, 1.0
        return res, np.ones(n, dtype=bool)
    if res.segmentos:
        res.estado = "RECORTADO"
    return res, mascara


# ---------------------------------------------------------------------------
# LECTURA Y CLASIFICACION DE UN ARCHIVO
# ---------------------------------------------------------------------------
def _leer_texto(ruta_abs: str):
    """Lee el CSV como texto (para que las celdas no tocadas salgan identicas)."""
    with open(ruta_abs, "rb") as fh:
        crudo = fh.read()
    bom = crudo.startswith(b"\xef\xbb\xbf")
    fin_linea = "\r\n" if b"\r\n" in crudo[:4096] else "\n"
    df = pd.read_csv(ruta_abs, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df, bom, fin_linea


def evaluar_archivo(ruta_abs: str, res: ResultadoArchivo):
    """Rellena `res`. Devuelve (df_texto, mascaras{canal: bool[]}, bom, fin_linea) o None si C."""
    try:
        df, bom, fin_linea = _leer_texto(ruta_abs)
    except Exception as exc:  # ilegible / vacio
        res.motivos.append(f"lectura_fallida:{type(exc).__name__}")
        return None

    res.n_filas = int(df.shape[0])
    col_t = next((c for c in df.columns if c in ALIAS_TIEMPO), None)
    res.columna_tiempo = col_t
    canales = [c for c in df.columns if c != col_t]
    if col_t is None:
        res.motivos.append("sin_columna_de_tiempo")
    if sorted(canales) != sorted(CANALES):
        res.motivos.append(f"canales_distintos_de_los_14:{len(canales)}_encontrados")
    if res.n_filas < N_MUESTRAS_MIN:
        res.motivos.append(f"registros_insuficientes:{res.n_filas}<{N_MUESTRAS_MIN}")
    if res.motivos:
        return None

    tm = pd.to_numeric(df[col_t], errors="coerce").to_numpy(dtype=float)
    if np.isnan(tm).any() or not np.all(np.diff(tm) > 0):
        res.motivos.append("tiempo_no_monotono_o_no_numerico")
        return None

    mascaras: dict[str, np.ndarray] = {}
    for canal in CANALES:
        x = pd.to_numeric(df[canal], errors="coerce").to_numpy(dtype=float)
        nodo, mascara = analizar_nodo(x, canal)
        res.nodos.append(nodo)
        mascaras[canal] = mascara

    res.n_nodos_utiles = sum(1 for nd in res.nodos if nd.estado != "DESCARTADO")
    res.n_muestras_recortadas = int(sum(nd.n_recortadas for nd in res.nodos
                                        if nd.estado == "RECORTADO"))
    if res.n_nodos_utiles < MIN_NODOS_UTILES:
        res.motivos.append(f"nodos_utiles:{res.n_nodos_utiles}<{MIN_NODOS_UTILES}")
        return None
    hay_cambio = any(nd.estado != "OK" for nd in res.nodos)
    res.estado = ESTADO_B if hay_cambio else ESTADO_A
    return df, mascaras, bom, fin_linea


def escribir_modificado(ruta_dest: str, df: pd.DataFrame, mascaras: dict,
                        bom: bool, fin_linea: str) -> None:
    """Escribe el CSV con las muestras anomalas vacias. Solo cambian esas celdas."""
    out = df.copy()
    for canal, m in mascaras.items():
        if m.any():
            out.loc[m, canal] = ""
    os.makedirs(os.path.dirname(ruta_dest), exist_ok=True)
    with open(ruta_dest, "w", encoding="utf-8", newline="") as fh:
        if bom:
            fh.write("\ufeff")
        out.to_csv(fh, index=False, lineterminator=fin_linea)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def procesar(captures: str, salida: str, trials_objetivo: int, limite: int | None,
             verbose: bool = True) -> dict:
    rutas = descubrir_archivos(captures)
    if limite is not None:
        rutas = rutas[:limite]
    if verbose:
        print(f"[escaneo] {len(rutas)} CSV candidatos")

    resultados: list[ResultadoArchivo] = []
    cache: dict[str, tuple] = {}
    for k, rel in enumerate(rutas, 1):
        usuario, sub, clase, trial = parsear_nombre(rel)
        ra = os.path.join(captures, rel)
        res = ResultadoArchivo(rel, usuario, sub, clase, trial, sha256=sha256_archivo(ra))
        motivo_et = validar_etiqueta(sub, clase)
        if motivo_et:
            res.motivos.append(motivo_et)
            # aun asi se lee para registrar n_filas
            try:
                res.n_filas = int(pd.read_csv(ra, dtype=str).shape[0])
            except Exception:
                pass
        else:
            ev = evaluar_archivo(ra, res)
            if ev is not None:
                cache[rel] = ev
        resultados.append(res)
        if verbose and k % 500 == 0:
            print(f"[analisis] {k}/{len(rutas)}")

    # --- duplicados exactos (sha256) ---------------------------------------
    por_hash: dict[str, list[ResultadoArchivo]] = {}
    for r in resultados:
        if r.n_filas:                      # ignora archivos vacios
            por_hash.setdefault(r.sha256, []).append(r)
    for grupo in por_hash.values():
        if len(grupo) < 2:
            continue
        grupo.sort(key=lambda r: (r.usuario, r.subcarpeta, r.trial if r.trial is not None else -1,
                                  r.ruta_relativa))
        etiquetas = {(r.usuario, r.clase) for r in grupo}
        if len(etiquetas) > 1:             # mismo contenido bajo etiquetas distintas: ambiguo
            for r in grupo:
                r.motivos.append("duplicado_exacto_con_etiqueta_distinta:" +
                                 "|".join(x.ruta_relativa for x in grupo if x is not r))
                r.estado = ESTADO_C
                cache.pop(r.ruta_relativa, None)
        else:                              # se conserva el primero, los demas son C
            for r in grupo[1:]:
                r.motivos.append(f"duplicado_exacto_de:{grupo[0].ruta_relativa}")
                r.estado = ESTADO_C
                cache.pop(r.ruta_relativa, None)

    # --- salidas -------------------------------------------------------------
    dirs = {ESTADO_A: os.path.join(salida, "A_utilizables"),
            ESTADO_B: os.path.join(salida, "B_modificados")}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    os.makedirs(salida, exist_ok=True)

    for r in resultados:
        if r.estado == ESTADO_A and r.ruta_relativa in cache:
            dest = os.path.join(dirs[ESTADO_A], r.ruta_relativa)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(os.path.join(captures, r.ruta_relativa), dest)
        elif r.estado == ESTADO_B and r.ruta_relativa in cache:
            df, mascaras, bom, fin = cache[r.ruta_relativa]
            escribir_modificado(os.path.join(dirs[ESTADO_B], r.ruta_relativa),
                                df, mascaras, bom, fin)
        elif r.estado != ESTADO_C:         # defensa: nada queda sin estado
            r.estado = ESTADO_C

    return escribir_reportes(resultados, salida, captures, trials_objetivo)


def escribir_reportes(resultados, salida, captures, trials_objetivo) -> dict:
    # manifiesto por archivo
    filas = []
    for r in resultados:
        afect = [nd.canal for nd in r.nodos if nd.estado != "OK"]
        filas.append({
            "ruta_relativa": r.ruta_relativa, "usuario": r.usuario, "subcarpeta": r.subcarpeta,
            "clase": r.clase, "trial": r.trial, "estado": r.estado,
            "incluido": r.estado in (ESTADO_A, ESTADO_B),
            "n_filas": r.n_filas, "n_nodos_utiles": r.n_nodos_utiles,
            "nodos_afectados": "|".join(afect),
            "nodos_descartados": "|".join(nd.canal for nd in r.nodos if nd.estado == "DESCARTADO"),
            "n_muestras_recortadas": r.n_muestras_recortadas,
            "motivos": "|".join(r.motivos), "sha256": r.sha256,
        })
    df_arch = pd.DataFrame(filas)
    df_arch.to_csv(os.path.join(salida, "manifiesto_archivos.csv"), index=False, encoding="utf-8-sig")

    # manifiesto por nodo (solo nodos con algo)
    nod, seg = [], []
    for r in resultados:
        for nd in r.nodos:
            if nd.estado == "OK":
                continue
            nod.append({"ruta_relativa": r.ruta_relativa, "canal": nd.canal, "estado_nodo": nd.estado,
                        "motivos": "|".join(nd.motivos), "n_muestras_recortadas": nd.n_recortadas,
                        "pct_recortado": round(100 * nd.frac_recortada, 2),
                        "dc_mediana_uV": nd.dc_mediana, "sigma_robusta_uV": nd.sigma_robusta})
            for s in nd.segmentos:
                seg.append({"ruta_relativa": r.ruta_relativa, "canal": s.canal,
                            "muestra_inicio": s.inicio, "muestra_fin": s.fin, "n_muestras": s.n,
                            "duracion_ms": round(s.n / FS * 1000, 1),
                            "inicio_ms": round(s.inicio / FS * 1000, 1),
                            "causas": "|".join(s.causas)})
    cols_n = ["ruta_relativa", "canal", "estado_nodo", "motivos", "n_muestras_recortadas",
              "pct_recortado", "dc_mediana_uV", "sigma_robusta_uV"]
    cols_s = ["ruta_relativa", "canal", "muestra_inicio", "muestra_fin", "n_muestras",
              "duracion_ms", "inicio_ms", "causas"]
    pd.DataFrame(nod, columns=cols_n).to_csv(os.path.join(salida, "manifiesto_nodos.csv"),
                                             index=False, encoding="utf-8-sig")
    pd.DataFrame(seg, columns=cols_s).to_csv(os.path.join(salida, "segmentos_recortados.csv"),
                                             index=False, encoding="utf-8-sig")

    # lista de regrabacion (C) y resumen por (usuario, clase)
    c = df_arch[df_arch.estado == ESTADO_C][
        ["usuario", "subcarpeta", "clase", "trial", "ruta_relativa", "motivos"]]
    c.sort_values(["usuario", "subcarpeta", "clase", "trial"]).to_csv(
        os.path.join(salida, "regrabar_archivos_C.csv"), index=False, encoding="utf-8-sig")

    etiq = df_arch[df_arch.clase.notna()]
    piv = (etiq.groupby(["usuario", "subcarpeta", "clase", "estado"]).size()
           .unstack("estado", fill_value=0).reindex(columns=[ESTADO_A, ESTADO_B, ESTADO_C], fill_value=0)
           .reset_index())
    piv["utilizables_A+B"] = piv[ESTADO_A] + piv[ESTADO_B]
    piv["trials_objetivo"] = trials_objetivo
    piv["faltan_por_regrabar"] = (trials_objetivo - piv["utilizables_A+B"]).clip(lower=0)
    piv = piv.rename(columns={ESTADO_A: "n_A", ESTADO_B: "n_B", ESTADO_C: "n_C"})
    piv.sort_values(["faltan_por_regrabar", "usuario", "clase"], ascending=[False, True, True]).to_csv(
        os.path.join(salida, "resumen_usuario_clase.csv"), index=False, encoding="utf-8-sig")

    cuenta = df_arch.estado.value_counts().to_dict()
    motivos_c: dict[str, int] = {}
    for m in c.motivos:
        for t in str(m).split("|"):
            if t:
                k = t.split(":")[0]
                motivos_c[k] = motivos_c.get(k, 0) + 1
    resumen = {
        "version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "captures": os.path.abspath(captures), "salida": os.path.abspath(salida),
        "archivos": int(len(df_arch)),
        "A_utilizables": int(cuenta.get(ESTADO_A, 0)),
        "B_modificados": int(cuenta.get(ESTADO_B, 0)),
        "C_regrabar": int(cuenta.get(ESTADO_C, 0)),
        "motivos_C": motivos_c,
        "parametros": {k: v for k, v in globals().items()
                       if k in ("FS", "N_MUESTRAS_MIN", "DC_TOL_UV", "VENTANA_BASE", "A_SEG_UV", "Z_SEG",
                                "HISTERESIS", "MARGEN_MUESTRAS", "FUSION_HUECO", "A_DERIVA_UV",
                                "RUN_PLANO_MIN", "RUN_PLANO_NODO", "NIVELES_MIN_NODO",
                                "SIGMA_NODO_MAX_UV", "MAX_FRAC_RECORTE_NODO", "MIN_NODOS_UTILES")},
        "convencion_recorte": "muestras anomalas del nodo = celda vacia (NaN); filas y tiempo intactos",
    }
    with open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    return resumen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global MIN_NODOS_UTILES, A_SEG_UV, Z_SEG, MAX_FRAC_RECORTE_NODO
    ap = argparse.ArgumentParser(description="Filtro de calidad EEG: estados A / B / C por archivo.")
    ap.add_argument("--captures", default="captures", help="carpeta de entrada (solo lectura)")
    ap.add_argument("--out", default="resultados_filtro", help="carpeta de salida (fuera de --captures)")
    ap.add_argument("--trials-objetivo", type=int, default=TRIALS_OBJETIVO_DEFECTO,
                    help="trials deseados por (usuario, clase) para calcular cuantos faltan")
    ap.add_argument("--min-nodos-utiles", type=int, default=MIN_NODOS_UTILES,
                    help="minimo de nodos sanos para no caer en C (1 = solo C si TODOS fallan)")
    ap.add_argument("--umbral-uv", type=float, default=A_SEG_UV, help="piso de amplitud de un transitorio")
    ap.add_argument("--z", type=float, default=Z_SEG, help="multiplo de sigma robusta")
    ap.add_argument("--max-recorte-nodo", type=float, default=MAX_FRAC_RECORTE_NODO)
    ap.add_argument("--limite", type=int, default=None, help="solo para pruebas")
    ap.add_argument("--silencioso", action="store_true")
    a = ap.parse_args(argv)

    MIN_NODOS_UTILES, A_SEG_UV, Z_SEG = a.min_nodos_utiles, a.umbral_uv, a.z
    MAX_FRAC_RECORTE_NODO = a.max_recorte_nodo

    entrada, salida = os.path.abspath(a.captures), os.path.abspath(a.out)
    if salida == entrada or salida.startswith(entrada + os.sep) or entrada.startswith(salida + os.sep):
        raise SystemExit("ERROR: entrada y salida no pueden anidarse; los originales no se modifican.")
    if not os.path.isdir(entrada):
        raise SystemExit(f"ERROR: no existe la carpeta de entrada: {entrada}")

    r = procesar(entrada, salida, a.trials_objetivo, a.limite, not a.silencioso)
    if not a.silencioso:
        n = max(r["archivos"], 1)
        print("\n--- RESUMEN ---")
        print(f"archivos            : {r['archivos']}")
        print(f"A utilizables       : {r['A_utilizables']:6d}  ({100*r['A_utilizables']/n:.1f} %)")
        print(f"B modificados       : {r['B_modificados']:6d}  ({100*r['B_modificados']/n:.1f} %)")
        print(f"C a regrabar        : {r['C_regrabar']:6d}  ({100*r['C_regrabar']/n:.1f} %)")
        for m, k in sorted(r["motivos_C"].items(), key=lambda t: -t[1]):
            print(f"   C <- {m}: {k}")
        print(f"salidas en: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
