#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_filter.py  --  Control de calidad y corrección de amplitud EEG.  v2.0.0

PRINCIPIO: el filtro NO corta segmentos, NO borra filas/canales/trials, NO rellena
con ceros y NO desplaza muestras. Número de muestras, duración, eje de tiempo
(columna Tm, texto idéntico) y frecuencia de muestreo salen exactamente iguales.
En B la amplitud original corregida sí se reemplaza en captures/; los valores
alterados se auditan en correcciones_B_muestras.csv. Respalde captures/ antes de
una corrida real si necesita conservar íntegramente los originales.

ESTADOS
  A  Señal correcta: no se modifica ni se copia; se registra en quality_report.csv.
  B  Señal corregible: el archivo original en captures/ se reemplaza atómicamente
     por la señal corregida. Cada modificación queda en correcciones_B.csv (por
     segmento) y en correcciones_B_muestras.csv (por muestra).
  C  No utilizable / no corregible de forma confiable: el original se conserva
     intacto; sólo se añade una fila a regrabar_archivos_C.csv.

MÉTODO DE CORRECCIÓN (ganancia local suave sobre la desviación respecto al nivel DC)
  d_c(t) = x_c(t) - mediana(x_c)                 desviación respecto al nivel DC
  A_c(t) = RMS de d_c en ventana de `ventana_ms` amplitud local
  a_c(t) = A_c(t) / escala_c                     amplitud normalizada por canal
  ref_c(t) = mediana_{k != c} a_k(t)             referencia robusta (leave-one-out)
  q_c(t) = A_c(t) / (escala_c * ref_c(t))        razón relativa de amplitud
  Un canal es anómalo donde q_c(t) > q_deteccion y A_c(t) > piso_abs_uv.
  La zona se extiende (histéresis) a donde q_c(t) > q_inicio.
  Para cada zona anómala (núcleo q > q_deteccion, extendida por histéresis a
  q > q_inicio) la ganancia es CONSTANTE sobre el núcleo:
      g = min(1, q_objetivo * referencia_nucleo / RMS_nucleo)
  con rampas de coseno (1 -> g y g -> 1) en la zona de transición. La salida es
      y_c(t) = DC_c + g_c(t) * d_c(t)
  Dentro del núcleo y_c es exactamente d_c escalada (misma forma, mismos cruces
  por cero, mismos picos y valles relativos, mismo espectro); las rampas suaves
  evitan escalones. Fuera de las zonas anómalas g == 1 exactamente (valores
  idénticos). Alternativa disponible: modo_ganancia="adaptativa".

Ver reporte_criterios_quality_filter.md para la justificación estadística de
cada criterio y la lista completa de parámetros (clase Config).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import ndimage

VERSION = "2.0.0"
CANALES = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1",
           "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]
ALIAS_TIEMPO = ("Tm", "Time", "time", "TIME")
LSB_UV = 0.5151515151
RIEL_SUP_UV = (2 ** 14 - 1) * LSB_UV

# ---- Valores calibrados con datos REALES (métricas por archivo del proyecto:
# ---- resultados_calidad/metricas_calidad.parquet, 6 867 archivos x 14 canales).
# Reproducibles con calibracion_desde_metricas.py. Ver reporte, sección 4.
ESCALA_CANAL_REAL = {   # razón típica (MAD del canal)/(mediana inter-canal), etapa centrada
    "F3": 0.889, "FC5": 0.7745, "AF3": 0.9613, "F7": 1.0, "T7": 0.9465, "P7": 0.7001,
    "O1": 0.8807, "O2": 1.019, "P8": 1.0378, "T8": 1.1078, "F8": 1.1748,
    "AF4": 1.0563, "FC6": 1.1001, "F4": 1.0691}
DC_REF_REAL_UV = {      # mediana de la DC cruda por canal
    "AF3": 4156.2, "AF4": 4235.1, "F3": 4269.3, "F4": 4134.1, "F7": 4416.1,
    "F8": 4436.5, "FC5": 4227.1, "FC6": 4267.5, "O1": 4390.1, "O2": 4301.0,
    "P7": 4541.6, "P8": 4486.5, "T7": 4252.1, "T8": 3758.8}

ESTADO_A, ESTADO_B, ESTADO_C = "A", "B", "C"
CARPETAS_EXCLUIDAS = ("ignorarsenales", "graficas", "graficos")
LETRAS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Ñ"]
DIGITOS = list("0123456789")
COMANDOS = ["↩", "───", "⟵"]
CLASES_POR_CARPETA = {"letters": set(LETRAS), "numbers": set(DIGITOS), "controls": set(COMANDOS)}


# ===========================================================================
# PARÁMETROS (todos modificables: --config params.json o --set clave=valor)
# ===========================================================================
@dataclass
class Config:
    # --- estructura -------------------------------------------------------
    fs: float = 128.0                      # Hz nominal (no se modifica nunca)
    n_muestras_min: int = 240              # < esto -> C (registros insuficientes)
    # --- medida de amplitud ----------------------------------------------
    ventana_ms: float = 250.0              # ventana del RMS local (~1 ciclo de 4 Hz)
    escala_canal: dict = field(default_factory=lambda: dict(ESCALA_CANAL_REAL))  # REAL
    sigma_log_archivo: float = 0.3224      # REAL: sigma robusta de ln(MAD canal / ref) entre canales
    inflacion_sigma_ventana: float = 1.4   # PROVISIONAL: ventana corta > archivo (ver reporte 4.3)
    # --- detección y atenuación ------------------------------------------
    z_deteccion: float = 4.0               # PROVISIONAL: q_deteccion = exp(z * sigma_ventana)
    z_inicio: float = 2.0                  # PROVISIONAL: inicio de la atenuación (histéresis)
    q_objetivo: float = 1.0                # amplitud tras corregir, en unidades de la referencia
    modo_ganancia: str = "segmento"        # "segmento": ganancia constante por zona + rampas (conserva la forma exacta)
                                           # "adaptativa": ganancia por muestra g=f(q) (iguala la envolvente; distorsiona la envolvente)
    piso_abs_uv: float = 25.0              # REAL: p95 de sigma robusta por canal-archivo (30.6 uV)
    rampa_muestras: int = 5                # suavizado final de la ganancia
    fusion_hueco: int = 8                  # une zonas separadas <= 8 muestras
    # --- fiabilidad de la referencia (B vs C) ----------------------------
    delta_ref: float = math.log(1.5)       # error tolerable de la referencia (+-50 %)
    conf_z: float = 1.96                   # confianza 95 % para la referencia
    n_ref_min: int | None = None           # None -> se deriva (ver n_ref_min_efectivo)
    max_elevacion_referencia: float | None = None   # None -> exp(delta_ref)
    min_muestras_fuera: int = 16           # muestras fuera de evento para comparar la referencia
    max_atenuacion: float = 20.0           # PROVISIONAL: g < 1/20 -> C (artefacto >20x la referencia domina la forma)
    max_frac_canal: float = 0.5            # canal anómalo > 50 % del registro -> C
    max_frac_archivo: float = 0.5          # muestras con algún canal anómalo > 50 % -> C
    # --- canales inválidos (no corregibles por atenuación) ----------------
    max_canales_invalidos: int = 0         # > esto -> C
    run_plano_min: int = 26                # REAL: 28 canal-archivo con >=26 idénticas (~200 ms)
    niveles_min: int = 16                  # REAL: <=16 niveles distintos en 26 canal-archivo
    sigma_min_uv: float = 0.5              # sigma robusta <= 0.5 uV = señal plana
    riel_margen_lsb: float = 5.0           # muestra a <= 5 LSB del riel del ADC = saturada
    dc_tol_uv: float = 1000.0              # REAL: sólo >1000 uV el canal es además anómalo (90 %)
    dc_ref_uv: dict = field(default_factory=lambda: dict(DC_REF_REAL_UV))
    z_bajo: float = 5.0                    # amplitud relativa muy baja: q < exp(-z*sigma_archivo)
    min_muestras_bajo: int | None = None   # racha de amplitud relativa muy baja que se considera sostenida (None -> ventana)
    politica_amplitud_comun: str = "mantener"   # "mantener" (A + aviso) | "regrabar" (C) si TODOS los canales suben a la vez
    # --- informativo ------------------------------------------------------
    sigma_poblacional_uv: float = 9.9      # REAL: mediana de sigma robusta (1.4826*MAD)
    umbral_elevacion_comun: float = 3.0    # aviso (no decide) si el nivel común supera 3x

    # ---- derivados ----
    @property
    def sigma_ventana(self) -> float:
        return self.sigma_log_archivo * self.inflacion_sigma_ventana

    @property
    def q_deteccion(self) -> float:
        return math.exp(self.z_deteccion * self.sigma_ventana)

    @property
    def q_inicio(self) -> float:
        return math.exp(self.z_inicio * self.sigma_ventana)

    @property
    def n_ref_min_efectivo(self) -> int:
        """Mínimo de canales de referencia sanos.
        (i) error estándar de la mediana: SE = 1.2533*sigma/sqrt(n); se exige
            conf_z*SE <= delta_ref  ->  n >= (1.2533*sigma*conf_z/delta_ref)^2
        (ii) punto de ruptura de la mediana: mayoría estricta (> C/2)."""
        if self.n_ref_min is not None:
            return int(self.n_ref_min)
        por_se = math.ceil((1.2533 * self.sigma_ventana * self.conf_z / self.delta_ref) ** 2)
        return max(por_se, len(CANALES) // 2 + 1)

    @property
    def max_elevacion_ref_efectivo(self) -> float:
        return self.max_elevacion_referencia if self.max_elevacion_referencia is not None \
            else math.exp(self.delta_ref)

    @property
    def ventana_muestras(self) -> int:
        return max(3, int(round(self.ventana_ms * self.fs / 1000.0)))

    def validar(self):
        assert self.politica_amplitud_comun in ("mantener", "regrabar")
        assert self.modo_ganancia in ("segmento", "adaptativa")
        assert self.q_objetivo <= self.q_inicio <= self.q_deteccion, \
            "se requiere q_objetivo <= q_inicio <= q_deteccion"
        assert self.max_atenuacion > 1 and self.rampa_muestras >= 1

    @classmethod
    def desde_dict(cls, d: dict) -> "Config":
        validos = {f.name for f in fields(cls)}
        malos = set(d) - validos
        if malos:
            raise ValueError(f"parámetros desconocidos: {sorted(malos)}")
        c = cls(**d)
        c.validar()
        return c

    def a_dict(self) -> dict:
        d = asdict(self)
        d.update({"_q_deteccion": self.q_deteccion, "_q_inicio": self.q_inicio,
                  "_sigma_ventana": self.sigma_ventana, "_n_ref_min_efectivo": self.n_ref_min_efectivo,
                  "_max_elevacion_ref_efectivo": self.max_elevacion_ref_efectivo,
                  "_ventana_muestras": self.ventana_muestras})
        return d


# Explicación (para regrabar_archivos_C.csv) de cada código de rechazo
EXPLICACION_C = {
    "ESTRUCTURA": "El archivo no tiene la estructura esperada (lectura, canales, tiempo o muestras insuficientes); no hay registro suficiente para analizarlo ni corregirlo.",
    "ETIQUETA": "La etiqueta (clase/carpeta) no es válida; el registro no se puede asignar a un símbolo.",
    "DUPLICADO": "Contenido idéntico a otro archivo; no aporta un trial independiente.",
    "NO_FINITO": "Contiene NaN o infinitos: los valores no existen, la atenuación de amplitud no puede reconstruirlos.",
    "CANAL_INVALIDO": "Un canal está plano, saturado, desconectado o con amplitud relativa extremadamente baja: no hay señal que atenuar, el problema no es de escala.",
    "REFERENCIA_INSUFICIENTE": "Demasiados canales anómalos a la vez: quedan menos canales sanos que el mínimo necesario para estimar con confianza la amplitud esperada.",
    "REFERENCIA_INESTABLE": "Los canales sanos discrepan entre sí: el error estándar de la referencia supera la tolerancia, la amplitud esperada no es confiable.",
    "REFERENCIA_CONTAMINADA": "Los canales usados como referencia están elevados durante el evento (causa común): la amplitud esperada estaría sesgada.",
    "AMPLITUD_BAJA_RELATIVA": "Uno o varios canales tienen una amplitud sostenidamente mucho menor que la referencia (o la mayoría de canales está elevada y no se puede saber cuál es el normal): no es un problema de exceso de amplitud y no se puede corregir por atenuación.",
    "AMPLITUD_COMUN": "Todos los canales están elevados a la vez y la política configurada es regrabar.",
    "ATENUACION_EXCESIVA": "La atenuación necesaria supera el máximo configurado: la señal es dominada por el artefacto y la forma de onda subyacente no es recuperable.",
    "ANOMALIA_EXTENSA": "La anomalía cubre una fracción demasiado grande del registro: no queda tramo limpio con el cual validar la corrección.",
    "ERROR_INTEGRIDAD": "La verificación de integridad del temporal corregido falló; el original no se reemplaza y se pide regrabar por seguridad.",
}


# ===========================================================================
# UTILIDADES
# ===========================================================================
def sigma_robusta(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def max_run_identicos(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    eq = np.concatenate(([False], x[1:] == x[:-1]))
    inicios = np.flatnonzero(~eq)
    return int(np.diff(np.append(inicios, x.size)).max())


def rachas(mask: np.ndarray):
    if not mask.any():
        return []
    b = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(b == 1)[0].tolist(), (np.where(b == -1)[0] - 1).tolist()))


def cerrar_huecos(mask: np.ndarray, hueco: int) -> np.ndarray:
    out = mask.copy()
    r = rachas(mask)
    for (_, f0), (i1, _) in zip(r, r[1:]):
        if i1 - f0 - 1 <= hueco:
            out[f0:i1] = True
    return out


def suavizar(g: np.ndarray, L: int) -> np.ndarray:
    if L <= 1:
        return g
    w = np.hanning(L + 2)[1:-1]
    w /= w.sum()
    return np.convolve(np.pad(g, (L // 2, L // 2), mode="edge"), w, mode="valid")


# ===========================================================================
# ANÁLISIS DE UN REGISTRO (función pura: ndarray N x 14 -> resultado)
# ===========================================================================
@dataclass
class Segmento:
    id: str
    canal: str
    inicio: int
    fin: int
    rms_antes: float
    rms_despues: float
    pico_antes: float
    pico_despues: float
    g_min: float
    g_mediana: float
    q_max: float
    a_max: float
    referencia_uv: float
    canales_referencia: list
    criterio: str
    corr_forma: float


@dataclass
class Analisis:
    estado: str = ESTADO_C
    codigos_C: list = field(default_factory=list)
    motivos: list = field(default_factory=list)
    Y: np.ndarray | None = None                 # señal de salida (mismo shape)
    segmentos: list = field(default_factory=list)
    muestras: list = field(default_factory=list)   # una fila por muestra modificada
    canales_corregidos: list = field(default_factory=list)
    canales_invalidos: dict = field(default_factory=dict)
    canales_afectados: list = field(default_factory=list)
    pct_por_canal: dict = field(default_factory=dict)
    amplitudes: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)


def analizar_registro(X: np.ndarray, cfg: Config) -> Analisis:
    """Clasifica y (si es B) corrige un registro. X: (N muestras, 14 canales) en µV."""
    cfg.validar()
    res = Analisis()
    N, C = X.shape
    assert C == len(CANALES)
    res.metricas["n_muestras"] = N

    # ---- 1. valores no finitos ------------------------------------------------
    nf = ~np.isfinite(X)
    if nf.any():
        res.codigos_C.append("NO_FINITO")
        res.canales_afectados = [CANALES[j] for j in np.where(nf.any(axis=0))[0]]
        res.pct_por_canal = {CANALES[j]: round(100 * nf[:, j].mean(), 2) for j in np.where(nf.any(axis=0))[0]}
        res.motivos.append(f"valores_no_finitos:{int(nf.sum())}")
        res.metricas["n_no_finitos"] = int(nf.sum())
        return _cerrar_C(res, N, C)

    dc = np.median(X, axis=0)
    D = X - dc
    res.metricas["max_amp_original_uv"] = float(np.abs(D).max())
    res.metricas["max_salto_uv"] = float(np.abs(np.diff(X, axis=0)).max())
    sig = sigma_robusta(D)
    m = np.array([cfg.escala_canal[c] for c in CANALES])

    # ---- 2. canales inválidos (no corregibles por atenuación) ----------------
    inval: dict[str, list] = {}
    for j, c in enumerate(CANALES):
        mot = []
        x = X[:, j]
        r = max_run_identicos(x)
        if r >= cfg.run_plano_min:
            mot.append(f"plano:racha_identica={r}>={cfg.run_plano_min}")
        if np.unique(x).size <= cfg.niveles_min:
            mot.append(f"plano:niveles={np.unique(x).size}<={cfg.niveles_min}")
        if sig[j] <= cfg.sigma_min_uv:
            mot.append(f"plano:sigma={sig[j]:.2f}uV<={cfg.sigma_min_uv}")
        if (x >= RIEL_SUP_UV - cfg.riel_margen_lsb * LSB_UV).any() or (x <= cfg.riel_margen_lsb * LSB_UV).any():
            mot.append("saturacion:muestra_en_el_riel_del_ADC")
        if abs(dc[j] - cfg.dc_ref_uv[c]) > cfg.dc_tol_uv:
            mot.append(f"electrodo_desconectado:DC={dc[j]:.0f}uV(ref {cfg.dc_ref_uv[c]:.0f}+-{cfg.dc_tol_uv:.0f})")
        if mot:
            inval[c] = mot
    # amplitud relativa extremadamente baja (nivel de archivo, misma métrica que la calibración real)
    an = sig / m
    for j, c in enumerate(CANALES):
        if c in inval:
            continue
        otros = np.delete(an, j)
        ref_f = np.median(otros)
        if ref_f > 0 and an[j] / ref_f < math.exp(-cfg.z_bajo * cfg.sigma_log_archivo):
            inval[c] = [f"amplitud_relativa_muy_baja:q={an[j] / ref_f:.3f}<{math.exp(-cfg.z_bajo * cfg.sigma_log_archivo):.3f}"]
    res.canales_invalidos = inval
    res.metricas["n_canales_invalidos"] = len(inval)
    if len(inval) > cfg.max_canales_invalidos:
        res.codigos_C.append("CANAL_INVALIDO")
        res.canales_afectados = list(inval)
        res.pct_por_canal = {c: 100.0 for c in inval}
        res.amplitudes = {c: f"sigma={sig[CANALES.index(c)]:.2f}uV;pico={np.abs(D[:, CANALES.index(c)]).max():.0f}uV" for c in inval}
        res.motivos += [f"{c}:{'+'.join(v)}" for c, v in inval.items()]
        return _cerrar_C(res, N, C)

    # ---- 3. amplitud local relativa --------------------------------------------
    W = cfg.ventana_muestras
    A = np.sqrt(ndimage.uniform_filter1d(D ** 2, size=W, axis=0, mode="nearest"))
    a = A / m
    ref = np.empty_like(a)
    for j in range(C):
        ref[:, j] = np.median(np.delete(a, j, axis=1), axis=1)
    esperado = np.maximum(ref * m, cfg.sigma_min_uv)
    q = A / esperado
    res.metricas["q_max"] = float(q.max())
    res.metricas["q_min"] = float(q.min())
    res.metricas["nivel_comun_uv"] = float(np.median(sig))
    pico_comun = float(np.max(np.median(a, axis=1)) / cfg.sigma_poblacional_uv)
    res.metricas["elevacion_comun_max"] = round(pico_comun, 2)
    if pico_comun > cfg.umbral_elevacion_comun:
        res.metricas["aviso"] = "amplitud_comun_elevada(no_decide)"

    # amplitud relativa sostenidamente muy baja (el otro extremo del criterio relativo)
    bajo = (q < 1.0 / cfg.q_deteccion) & (esperado > cfg.piso_abs_uv)
    min_bajo = cfg.min_muestras_bajo if cfg.min_muestras_bajo is not None else W
    bajos = [j for j in range(C) if any(f - i + 1 >= min_bajo for i, f in rachas(bajo[:, j]))]
    if bajos:
        res.codigos_C.append("AMPLITUD_BAJA_RELATIVA")
        res.canales_afectados = [CANALES[j] for j in bajos]
        res.pct_por_canal = {CANALES[j]: round(100 * bajo[:, j].mean(), 2) for j in bajos}
        res.amplitudes = {CANALES[j]: f"q_min={q[:, j].min():.3f};referencia={esperado[:, j].max():.0f}uV" for j in bajos}
        res.motivos.append(f"amplitud_relativa_sostenida_<1/{cfg.q_deteccion:.1f}_en:{','.join(CANALES[j] for j in bajos)}")
        return _cerrar_C(res, N, C)
    if cfg.politica_amplitud_comun == "regrabar" and pico_comun > cfg.umbral_elevacion_comun:
        res.codigos_C.append("AMPLITUD_COMUN")
        res.motivos.append(f"amplitud_comun_elevada_x{pico_comun:.1f}")
        return _cerrar_C(res, N, C)

    Q_det, Q_ini, piso = cfg.q_deteccion, cfg.q_inicio, cfg.piso_abs_uv
    core = (q > Q_det) & (A > piso) | (A > 150.0)
    cand = (q > Q_ini) & (A > piso)
    region = np.zeros((N, C), dtype=bool)
    for j in range(C):
        if not core[:, j].any():
            continue
        lab, n = ndimage.label(cand[:, j])
        keep = np.unique(lab[core[:, j]])
        reg = np.isin(lab, keep[keep > 0])
        region[:, j] = cerrar_huecos(reg, cfg.fusion_hueco)
    flag_cols = np.where(region.any(axis=0))[0]
    res.metricas["n_canales_anomalos"] = int(flag_cols.size)

    if flag_cols.size == 0:
        res.estado = ESTADO_A
        res.Y = X.copy()
        res.motivos.append("sin_anomalias_de_amplitud_relativa")
        _metricas_finales(res, X, X, N, C)
        return res

    # ---- 4. fiabilidad de la referencia por evento -----------------------------
    frac_canal = region.mean(axis=0)
    res.canales_afectados = [CANALES[j] for j in flag_cols]
    res.pct_por_canal = {CANALES[j]: round(100 * frac_canal[j], 2) for j in flag_cols}
    res.amplitudes = {CANALES[j]: f"rms_max={A[:, j].max():.0f}uV;pico={np.abs(D[:, j]).max():.0f}uV;q_max={q[:, j].max():.1f}" for j in flag_cols}
    union = region.any(axis=1)
    res.metricas["pct_muestras_afectadas_archivo"] = round(100 * union.mean(), 2)
    res.metricas["pct_muestras_afectadas_canales"] = round(100 * region.sum() / (N * C), 2)
    k_t = region.sum(axis=1)
    res.metricas["k_max_simultaneos"] = int(k_t.max())
    res.metricas["solape_multicanal_pct"] = round(100 * float((k_t >= 2).sum() / max(union.sum(), 1)), 1)
    n_min = cfg.n_ref_min_efectivo
    sig_prior = cfg.sigma_ventana
    peor_nref, peor_se, peor_shift = C, 0.0, 1.0
    eventos = rachas(cerrar_huecos(union, cfg.fusion_hueco))
    fuera = ~union
    for (i0, i1) in eventos:
        sl = slice(i0, i1 + 1)
        regs = region[sl]
        n_ref_t = C - regs.sum(axis=1)
        peor_nref = min(peor_nref, int(n_ref_t.min()))
        ses = []
        for t in range(i1 - i0 + 1):
            ok = ~regs[t]
            n_ok = int(ok.sum())
            if n_ok == 0:
                ses.append(np.inf)
                continue
            s_obs = sigma_robusta(np.log(np.maximum(a[i0 + t, ok], 1e-9))) if n_ok >= 4 else 0.0
            ses.append(1.2533 * max(sig_prior, float(s_obs)) / math.sqrt(n_ok))
        se_med = float(np.median(ses))
        peor_se = max(peor_se, se_med)
        chans_ev = np.where(regs.any(axis=0))[0]
        ref_ch = [j for j in range(C) if j not in chans_ev]
        if ref_ch and fuera.sum() >= cfg.min_muestras_fuera:
            a_in = np.sqrt(np.mean(D[sl][:, ref_ch] ** 2, axis=0))
            a_out = np.sqrt(np.mean(D[fuera][:, ref_ch] ** 2, axis=0))
            shift = float(np.median(a_in / np.maximum(a_out, 1e-9)))
            peor_shift = max(peor_shift, shift)
    res.metricas.update({"n_ref_min_observado": peor_nref, "n_ref_min_requerido": n_min,
                         "se_ref_max": round(peor_se, 4), "tolerancia_se": round(cfg.delta_ref / cfg.conf_z, 4),
                         "elevacion_referencia_max": round(peor_shift, 3)})
    if peor_nref < n_min:
        res.codigos_C.append("REFERENCIA_INSUFICIENTE")
        res.motivos.append(f"canales_de_referencia_sanos={peor_nref}<{n_min}")
    if not (peor_se * cfg.conf_z <= cfg.delta_ref) and "REFERENCIA_INSUFICIENTE" not in res.codigos_C:
        res.codigos_C.append("REFERENCIA_INESTABLE")
        res.motivos.append(f"error_estandar_referencia={peor_se:.3f}*{cfg.conf_z}>{cfg.delta_ref:.3f}")
    if peor_shift > cfg.max_elevacion_ref_efectivo:
        res.codigos_C.append("REFERENCIA_CONTAMINADA")
        res.motivos.append(f"referencia_elevada_x{peor_shift:.2f}>x{cfg.max_elevacion_ref_efectivo:.2f}_en_el_evento")
    if frac_canal.max() > cfg.max_frac_canal or union.mean() > cfg.max_frac_archivo:
        res.codigos_C.append("ANOMALIA_EXTENSA")
        res.motivos.append(f"canal_max={100 * frac_canal.max():.0f}%>{100 * cfg.max_frac_canal:.0f}% o archivo={100 * union.mean():.0f}%>{100 * cfg.max_frac_archivo:.0f}%")

    # ---- 5. ganancia (también necesaria para validar la atenuación máxima) ------
    G = np.ones((N, C))
    lq_s, lq_f = math.log(Q_ini), math.log(Q_det)
    for j in flag_cols:
        g = np.ones(N)
        for (r0, r1) in rachas(region[:, j]):
            sl = slice(r0, r1 + 1)
            idx = np.where(core[sl, j])[0]
            if idx.size == 0:
                continue
            if cfg.modo_ganancia == "adaptativa":
                lq = np.log(np.maximum(q[sl, j], 1e-12))
                u = np.clip((lq - lq_s) / (lq_f - lq_s), 0, 1)
                sm = 0.5 * (1 - np.cos(np.pi * u))
                g[sl] = np.minimum(np.exp(-sm * (lq - math.log(cfg.q_objetivo))), 1.0)
                continue
            # modo "segmento": ganancia CONSTANTE sobre el núcleo anómalo (forma idéntica, sólo cambia la escala)
            c0, c1 = r0 + idx[0], r0 + idx[-1]
            a_nuc = math.sqrt(float(np.mean(D[c0:c1 + 1, j] ** 2)))
            e_nuc = float(np.median(esperado[c0:c1 + 1, j]))
            gs = min(1.0, cfg.q_objetivo * e_nuc / max(a_nuc, 1e-9))
            g[c0:c1 + 1] = gs
            if c0 > r0:                       # rampa de entrada (coseno) dentro de la zona de transición
                n = c0 - r0
                g[r0:c0] = 1 - (1 - gs) * 0.5 * (1 - np.cos(np.pi * (np.arange(n) + 1) / (n + 1)))
            if r1 > c1:                       # rampa de salida
                n = r1 - c1
                g[c1 + 1:r1 + 1] = 1 - (1 - gs) * 0.5 * (1 + np.cos(np.pi * (np.arange(n) + 1) / (n + 1)))
        g = suavizar(g, cfg.rampa_muestras)
        ext = ndimage.binary_dilation(region[:, j], structure=np.ones(cfg.rampa_muestras, dtype=bool))
        g = np.where(ext, g, 1.0)
        g[np.abs(g - 1.0) < 1e-6] = 1.0
        G[:, j] = g
    res.metricas["ganancia_min"] = float(G.min())
    if G.min() < 1.0 / cfg.max_atenuacion:
        res.codigos_C.append("ATENUACION_EXCESIVA")
        res.motivos.append(f"ganancia_requerida={G.min():.4f}<1/{cfg.max_atenuacion:g}")

    if res.codigos_C:
        res.metricas["q_max"] = float(q.max())
        return _cerrar_C(res, N, C)

    # ---- 6. aplicar corrección y auditar ------------------------------------------
    Y = X.copy()
    for j in flag_cols:
        sel = G[:, j] < 1.0                      # sólo se tocan muestras con ganancia < 1
        Y[sel, j] = np.round(dc[j] + G[sel, j] * D[sel, j], 4)
    mod = np.abs(Y - X) > 1e-9
    Dy = Y - dc
    for j in flag_cols:
        res.canales_corregidos.append(CANALES[j])
        n_seg = 0
        for (i0, i1) in rachas(mod[:, j]):
            n_seg += 1
            sl = slice(i0, i1 + 1)
            ref_ch = [CANALES[k] for k in range(C) if k != j and not region[sl, k].any()]
            tiene_core = bool(core[sl, j].any())
            crit = (f"q_max={q[sl, j].max():.2f}>q_deteccion={Q_det:.2f} y A={A[sl, j].max():.0f}uV>piso={piso:g}uV"
                    if tiene_core else f"extension_por_histeresis(q>{Q_ini:.2f})")
            o, y_ = D[sl, j], Dy[sl, j]
            cf = float(np.corrcoef(o, y_)[0, 1]) if (o.std() > 0 and y_.std() > 0 and len(o) > 2) else float("nan")
            seg = Segmento(id=f"{CANALES[j]}-{n_seg}", canal=CANALES[j], inicio=i0, fin=i1,
                           rms_antes=float(np.sqrt(np.mean(o ** 2))), rms_despues=float(np.sqrt(np.mean(y_ ** 2))),
                           pico_antes=float(np.abs(o).max()), pico_despues=float(np.abs(y_).max()),
                           g_min=float(G[sl, j].min()), g_mediana=float(np.median(G[sl, j])),
                           q_max=float(q[sl, j].max()), a_max=float(A[sl, j].max()),
                           referencia_uv=float(np.median(esperado[sl, j])), canales_referencia=ref_ch,
                           criterio=crit, corr_forma=cf)
            res.segmentos.append(seg)
            for t in range(i0, i1 + 1):
                if mod[t, j]:
                    res.muestras.append({"canal": CANALES[j], "muestra": t, "valor_original": float(X[t, j]),
                                         "valor_corregido": float(Y[t, j]), "ganancia": float(G[t, j]),
                                         "q": float(q[t, j]), "amplitud_local_uv": float(A[t, j]),
                                         "referencia_uv": float(esperado[t, j]), "id_segmento": seg.id,
                                         "criterio": crit})
    res.estado = ESTADO_B
    res.Y = Y
    res.motivos.append(f"canales_corregidos={','.join(res.canales_corregidos)}")
    _metricas_finales(res, X, Y, N, C)
    Ay = np.sqrt(ndimage.uniform_filter1d(Dy ** 2, size=W, axis=0, mode="nearest"))
    ay = Ay / m
    refy = np.stack([np.median(np.delete(ay, j, axis=1), axis=1) for j in range(C)], axis=1)
    res.metricas["q_max_despues"] = float((Ay / np.maximum(refy * m, cfg.sigma_min_uv)).max())
    return res


def _metricas_finales(res: Analisis, X, Y, N, C):
    dc = np.median(X, axis=0)
    res.metricas["max_amp_corregida_uv"] = float(np.abs(Y - dc).max())
    res.metricas["n_muestras_modificadas"] = int((np.abs(Y - X) > 1e-9).sum())
    res.metricas["pct_muestras_modificadas"] = round(100 * res.metricas["n_muestras_modificadas"] / (N * C), 3)


def _cerrar_C(res: Analisis, N, C) -> Analisis:
    res.estado = ESTADO_C
    res.Y = None
    if "pct_muestras_afectadas_canales" not in res.metricas and res.pct_por_canal:
        res.metricas["pct_muestras_afectadas_canales"] = round(sum(res.pct_por_canal.values()) / C, 2)
    return res


# ===========================================================================
# ARCHIVOS
# ===========================================================================
def sha256_archivo(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for t in iter(lambda: fh.read(1 << 20), b""):
            h.update(t)
    return h.hexdigest()


def descubrir_archivos(raiz: str) -> list[str]:
    print(f"Descubriendo archivos CSV bajo {raiz} ...")
    out = []
    for dp, dn, fn in os.walk(raiz):
        dn[:] = sorted(d for d in dn if d.lower() not in CARPETAS_EXCLUIDAS)
        rel_dir = os.path.relpath(dp, raiz)
        rel_dir = "" if rel_dir == "." else rel_dir
        for n in sorted(fn):
            nombre = n.casefold()
            if not nombre.endswith(".csv"):
                continue
            # Exclude known non-signal exports that may live under captures.
            # Keep other unknown folders/files visible so malformed EEG is not silently lost.
            if nombre == "resultados_clasificacion.csv" or nombre.startswith(("metricas_", "metrics_")):
                continue
            out.append(os.path.join(rel_dir, n).replace("\\", "/"))
    return sorted(out)


def validar_usuario(usuario: str) -> str:
    usuario = usuario.strip()
    if (
        not usuario or usuario in {".", ".."}
        or "/" in usuario or "\\" in usuario or os.path.splitdrive(usuario)[0]
    ):
        raise ValueError("El usuario debe ser el nombre de una sola carpeta dentro de captures/.")
    return usuario


def carpeta_reportes_usuario(salida: str, usuario: str) -> str:
    usuario = validar_usuario(usuario)
    salida = os.path.abspath(salida)
    if os.path.basename(os.path.normpath(salida)).casefold() == usuario.casefold():
        return salida
    return os.path.join(salida, usuario)


def rutas_de_usuario(captures: str, usuario: str) -> list[str]:
    """Return CSV paths under exactly one top-level user directory."""
    usuario = validar_usuario(usuario)
    carpeta_usuario = os.path.join(captures, usuario)
    if os.path.islink(carpeta_usuario) or not os.path.isdir(carpeta_usuario):
        raise FileNotFoundError(f"No existe la carpeta del usuario dentro de captures/: {carpeta_usuario}")
    rutas = [
        usuario + "/" + ruta
        for ruta in descubrir_archivos(carpeta_usuario)
    ]
    if not rutas:
        raise FileNotFoundError(f"No se encontraron archivos CSV para el usuario {usuario!r}.")
    return rutas


def parsear_nombre(rel: str):
    p = rel.split("/")
    usuario = p[0] if len(p) > 1 else ""
    sub = p[1] if len(p) > 2 else ""
    t = os.path.splitext(p[-1])[0].split("_")
    clase, trial = None, None
    if len(t) >= 3:
        clase = "_".join(t[1:-1])
        try:
            trial = int(t[-1])
        except ValueError:
            pass
    return usuario, sub, clase, trial


def validar_etiqueta(sub: str, clase: str | None) -> str | None:
    todas = set().union(*CLASES_POR_CARPETA.values())
    if clase is None or clase not in todas:
        return f"etiqueta_invalida:{clase}"
    esp = CLASES_POR_CARPETA.get(sub.lower())
    if esp is None:
        return f"carpeta_desconocida:{sub}"
    if clase not in esp:
        return f"clase_incoherente_con_carpeta:{clase}@{sub}"
    return None


def leer_texto(ruta: str):
    with open(ruta, "rb") as fh:
        crudo = fh.read()
    bom = crudo.startswith(b"\xef\xbb\xbf")
    fin = "\r\n" if b"\r\n" in crudo[:4096] else "\n"
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df, bom, fin


def fmt(v: float) -> str:
    return f"{v:.4f}"


def construir_salida(df: pd.DataFrame, Y: np.ndarray, X: np.ndarray) -> pd.DataFrame:
    """Copia del DataFrame de TEXTO con sólo las celdas modificadas reescritas."""
    out = df.copy()
    mod = np.abs(Y - X) > 1e-9
    for j, c in enumerate(CANALES):
        if mod[:, j].any():
            col = out[c].to_numpy(dtype=object).copy()
            for t in np.where(mod[:, j])[0]:
                col[t] = fmt(Y[t, j])
            out[c] = col
    return out


def verificar_integridad(df0: pd.DataFrame, df1: pd.DataFrame, X: np.ndarray, Y: np.ndarray,
                         col_t: str) -> None:
    """Lanza AssertionError si se violó alguna invariante."""
    assert df0.shape == df1.shape, "cambió el número de filas/columnas"
    assert list(df0.columns) == list(df1.columns), "cambiaron las columnas"
    assert df0[col_t].equals(df1[col_t]), "cambió la columna de tiempo"
    mod = np.abs(Y - X) > 1e-9
    for j, c in enumerate(CANALES):
        sin = ~mod[:, j]
        assert (df0[c].to_numpy()[sin] == df1[c].to_numpy()[sin]).all(), f"cambió una muestra no auditada en {c}"
    assert np.isfinite(pd.to_numeric(df1[CANALES].stack(), errors="coerce")).all(), "valores no finitos en la salida"


def escribir_corregido_en_sitio(ruta: str, df0: pd.DataFrame, df1: pd.DataFrame,
                                X: np.ndarray, Y: np.ndarray, col_t: str,
                                bom: bool, fin: str) -> None:
    """Validate a temporary sibling, then atomically replace the original path."""
    if os.path.islink(ruta):
        raise AssertionError("no se sobrescribe un enlace simbólico")
    modo = stat.S_IMODE(os.stat(ruta).st_mode)
    directorio = os.path.dirname(ruta)
    fd, temporal = tempfile.mkstemp(prefix=".quality_filter_", suffix=".tmp", dir=directorio)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            if bom:
                fh.write("\ufeff")
            df1.to_csv(fh, index=False, lineterminator=fin)
            fh.flush()
            os.fsync(fh.fileno())
        comprobacion, _, _ = leer_texto(temporal)
        verificar_integridad(df0, comprobacion, X, Y, col_t)
        assert comprobacion.equals(df1), "el CSV escrito difiere del DataFrame corregido"
        os.chmod(temporal, modo)
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def leer_matriz(df: pd.DataFrame, col_t: str):
    tm = pd.to_numeric(df[col_t], errors="coerce").to_numpy(float)
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in CANALES])
    return tm, X


def evaluar_archivo(ruta_abs: str, rel: str, cfg: Config):
    """Devuelve dict con todo lo necesario. Nunca modifica el original."""
    usuario, sub, clase, trial = parsear_nombre(rel)
    info = {"rel": rel, "usuario": usuario, "subcarpeta": sub, "clase": clase, "trial": trial,
            "sha256": sha256_archivo(ruta_abs), "n_filas": None, "analisis": None, "df": None,
            "X": None, "tm": None, "col_t": None, "bom": False, "fin": "\n"}
    et = validar_etiqueta(sub, clase)
    a = Analisis()
    try:
        df, bom, fin = leer_texto(ruta_abs)
    except Exception as exc:
        a.codigos_C.append("ESTRUCTURA")
        a.motivos.append(f"lectura_fallida:{type(exc).__name__}")
        info["analisis"] = a
        return info
    info.update({"df": df, "bom": bom, "fin": fin, "n_filas": int(df.shape[0])})
    if et:
        a.codigos_C.append("ETIQUETA")
        a.motivos.append(et)
        info["analisis"] = a
        return info
    col_t = next((c for c in df.columns if c in ALIAS_TIEMPO), None)
    info["col_t"] = col_t
    canales = [c for c in df.columns if c != col_t]
    errs = []
    if col_t is None:
        errs.append("sin_columna_de_tiempo")
    if sorted(canales) != sorted(CANALES):
        errs.append(f"canales_distintos_de_los_14:{len(canales)}")
    if df.shape[0] < cfg.n_muestras_min:
        errs.append(f"registros_insuficientes:{df.shape[0]}<{cfg.n_muestras_min}")
    if errs:
        a.codigos_C.append("ESTRUCTURA")
        a.motivos += errs
        info["analisis"] = a
        return info
    tm, X = leer_matriz(df, col_t)
    if np.isnan(tm).any() or not np.all(np.diff(tm) > 0):
        a.codigos_C.append("ESTRUCTURA")
        a.motivos.append("tiempo_no_monotono_o_no_numerico")
        info["analisis"] = a
        return info
    info.update({"tm": tm, "X": X})
    info["analisis"] = analizar_registro(X, cfg)
    return info


# ===========================================================================
# PIPELINE
# ===========================================================================
def procesar(captures: str, salida: str, cfg: Config, trials_objetivo: int = 30,
             limite: int | None = None, verbose: bool = True,
             overwrite_reports: bool = False, usuario: str | None = None) -> dict:
    del overwrite_reports
    cfg.validar()
    captures = os.path.abspath(captures)
    print(captures)
    if usuario is not None:
        salida = carpeta_reportes_usuario(salida, usuario)
    else:
        salida = os.path.abspath(salida)
    cap_norm, out_norm = os.path.normcase(captures), os.path.normcase(salida)
    if out_norm == cap_norm or out_norm.startswith(cap_norm + os.sep) or cap_norm.startswith(out_norm + os.sep):
        raise ValueError("captures y carpeta de reportes no pueden anidarse; no se escribirá dentro de las señales.")
    if not os.path.isdir(captures):
        raise FileNotFoundError(f"No existe la carpeta captures: {captures}")
    rutas = (
        rutas_de_usuario(captures, usuario)
        if usuario is not None else descubrir_archivos(captures)
    )

    if limite:
        rutas = rutas[:limite]
    if verbose:
        print(f"[escaneo] {len(rutas)} CSV")
    infos = []
    for k, rel in enumerate(rutas, 1):
        infos.append(evaluar_archivo(os.path.join(captures, rel), rel, cfg))
        if verbose and k % 500 == 0:
            print(f"[analisis] {k}/{len(rutas)}")

    # duplicados exactos
    por_hash: dict[str, list] = {}
    for i in infos:
        if i["n_filas"]:
            por_hash.setdefault(i["sha256"], []).append(i)
    for g in por_hash.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda i: (i["usuario"], i["subcarpeta"], i["trial"] if i["trial"] is not None else -1, i["rel"]))
        distintas = len({(i["usuario"], i["clase"]) for i in g}) > 1
        for i in (g if distintas else g[1:]):
            a = i["analisis"]
            a.codigos_C.append("DUPLICADO")
            a.motivos.append("duplicado_exacto_de:" + "|".join(x["rel"] for x in g if x is not i))
            a.estado, a.Y = ESTADO_C, None

    os.makedirs(salida, exist_ok=True)

    q_rows, b_rows, bs_rows, c_rows = [], [], [], []
    for i in infos:
        a: Analisis = i["analisis"]
        if a.codigos_C:
            a.estado = ESTADO_C
        # --- escribir B con verificación de integridad
        if a.estado == ESTADO_B:
            try:
                out_df = construir_salida(i["df"], a.Y, i["X"])
                verificar_integridad(i["df"], out_df, i["X"], a.Y, i["col_t"])
                ruta_original = os.path.join(captures, i["rel"])
                escribir_corregido_en_sitio(
                    ruta_original, i["df"], out_df, i["X"], a.Y,
                    i["col_t"], i["bom"], i["fin"],
                )
            except AssertionError as exc:
                a.codigos_C.append("ERROR_INTEGRIDAD")
                a.motivos.append(str(exc))
                a.estado, a.Y, a.segmentos, a.muestras = ESTADO_C, None, [], []

        tm = i["tm"]
        def tmv(t):
            return float(tm[t]) if tm is not None else None
        N = i["n_filas"]
        for s in a.segmentos:
            b_rows.append({
                "archivo": i["rel"], "usuario": i["usuario"], "clase": i["clase"], "trial": i["trial"],
                "canal": s.canal, "id_segmento": s.id, "muestra_inicial": s.inicio, "muestra_final": s.fin,
                "n_muestras": s.fin - s.inicio + 1,
                "tiempo_inicial_ms": round(s.inicio / cfg.fs * 1000, 2), "tiempo_final_ms": round(s.fin / cfg.fs * 1000, 2),
                "duracion_ms": round((s.fin - s.inicio + 1) / cfg.fs * 1000, 2),
                "tm_inicial": tmv(s.inicio), "tm_final": tmv(s.fin),
                "rms_antes_uv": round(s.rms_antes, 3), "rms_despues_uv": round(s.rms_despues, 3),
                "pico_antes_uv": round(s.pico_antes, 3), "pico_despues_uv": round(s.pico_despues, 3),
                "factor_atenuacion_min": round(s.g_min, 5), "factor_atenuacion_mediano": round(s.g_mediana, 5),
                "razon_q_max": round(s.q_max, 3), "amplitud_local_max_uv": round(s.a_max, 3),
                "metodo": f"ganancia_{cfg.modo_ganancia}_sobre_desviacion_respecto_a_mediana_DC;RMS_local_{cfg.ventana_ms:g}ms;rampas_coseno",
                "referencia": "mediana_leave_one_out_de_amplitud_normalizada_entre_canales",
                "referencia_uv": round(s.referencia_uv, 3), "canales_referencia": "|".join(s.canales_referencia),
                "criterio_activacion": s.criterio,
                "motivo": "amplitud local anómala respecto a la referencia inter-canal; corrección de escala sin cortar muestras",
                "corr_forma_original_vs_corregida": None if np.isnan(s.corr_forma) else round(s.corr_forma, 5)})
        for r in a.muestras:
            bs_rows.append({"archivo": i["rel"], "usuario": i["usuario"], "clase": i["clase"], "trial": i["trial"],
                            "canal": r["canal"], "muestra": r["muestra"], "tm": tmv(r["muestra"]),
                            "tiempo_ms": round(r["muestra"] / cfg.fs * 1000, 2),
                            "valor_original": r["valor_original"], "valor_corregido": r["valor_corregido"],
                            "delta": round(r["valor_corregido"] - r["valor_original"], 4),
                            "ganancia": round(r["ganancia"], 6), "razon_q": round(r["q"], 3),
                            "amplitud_local_uv": round(r["amplitud_local_uv"], 3),
                            "referencia_uv": round(r["referencia_uv"], 3),
                            "id_segmento": r["id_segmento"], "criterio": r["criterio"]})
        met = a.metricas
        if a.estado == ESTADO_C:
            c_rows.append({
                "archivo": i["rel"], "usuario": i["usuario"], "clase": i["clase"], "trial": i["trial"],
                "canales_afectados": "|".join(a.canales_afectados), "pct_muestras_afectadas": met.get("pct_muestras_afectadas_canales"),
                "pct_por_canal": "|".join(f"{c}:{v}%" for c, v in a.pct_por_canal.items()),
                "amplitudes_detectadas": "|".join(f"{c}:{v}" for c, v in a.amplitudes.items()),
                "codigos_C": "|".join(a.codigos_C), "motivo": "|".join(a.motivos),
                "metricas": json.dumps({k: v for k, v in met.items()}, ensure_ascii=False),
                "explicacion_no_confiable": " ".join(EXPLICACION_C.get(c, c) for c in a.codigos_C),
                "accion": f"REGRABAR: {i['usuario']} / {i['clase']} (trial {i['trial']}). El archivo original se conserva intacto."})
        ncan = len(CANALES)
        q_rows.append({
            "archivo": i["rel"], "usuario": i["usuario"], "subcarpeta": i["subcarpeta"], "clase": i["clase"], "trial": i["trial"],
            "clasificacion": a.estado, "n_filas": N, "n_canales": ncan if i["X"] is not None else None,
            "canales_corregidos": "|".join(a.canales_corregidos), "n_segmentos_corregidos": len(a.segmentos),
            "pct_muestras_modificadas": met.get("pct_muestras_modificadas", 0.0 if a.estado == ESTADO_A else None),
            "max_amplitud_original_uv": None if "max_amp_original_uv" not in met else round(met["max_amp_original_uv"], 2),
            "max_amplitud_corregida_uv": None if "max_amp_corregida_uv" not in met else round(met["max_amp_corregida_uv"], 2),
            "q_max_antes": None if "q_max" not in met else round(met["q_max"], 3),
            "q_max_despues": None if "q_max_despues" not in met else round(met["q_max_despues"], 3),
            "k_max_simultaneos": met.get("k_max_simultaneos", 0 if a.estado == ESTADO_A else None),
            "n_ref_min_observado": met.get("n_ref_min_observado"), "se_ref_max": met.get("se_ref_max"),
            "elevacion_referencia_max": met.get("elevacion_referencia_max"),
            "max_salto_uv": None if "max_salto_uv" not in met else round(met["max_salto_uv"], 2),
            "nivel_comun_uv": None if "nivel_comun_uv" not in met else round(met["nivel_comun_uv"], 2),
            "aviso": met.get("aviso", ""), "canales_invalidos": "|".join(a.canales_invalidos),
            "codigos_C": "|".join(a.codigos_C), "motivo": "|".join(a.motivos), "sha256_original": i["sha256"]})

    cols_b = list(b_rows[0].keys()) if b_rows else ["archivo", "usuario", "clase", "trial", "canal", "id_segmento", "muestra_inicial", "muestra_final", "n_muestras", "tiempo_inicial_ms", "tiempo_final_ms", "duracion_ms", "tm_inicial", "tm_final", "rms_antes_uv", "rms_despues_uv", "pico_antes_uv", "pico_despues_uv", "factor_atenuacion_min", "factor_atenuacion_mediano", "razon_q_max", "amplitud_local_max_uv", "metodo", "referencia", "referencia_uv", "canales_referencia", "criterio_activacion", "motivo", "corr_forma_original_vs_corregida"]
    cols_s = list(bs_rows[0].keys()) if bs_rows else ["archivo", "usuario", "clase", "trial", "canal", "muestra", "tm", "tiempo_ms", "valor_original", "valor_corregido", "delta", "ganancia", "razon_q", "amplitud_local_uv", "referencia_uv", "id_segmento", "criterio"]
    cols_c = ["archivo", "usuario", "clase", "trial", "canales_afectados", "pct_muestras_afectadas", "pct_por_canal", "amplitudes_detectadas", "codigos_C", "motivo", "metricas", "explicacion_no_confiable", "accion"]
    pd.DataFrame(b_rows, columns=cols_b).to_csv(os.path.join(salida, "correcciones_B.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(bs_rows, columns=cols_s).to_csv(os.path.join(salida, "correcciones_B_muestras.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(c_rows, columns=cols_c).to_csv(os.path.join(salida, "regrabar_archivos_C.csv"), index=False, encoding="utf-8-sig")
    qr = pd.DataFrame(q_rows)
    qr.to_csv(os.path.join(salida, "quality_report.csv"), index=False, encoding="utf-8-sig")

    etiq = qr[qr.clase.notna()]
    if len(etiq):
        piv = (etiq.groupby(["usuario", "subcarpeta", "clase", "clasificacion"]).size().unstack("clasificacion", fill_value=0)
               .reindex(columns=[ESTADO_A, ESTADO_B, ESTADO_C], fill_value=0).reset_index())
        piv["utilizables_A+B"] = piv[ESTADO_A] + piv[ESTADO_B]
        piv["trials_objetivo"] = trials_objetivo
        piv["faltan_por_regrabar"] = (trials_objetivo - piv["utilizables_A+B"]).clip(lower=0)
        piv.rename(columns={ESTADO_A: "n_A", ESTADO_B: "n_B", ESTADO_C: "n_C"}).sort_values(
            ["faltan_por_regrabar", "usuario", "clase"], ascending=[False, True, True]).to_csv(
            os.path.join(salida, "resumen_usuario_clase.csv"), index=False, encoding="utf-8-sig")
    cuenta = qr.clasificacion.value_counts().to_dict()
    cods: dict[str, int] = {}
    for cs in qr.codigos_C.fillna(""):
        for c in str(cs).split("|"):
            if c:
                cods[c] = cods.get(c, 0) + 1
    resumen = {"version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(),
               "captures": os.path.abspath(captures), "salida": os.path.abspath(salida),
               "usuario": usuario,
               "archivos": int(len(qr)), "A": int(cuenta.get("A", 0)), "B": int(cuenta.get("B", 0)),
               "C": int(cuenta.get("C", 0)), "codigos_C": cods,
               "segmentos_corregidos": len(b_rows), "muestras_modificadas": len(bs_rows)}
    with open(os.path.join(salida, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(salida, "parametros_usados.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.a_dict(), fh, indent=2, ensure_ascii=False)
    return resumen


# ===========================================================================
# CALIBRACIÓN con EEG real (ejecutar sobre capturas reales; no se ejecutó aquí)
# ===========================================================================
def calibrar(captures: str, salida_json: str, cfg: Config, limite: int | None = None,
             usuario: str | None = None) -> dict:
    """Mide la distribución VENTANA-A-VENTANA de ln(q) en registros sin canales inválidos
    y propone inflacion_sigma_ventana y z_deteccion. Es el dato que falta para fijar
    definitivamente los umbrales (sólo hay métricas por archivo, no por ventana)."""
    lq_all, escalas = [], []
    rutas = (
        rutas_de_usuario(captures, usuario)
        if usuario is not None else descubrir_archivos(captures)
    )
    if limite:
        rutas = rutas[:limite]
    m = np.array([cfg.escala_canal[c] for c in CANALES])
    for rel in rutas:
        i = evaluar_archivo(os.path.join(captures, rel), rel, cfg)
        if i["X"] is None or i["analisis"].codigos_C:
            continue
        X = i["X"]
        D = X - np.median(X, axis=0)
        W = cfg.ventana_muestras
        A = np.sqrt(ndimage.uniform_filter1d(D ** 2, size=W, axis=0, mode="nearest"))
        a = A / m
        ref = np.stack([np.median(np.delete(a, j, axis=1), axis=1) for j in range(len(CANALES))], axis=1)
        lq_all.append(np.log(np.maximum(A / (m * ref), 1e-9)).ravel())
        escalas.append(sigma_robusta(D) / m)
    if not lq_all:
        raise SystemExit("sin registros válidos para calibrar")
    lq = np.concatenate(lq_all)
    s_win = float(sigma_robusta(lq[:, None])[0])
    out = {"usuario": usuario, "n_ventanas": int(lq.size), "sigma_log_ventana_robusta": round(s_win, 4),
           "inflacion_sigma_ventana_sugerida": round(s_win / cfg.sigma_log_archivo, 3),
           "percentiles_ln_q": {str(p): round(float(np.percentile(lq, p)), 4) for p in (50, 90, 99, 99.9, 99.99)},
           "q_equivalente": {str(p): round(float(np.exp(np.percentile(lq, p))), 3) for p in (99, 99.9, 99.99)},
           "nota": "Fije z_deteccion de modo que q_deteccion quede por encima del percentil 99.9 de registros limpios y revise a mano una muestra de los marcados."}
    with open(salida_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def default_captures_dir() -> str:
    """Find the nearest project captures/ ancestor of this script, independent of cwd."""
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(directory, "captures")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    # Useful error path when no captures/ exists anywhere above the script.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Control de calidad EEG: A sólo se reporta, B sobrescribe el original en captures/, C sólo se reporta.")
    ap.add_argument("usuario", help="nombre exacto de la carpeta del usuario dentro de captures/")
    ap.add_argument("--captures", default=default_captures_dir(), help="raíz captures/; predeterminado: la carpeta captures/ más cercana en los ancestros del script")
    ap.add_argument("--out", default=None, help="carpeta base de reportes; se crea dentro una subcarpeta con el usuario")
    ap.add_argument("--config", help="JSON con parámetros (ver parametros_usados.json)")
    ap.add_argument("--set", action="append", default=[], metavar="clave=valor", help="sobrescribe un parámetro")
    ap.add_argument("--trials-objetivo", type=int, default=30)
    ap.add_argument("--calibrar", action="store_true", help="solo calcula la distribución ventana-a-ventana (EEG real)")
    ap.add_argument("--limite", type=int)
    ap.add_argument("--silencioso", action="store_true")
    ap.add_argument("--overwrite-reports", action="store_true", help="Compatibilidad: los reportes siempre se reemplazan; no restaura respaldos de señales B.")
    a = ap.parse_args(argv)
    d = {}

    if a.config:
        d.update(json.load(open(a.config, encoding="utf-8")))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d[k] = json.loads(v)
    cfg = Config.desde_dict(d)
    entrada = os.path.abspath(a.captures)
    salida = os.path.abspath(a.out) if a.out else os.path.dirname(os.path.abspath(__file__))
    print(f"Salida: {salida}")
    salida = carpeta_reportes_usuario(salida, a.usuario)
    if salida == entrada or salida.startswith(entrada + os.sep) or entrada.startswith(salida + os.sep):
        raise SystemExit("ERROR: captures y carpeta de reportes no pueden anidarse; el destino de reportes debe quedar fuera de captures.")
    if not os.path.isdir(entrada):
        raise SystemExit(f"ERROR: no existe {entrada}")
    if a.calibrar:
        os.makedirs(salida, exist_ok=True)
        calibration_path = os.path.join(salida, "calibracion_sugerida.json")
        print(json.dumps(calibrar(entrada, calibration_path, cfg, a.limite, a.usuario), indent=2))
        return 0
    try:
        r = procesar(entrada, salida, cfg, a.trials_objetivo, a.limite,
                     not a.silencioso, overwrite_reports=a.overwrite_reports,
                     usuario=a.usuario)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if not a.silencioso:
        n = max(r["archivos"], 1)
        print("\n--- RESUMEN ---")
        for k in "ABC":
            print(f"{k}: {r[k]:6d} ({100 * r[k] / n:.1f} %)")
        for c, v in sorted(r["codigos_C"].items(), key=lambda t: -t[1]):
            print(f"   C <- {c}: {v}")
        print(f"segmentos corregidos: {r['segmentos_corregidos']}  muestras modificadas: {r['muestras_modificadas']}\nsalidas en: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
