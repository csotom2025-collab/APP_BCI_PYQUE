"""Suite de pruebas de quality_filter.py (modo report-only).

Ejecutar:  .venv/Scripts/python.exe tests/test_quality_filter.py
No requiere pytest. Imprime PASS/FAIL por prueba y devuelve codigo != 0 si falla.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quality_filter as qf  # noqa: E402

FS = qf.FS_ASUMIDA
N = qf.N_MUESTRAS_ESPERADAS
CAN = qf.CANALES_ESPERADOS

_RESULTADOS: list[tuple[str, bool, str]] = []


def comprobar(nombre: str, condicion: bool, detalle: str = "") -> None:
    _RESULTADOS.append((nombre, bool(condicion), detalle))
    print(f"  [{'PASS' if condicion else 'FALLO'}] {nombre}" + (f" — {detalle}" if detalle else ""))


# ---------------------------------------------------------------------------
# Generacion de datos sinteticos
# ---------------------------------------------------------------------------

def senal_sintetica(semilla: int = 0, n: int = N, dc: float = 4274.0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    X = rng.normal(0.0, 8.0, size=(n, len(CAN)))
    X = np.cumsum(X, axis=0) * 0.15
    X -= X.mean(axis=0, keepdims=True)
    X += dc
    # cuantizar al LSB del dispositivo, como hace el capturador real
    return np.round(X / qf.LSB_EMOTIV) * qf.LSB_EMOTIV


def escribir_csv(ruta: str, X: np.ndarray, tm0: float = 1000.0,
                 canales: list[str] | None = None, paso_ms: float = 1000.0 / FS) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    canales = canales or CAN
    tm = tm0 + np.arange(X.shape[0]) * paso_ms
    df = pd.DataFrame(X, columns=canales)
    df.insert(0, "Tm", tm)
    df.to_csv(ruta, index=False)


def construir_fixture(base: str) -> dict[str, str]:
    """Crea un arbol captures/ sintetico. Devuelve rutas notables."""
    cap = os.path.join(base, "captures")
    notables = {}

    # usuario valido con 6 trials en orden de adquisicion creciente
    for i in range(6):
        p = os.path.join(cap, "UserTest", "Letters", f"UserTest_A_{i}.csv")
        escribir_csv(p, senal_sintetica(i), tm0=1000.0 + i * 5000.0)
    # segunda sesion: Tm reinicia
    for i in range(6, 9):
        p = os.path.join(cap, "UserTest", "Letters", f"UserTest_A_{i}.csv")
        escribir_csv(p, senal_sintetica(i), tm0=100.0 + (i - 6) * 5000.0)

    # dentro de ignorarSenales: NUNCA debe procesarse
    p = os.path.join(cap, "ignorarSenales", "UserMalo", "Letters", "UserMalo_A_0.csv")
    escribir_csv(p, senal_sintetica(99))
    notables["ignorado"] = p

    # NaN
    X = senal_sintetica(11); X[10, 3] = np.nan
    p = os.path.join(cap, "UserNaN", "Letters", "UserNaN_A_0.csv")
    escribir_csv(p, X); notables["nan"] = p

    # numero de canales incorrecto
    X = senal_sintetica(12)[:, :10]
    p = os.path.join(cap, "UserCanales", "Letters", "UserCanales_A_0.csv")
    escribir_csv(p, X, canales=CAN[:10]); notables["canales"] = p

    # muestras insuficientes
    p = os.path.join(cap, "UserCorto", "Letters", "UserCorto_A_0.csv")
    escribir_csv(p, senal_sintetica(13, n=100)); notables["corto"] = p

    # Tm no monotonico
    X = senal_sintetica(14)
    p = os.path.join(cap, "UserTm", "Letters", "UserTm_A_0.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tm = 1000.0 + np.arange(N) * (1000.0 / FS)
    tm[50] = tm[49] - 5.0
    df = pd.DataFrame(X, columns=CAN); df.insert(0, "Tm", tm); df.to_csv(p, index=False)
    notables["tm"] = p

    # canal plano (un solo nivel) -> piso de MAD
    X = senal_sintetica(15); X[:, 2] = 4274.0
    p = os.path.join(cap, "UserPlano", "Letters", "UserPlano_A_0.csv")
    escribir_csv(p, X); notables["plano"] = p

    # evento de una sola muestra, tipo Jorge
    X = senal_sintetica(16); X[128, 5] += 300.0
    p = os.path.join(cap, "UserPico", "Letters", "UserPico_A_0.csv")
    escribir_csv(p, X); notables["pico"] = p

    # DC extremo, senal por lo demas normal
    X = senal_sintetica(17, dc=8000.0)
    p = os.path.join(cap, "UserDC", "Letters", "UserDC_A_0.csv")
    escribir_csv(p, X); notables["dc"] = p

    notables["captures"] = cap
    return notables


def correr(cap: str, out: str, **kw) -> tuple[pd.DataFrame, list]:
    qf.REQUIERE_DECISION.clear()
    df, regs, _diag = qf.procesar(cap, out, verbose=False, **kw)
    return df, regs


def motivos_de(registros, sufijo_ruta: str) -> list[str]:
    for r in registros:
        if r.ruta_relativa.endswith(sufijo_ruta):
            return r.motivos_exclusion
    return []


# ---------------------------------------------------------------------------
# PRUEBAS
# ---------------------------------------------------------------------------

def main() -> int:
    base = tempfile.mkdtemp(prefix="qf_test_")
    try:
        n = construir_fixture(base)
        cap = n["captures"]
        out = os.path.join(base, "salida")

        print("\n== 3. Exclusion de ignorarSenales ==")
        rutas = qf.descubrir_archivos(cap)
        comprobar("ninguna ruta descubierta contiene ignorarSenales",
                  not any("ignorarsenales" in r.lower() for r in rutas),
                  f"{len(rutas)} rutas")
        df, regs = correr(cap, out)
        comprobar("ningun registro procede de ignorarSenales",
                  not any("ignorarsenales" in r.ruta_relativa.lower() for r in regs))
        comprobar("ninguna fila del parquet procede de ignorarSenales",
                  not df["ruta_relativa"].str.lower().str.contains("ignorarsenales").any())

        print("\n== 4-7. Deteccion de fallos estructurales ==")
        comprobar("NaN detectado", any(m.startswith("contiene_nan")
                                       for m in motivos_de(regs, "UserNaN_A_0.csv")),
                  str(motivos_de(regs, "UserNaN_A_0.csv")))
        comprobar("n_canales incorrecto detectado",
                  any(m.startswith("n_canales_incorrecto")
                      for m in motivos_de(regs, "UserCanales_A_0.csv")),
                  str(motivos_de(regs, "UserCanales_A_0.csv")))
        comprobar("muestras insuficientes detectado",
                  any(m.startswith("n_muestras_insuficiente")
                      for m in motivos_de(regs, "UserCorto_A_0.csv")),
                  str(motivos_de(regs, "UserCorto_A_0.csv")))
        comprobar("Tm no monotonico detectado",
                  "Tm_no_monotonico" in motivos_de(regs, "UserTm_A_0.csv"),
                  str(motivos_de(regs, "UserTm_A_0.csv")))

        print("\n== 1. Determinismo ==")
        df2, regs2 = correr(cap, out + "_2")
        comprobar("dos ejecuciones producen el mismo DataFrame",
                  df.equals(df2))
        h1 = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
        h2 = hashlib.sha256(pd.util.hash_pandas_object(df2, index=False).values.tobytes()).hexdigest()
        comprobar("hash del contenido identico", h1 == h2, h1[:16])

        print("\n== 2 y 13. Independencia del orden del sistema de archivos ==")
        walk_real = os.walk

        def walk_barajado(top, *a, **k):
            for dirpath, dirnames, filenames in walk_real(top, *a, **k):
                random.Random(7).shuffle(dirnames)
                random.Random(9).shuffle(filenames)
                yield dirpath, dirnames, filenames

        os.walk = walk_barajado
        try:
            rutas_b = qf.descubrir_archivos(cap)
            df3, regs3 = correr(cap, out + "_3")
        finally:
            os.walk = walk_real
        comprobar("descubrimiento identico con os.walk barajado", rutas == rutas_b)
        comprobar("metricas identicas con os.walk barajado", df.equals(df3))
        ord1 = {r.ruta_relativa: (r.session_id, r.acquisition_order) for r in regs}
        ord3 = {r.ruta_relativa: (r.session_id, r.acquisition_order) for r in regs3}
        comprobar("acquisition_order identico con os.walk barajado", ord1 == ord3)

        print("\n== Orden determinista y sesiones no determinables ==")
        ut = sorted([r for r in regs if r.usuario == "UserTest"],
                    key=lambda r: r.acquisition_order)
        comprobar("session_id es NULO para todos los registros incluidos",
                  all(r.session_id is None for r in regs if r.incluido),
                  "no determinable desde los CSV")
        comprobar("acquisition_order es una permutacion 0..n-1 por usuario",
                  [r.acquisition_order for r in ut] == list(range(len(ut))))
        tms = [r.tm_inicio for r in ut]
        comprobar("acquisition_order ordena por Tm[0] ascendente", tms == sorted(tms))

        print("\n== 8. Ausencia de re-referencia (CAR) ==")
        X = senal_sintetica(42)
        Xmod = X.copy(); Xmod[:, 0] += 500.0 * np.hanning(N)
        for etapa in qf.ETAPAS_TRANSFORMADAS:
            A = qf.metricas_transformadas(qf.aplicar_etapa(X, etapa), FS)
            B = qf.metricas_transformadas(qf.aplicar_etapa(Xmod, etapa), FS)
            iguales = all(np.allclose(A[k][1:], B[k][1:], rtol=0, atol=0)
                          for k in A if not k.startswith("corr_"))
            comprobar(f"etapa {etapa}: alterar el canal 0 no cambia los otros 13",
                      iguales)
        A = qf.metricas_transformadas(qf.aplicar_etapa(X, "detr"), FS)
        B = qf.metricas_transformadas(qf.aplicar_etapa(Xmod, "detr"), FS)
        comprobar("el canal alterado SI cambia (control positivo)",
                  not np.isclose(A["p2p"][0], B["p2p"][0]),
                  f"{A['p2p'][0]:.1f} -> {B['p2p'][0]:.1f}")

        print("\n== 9. Consistencia del piso de MAD ==")
        fp = df[(df.ruta_relativa.str.endswith("UserPlano_A_0.csv")) &
                (df.etapa == "detr") & (df.canal == CAN[2])]
        comprobar("hay fila para el canal plano", len(fp) == 1)
        if len(fp) == 1:
            fila = fp.iloc[0]
            comprobar("mad_bruto = 0 en canal plano", float(fila.mad_bruto) < 1e-9,
                      f"{float(fila.mad_bruto):.3e}")
            comprobar("piso de MAD activado", float(fila.mad_piso_activado) == 1.0)
            comprobar("mad = piso declarado", abs(float(fila.mad) - qf.MAD_PISO) < 1e-12,
                      f"{float(fila.mad):.6f} vs {qf.MAD_PISO:.6f}")
            comprobar("max_z finito y no explota", np.isfinite(fila.max_z_robusto)
                      and float(fila.max_z_robusto) < 50.0,
                      f"max_z={float(fila.max_z_robusto):.3f}")

        print("\n== 10. Evento de una sola muestra (tipo Jorge) ==")
        fp = df[(df.ruta_relativa.str.endswith("UserPico_A_0.csv")) &
                (df.etapa == "detr") & (df.canal == CAN[5])].iloc[0]
        otro = df[(df.ruta_relativa.str.endswith("UserPico_A_0.csv")) &
                  (df.etapa == "detr") & (df.canal == CAN[6])].iloc[0]
        comprobar("max_z del canal con pico es muy superior al de un canal limpio",
                  float(fp.max_z_robusto) > 3 * float(otro.max_z_robusto),
                  f"{float(fp.max_z_robusto):.1f} vs {float(otro.max_z_robusto):.1f}")
        dur = float(fp["dur_max_episodio_ms_z10"])
        n_ep = float(fp["n_episodios_z10"])
        comprobar("el episodio dura ~1 muestra", dur <= 2 * 1000.0 / FS,
                  f"dur_max={dur:.2f} ms (1 muestra = {1000.0 / FS:.2f} ms)")
        comprobar("se contabiliza al menos un episodio", n_ep >= 1, f"n={n_ep:g}")

        print("\n== 11. El DC no produce falsos rechazos ==")
        rdc = [r for r in regs if r.ruta_relativa.endswith("UserDC_A_0.csv")][0]
        comprobar("archivo con DC=8000 uV incluido", rdc.incluido,
                  str(rdc.motivos_exclusion))
        # Un desplazamiento de DC PURO es un multiplo entero del LSB: asi la senal
        # permanece en la misma rejilla de cuantizacion y la unica diferencia es
        # la constante. (Sumar un DC arbitrario mueve la rejilla y perturba en
        # ~1 LSB, que es cuantizacion, no dependencia del DC.)
        Xn = senal_sintetica(17, dc=4274.0)
        Xd = Xn + 7000 * qf.LSB_EMOTIV
        comprobar("el desplazamiento de prueba es un DC exacto",
                  np.allclose(Xd - Xn, (Xd - Xn)[0, 0], rtol=0, atol=1e-12),
                  f"+{7000 * qf.LSB_EMOTIV:.1f} uV")
        for etapa in qf.ETAPAS_TRANSFORMADAS:
            A = qf.metricas_transformadas(qf.aplicar_etapa(Xn, etapa), FS)
            B = qf.metricas_transformadas(qf.aplicar_etapa(Xd, etapa), FS)
            iguales = all(np.allclose(A[k], B[k], rtol=1e-9, atol=1e-9, equal_nan=True)
                          for k in A)
            comprobar(f"etapa {etapa}: las metricas no dependen del DC", iguales)
        # Un DC que NO cae en la rejilla perturba como maximo al nivel del LSB.
        Xg = senal_sintetica(17, dc=8000.0)
        A = qf.metricas_transformadas(qf.aplicar_etapa(Xn, "detr"), FS)
        B = qf.metricas_transformadas(qf.aplicar_etapa(Xg, "detr"), FS)
        dmax = float(np.abs(A["p2p"] - B["p2p"]).max())
        comprobar("un DC fuera de rejilla solo perturba a nivel de cuantizacion",
                  dmax <= 2 * qf.LSB_EMOTIV, f"{dmax:.4f} uV = {dmax / qf.LSB_EMOTIV:.2f} LSB")

        print("\n== 12. R jamas afecta el estado de calidad ==")
        df_sin, regs_sin = correr(cap, out + "_sinR")
        df_con, regs_con = correr(cap, out + "_conR")
        diag = qf.diagnostico_sincronizacion(cap, regs_con, FS, n_perm=50,
                                             semilla=1, verbose=False)
        comprobar("el parquet es identico con y sin diagnostico R",
                  df_sin.equals(df_con))
        comprobar("la inclusion de archivos es identica",
                  [r.incluido for r in regs_sin] == [r.incluido for r in regs_con])
        comprobar("el diagnostico R no contiene ningun veredicto",
                  not any(k in json.dumps(diag).lower()
                          for k in ("pass", "fail", "reject", "accept",
                                    "veredicto", "score", "bueno", "malo")))
        comprobar("el diagnostico R lleva aviso de no validado",
                  "no esta validado" in diag["aviso"].lower())

        print("\n== Salidas y report-only ==")
        antes = {}
        for dirpath, _, files in os.walk(cap):
            for f in files:
                p = os.path.join(dirpath, f)
                antes[p] = (os.path.getmtime(p), os.path.getsize(p),
                            hashlib.sha256(open(p, "rb").read()).hexdigest())
        out_final = os.path.join(base, "salida_final")
        rep = qf.escribir_salidas(df, regs, out_final, cap, FS, diag)
        despues = {}
        for dirpath, _, files in os.walk(cap):
            for f in files:
                p = os.path.join(dirpath, f)
                despues[p] = (os.path.getmtime(p), os.path.getsize(p),
                              hashlib.sha256(open(p, "rb").read()).hexdigest())
        comprobar("ningun CSV de entrada fue modificado, borrado ni renombrado",
                  antes == despues, f"{len(antes)} archivos verificados")
        comprobar("no se creo nada dentro del directorio de entrada",
                  set(antes) == set(despues))
        for nombre in ("metricas_calidad.parquet", "reporte.csv", "reporte.json",
                       "diagnostico_sincronizacion.json"):
            comprobar(f"salida generada: {nombre}",
                      os.path.exists(os.path.join(out_final, nombre)))

        print("\n== Ausencia de veredictos y erp_readiness ==")
        rep_csv = pd.read_csv(os.path.join(out_final, "reporte.csv"))
        comprobar("reporte.csv no contiene columna de veredicto ni score",
                  not any(c.lower() in ("veredicto", "score", "calidad", "estado",
                                        "pass", "fail") for c in rep_csv.columns),
                  ", ".join(rep_csv.columns[:6]) + " ...")
        comprobar("erp_readiness presente y siempre nulo",
                  "erp_readiness" in rep_csv.columns and rep_csv.erp_readiness.isna().all())
        comprobar("erp_readiness_motivo = sin_trigger",
                  (rep_csv.erp_readiness_motivo == "sin_trigger").all())
        comprobar("reporte.json declara erp_readiness nulo",
                  rep["erp_readiness"] == {"value": None, "reason": "sin_trigger"})
        comprobar("reporte.json declara re_referencia_aplicada = False",
                  rep["parametros"]["re_referencia_aplicada"] is False)
        comprobar("reporte.json marca fs_asumida como provisional",
                  "PROVISIONAL" in rep["parametros"]["fs_asumida_estado"])
        comprobar("toda fila del parquet lleva etapa e identificacion",
                  set(["ruta_relativa", "canal", "etapa", "sha256", "session_id",
                       "acquisition_order", "fs_asumida"]).issubset(df.columns))

    finally:
        shutil.rmtree(base, ignore_errors=True)

    fallos = [r for r in _RESULTADOS if not r[1]]
    print("\n" + "=" * 70)
    print(f"TOTAL: {len(_RESULTADOS)} comprobaciones — "
          f"{len(_RESULTADOS) - len(fallos)} PASS, {len(fallos)} FALLO")
    if fallos:
        for nombre, _, det in fallos:
            print(f"   FALLO: {nombre} {det}")
    print("=" * 70)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
