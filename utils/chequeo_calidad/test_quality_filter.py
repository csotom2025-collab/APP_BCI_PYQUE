"""Pruebas de quality_filter.py v1.0.0 (datos sinteticos). Ejecutar:
    python tests/test_quality_filter.py      (no requiere pytest)
"""
from __future__ import annotations
import hashlib, os, shutil, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quality_filter as qf

CAN, N = qf.CANALES, qf.N_MUESTRAS_ESPERADAS
RES = []

def ok(nombre, cond, det=""):
    RES.append(bool(cond)); print(f"  [{'PASS' if cond else 'FALLO'}] {nombre}" + (f" - {det}" if det else ""))

def eeg(semilla, n=N, sigma=8.0):
    """EEG sintetico: ruido rosado + deriva lenta, DC realista, cuantizado al LSB."""
    rng = np.random.default_rng(semilla)
    X = np.empty((n, 14))
    for j, c in enumerate(CAN):
        w = rng.normal(size=n)
        f = np.fft.rfft(w); fr = np.fft.rfftfreq(n, 1 / qf.FS); fr[0] = fr[1]
        pink = np.fft.irfft(f / np.sqrt(fr), n); pink = pink / pink.std() * sigma
        deriva = 15 * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * np.arange(n) / qf.FS + rng.uniform(0, 6))
        X[:, j] = qf.DC_REF_UV[c] + rng.normal(0, 20) + pink + deriva
    return np.round(X / qf.LSB_UV) * qf.LSB_UV

def escribir(ruta, X, canales=None, tm0=27850.0):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df = pd.DataFrame(X, columns=canales or CAN)
    df.insert(0, "Tm", tm0 + np.arange(len(X)) * 7.8125)
    df.to_csv(ruta, index=False)

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def correr(base, **kw):
    return qf.procesar(os.path.join(base, "captures"), os.path.join(base, "out"),
                       kw.get("obj", 30), None, verbose=False)

def fila(base, rel):
    m = pd.read_csv(os.path.join(base, "out", "manifiesto_archivos.csv"))
    return m[m.ruta_relativa == rel].iloc[0]

def prueba_principal():
    print("\n== Estados A / B / C ==")
    base = tempfile.mkdtemp(); cap = os.path.join(base, "captures")
    def P(n): return f"UserT/Letters/UserT_{n}.csv"
    casos = {}
    # A: limpio
    escribir(os.path.join(cap, P("A_0")), eeg(1))
    # B1: pico de 200 ms (26 muestras) y 250 uV solo en T7
    X = eeg(2); ini = 100; X[ini:ini + 26, CAN.index("T7")] += 250 * np.hanning(26)
    escribir(os.path.join(cap, P("B_0")), X); b_orig = X.copy()
    # B2: parpadeo frontal (AF3) ~250 ms, 180 uV
    X = eeg(3); X[60:92, CAN.index("AF3")] += 180 * np.hanning(32); escribir(os.path.join(cap, P("C_0")), X)
    # B3: saturacion en el riel en P8
    X = eeg(4); X[200:210, CAN.index("P8")] = qf.RIEL_SUP_UV; escribir(os.path.join(cap, P("D_0")), X)
    # B4: tramo plano de 40 muestras en O1
    X = eeg(5); X[30:70, CAN.index("O1")] = X[30, CAN.index("O1")]; escribir(os.path.join(cap, P("E_0")), X)
    # B5: nodo desconectado (DC 700 uV) pero 13 sanos
    X = eeg(6); X[:, CAN.index("F8")] = 700 + np.random.default_rng(0).normal(0, 3, N); escribir(os.path.join(cap, P("F_0")), X)
    # B6: 6 nodos muertos (quedan 8 utiles >= 7)
    X = eeg(7)
    for c in CAN[:6]: X[:, CAN.index(c)] = 4000.0
    escribir(os.path.join(cap, P("G_0")), X)
    # C1: 8 nodos muertos (quedan 6 < 7)
    X = eeg(8)
    for c in CAN[:8]: X[:, CAN.index(c)] = 4000.0
    escribir(os.path.join(cap, P("H_0")), X)
    # C2: todos los nodos con ruido enorme
    X = eeg(9, sigma=400.0); escribir(os.path.join(cap, P("I_0")), X)
    # C3: pocas muestras
    escribir(os.path.join(cap, P("J_0")), eeg(10, n=200))
    # C4: falta un canal
    X = eeg(11)[:, :13]; escribir(os.path.join(cap, P("K_0")), X, canales=CAN[:13])
    # C5: carpeta UNKNOWN / etiqueta invalida
    escribir(os.path.join(cap, "UserT/UNKNOWN/UserT_UNKNOWN_1.csv"), eeg(12))
    # C6: duplicado exacto dentro de la misma clase
    escribir(os.path.join(cap, "UserT/Numbers/UserT_7_1.csv"), eeg(13))
    shutil.copy(os.path.join(cap, "UserT/Numbers/UserT_7_1.csv"), os.path.join(cap, "UserT/Numbers/UserT_7_2.csv"))
    # C7: tiempo no monotono
    X = eeg(14); p = os.path.join(cap, "UserT/Controls/UserT_↩_0.csv"); escribir(p, X)
    d = pd.read_csv(p); d.loc[50, "Tm"] = 0; d.to_csv(p, index=False)
    # C8: clase incoherente con carpeta (letra dentro de Numbers)
    escribir(os.path.join(cap, "UserT/Numbers/UserT_A_5.csv"), eeg(15))
    hashes = {rel: sha(os.path.join(cap, rel)) for rel in qf.descubrir_archivos(cap)}
    r = correr(base)

    esperado = {P("A_0"): "A", P("B_0"): "B", P("C_0"): "B", P("D_0"): "B", P("E_0"): "B", P("F_0"): "B",
                P("G_0"): "B", P("H_0"): "C", P("I_0"): "C", P("J_0"): "C", P("K_0"): "C",
                "UserT/UNKNOWN/UserT_UNKNOWN_1.csv": "C", "UserT/Numbers/UserT_7_1.csv": "A",
                "UserT/Numbers/UserT_7_2.csv": "C", "UserT/Controls/UserT_↩_0.csv": "C",
                "UserT/Numbers/UserT_A_5.csv": "C"}
    for rel, est in esperado.items():
        f = fila(base, rel)
        ok(f"{rel.split('/')[-1]:<22} -> {est}", f.estado == est, f"obtuvo {f.estado}; {f.nodos_afectados if isinstance(f.nodos_afectados,str) else ''} {str(f.motivos)[:60]}")

    print("\n== B: solo se corta el nodo afectado ==")
    f = fila(base, P("B_0")); ok("nodos_afectados == T7", f.nodos_afectados == "T7", str(f.nodos_afectados))
    salida = pd.read_csv(os.path.join(base, "out", "B_modificados", P("B_0")))
    orig = pd.read_csv(os.path.join(cap, P("B_0")))
    for c in CAN + ["Tm"]:
        if c != "T7":
            ok(f"  {c} identico", salida[c].equals(orig[c]) or (c in CAN and np.array_equal(salida[c].values, orig[c].values)))
    nan_idx = np.where(salida["T7"].isna())[0]
    fuerte = 100 + np.where(250 * np.hanning(26) > 60)[0]      # muestras con >60 uV de pico
    ok("T7: el tramo recortado cubre todo el pico (>60 uV)", nan_idx.min() <= fuerte.min() and nan_idx.max() >= fuerte.max(), f"recorta {nan_idx.min()}-{nan_idx.max()}, pico fuerte {fuerte.min()}-{fuerte.max()}")
    ok("T7: recorte acotado (< 60 muestras)", len(nan_idx) < 60, f"{len(nan_idx)} muestras")
    ok("T7: muestras fuera del tramo intactas", np.array_equal(salida["T7"].dropna().values,
       orig["T7"].drop(index=nan_idx).values))
    ok("filas conservadas", len(salida) == len(orig))
    seg = pd.read_csv(os.path.join(base, "out", "segmentos_recortados.csv"))
    s = seg[(seg.ruta_relativa == P("B_0"))]
    ok("segmentos_recortados.csv lista el tramo con su duracion en ms", len(s) == 1 and s.iloc[0].duracion_ms > 150, s.to_dict("records").__str__()[:120])

    print("\n== Integridad del sistema ==")
    ok("A copiado byte a byte", sha(os.path.join(base, "out", "A_utilizables", P("A_0"))) == hashes[P("A_0")])
    ok("originales sin modificar", all(sha(os.path.join(cap, k)) == v for k, v in hashes.items()))
    ok("C no genera archivo", not os.path.exists(os.path.join(base, "out", "A_utilizables", P("H_0"))) and
       not os.path.exists(os.path.join(base, "out", "B_modificados", P("H_0"))))
    ok("conteos suman el total", r["A_utilizables"] + r["B_modificados"] + r["C_regrabar"] == r["archivos"])
    rg = pd.read_csv(os.path.join(base, "out", "regrabar_archivos_C.csv"))
    ok("regrabar_archivos_C.csv contiene 8 archivos", len(rg) == 8, str(len(rg)))
    shutil.rmtree(base)

def prueba_falsos_positivos():
    print("\n== Falsos positivos sobre EEG limpio sintetico (300 archivos, sigma 4-25 uV) ==")
    rng = np.random.default_rng(0); malos = 0
    for k in range(300):
        X = eeg(1000 + k, sigma=float(rng.uniform(4, 25)))
        for j, c in enumerate(CAN):
            nodo, _ = qf.analizar_nodo(X[:, j], c)
            if nodo.estado != "OK":
                malos += 1; break
    ok("<= 2 % de archivos limpios marcados", malos <= 6, f"{malos}/300")

def prueba_determinismo_y_seguridad():
    print("\n== Determinismo y seguridad ==")
    base = tempfile.mkdtemp(); cap = os.path.join(base, "captures")
    X = eeg(21); X[50:76, 4] += 300 * np.hanning(26); escribir(os.path.join(cap, "U/Letters/U_A_0.csv"), X)
    correr(base); h1 = sha(os.path.join(base, "out", "manifiesto_archivos.csv")); m1 = sha(os.path.join(base, "out", "B_modificados/U/Letters/U_A_0.csv"))
    correr(base); h2 = sha(os.path.join(base, "out", "manifiesto_archivos.csv")); m2 = sha(os.path.join(base, "out", "B_modificados/U/Letters/U_A_0.csv"))
    ok("dos corridas -> salidas identicas", h1 == h2 and m1 == m2)
    try:
        qf.main(["--captures", cap, "--out", os.path.join(cap, "dentro"), "--silencioso"]); ok("salida dentro de entrada rechazada", False)
    except SystemExit:
        ok("salida dentro de entrada rechazada", True)
    shutil.rmtree(base)

if __name__ == "__main__":
    prueba_principal(); prueba_falsos_positivos(); prueba_determinismo_y_seguridad()
    print(f"\n{sum(RES)}/{len(RES)} pruebas OK"); sys.exit(0 if all(RES) else 1)
