"""
P300 Speller Classifier Pipeline
=================================
Estructura esperada de archivos:
  <root>/
    Letters/
      User{name}_{cmd}_{trial}.csv
    Numbers/
      User{name}_{cmd}_{trial}.csv
    Controls/
      User{name}_{cmd}_{trial}.csv

Cada CSV: señal EEG cruda, 500 muestras x 16 canales + columna 'Tm'
Naming: el índice {cmd} define el target. Todos los trials del mismo {cmd}
son TARGET (label=1). Los de distinto {cmd} dentro del mismo tipo son NON-TARGET (label=0).
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis
import pywt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
FS          = 250          # Hz — frecuencia de muestreo ADS1299
N_SAMPLES   = 500          # muestras por trial (2 seg)
BASELINE_MS = 500          # ms de pre-estímulo para corrección de baseline
BP_LOW      = 0.5          # Hz — límite inferior bandpass
BP_HIGH     = 30.0         # Hz — límite superior bandpass
DOWNSAMPLE  = 4            # factor: 250 Hz → 62.5 Hz después de filtrar
N_FOLDS     = 5            # folds para cross-validation

EEG_COLS = ["Oz","Po7","Po4","Po3","P4","P3","Po8","Pz",
            "Fz","F2","F3","F4","AF3","Cz","AF4","F1"]

FEATURE_SETS = [
    "Estadisticas","Estadisticas_tm", "Frecuencias_Abs", "Frecuencias_Rel",
    "Frecuencias_Est", "Wavelets","welch", "Frecuencias_Todas", "TODAS"
]

CATEGORY_DIRS = ["Letters", "Numbers", "Controls"]


# ─────────────────────────────────────────────
#  1. CARGA DE DATOS
# ─────────────────────────────────────────────
def parse_filename(fname):
    """
    Extrae (user, cmd_idx, trial_idx) de 'User{name}_{cmd}_{trial}.csv'
    cmd_idx puede ser numérico (0,1,2) o alfanumérico (A,B,C,1,2,space,back,enter…)
    trial_idx siempre es numérico.
    Retorna None si el nombre no coincide con el patrón.
    """
    # Patrón: User<name>_<cmd>_<trial>.csv
    # <cmd> puede ser letras, números o guiones/palabras (ej: 'space', 'back', 'A')
    # <trial> es siempre numérico al final
    m = re.match(r"User(.+)_([A-Za-z0-9_\-─⟵↩]+)_(\d+)\.csv", fname)
    if not m:
        return None
    cmd_raw = m.group(2)
    # Intentar convertir a int si es puramente numérico
    try:
        cmd_idx = int(cmd_raw)
    except ValueError:
        cmd_idx = cmd_raw          # Mantener como string (ej: 'A', 'space')
    return m.group(1), cmd_idx, int(m.group(3))


def load_category(category_path):
    """
    Carga todos los CSVs de una carpeta de categoría.
    Retorna dict: {cmd_idx: [DataFrame, ...]}  (una lista de trials por comando)
    """
    category_path = Path(category_path)
    commands = defaultdict(list)

    csv_files = sorted(category_path.glob("*.csv"))
    if not csv_files:
        print(f"  [!] Sin archivos CSV en {category_path}")
        return commands

    for fpath in csv_files:
        parsed = parse_filename(fpath.name)
        if parsed is None:
            print(f"  [!] Nombre no reconocido, omitiendo: {fpath.name}")
            continue
        _, cmd_idx, _ = parsed
        try:
            df = pd.read_csv(fpath)
            # Verificar que tenga las columnas EEG esperadas
            missing = [c for c in EEG_COLS if c not in df.columns]
            if missing:
                print(f"  [!] Columnas faltantes en {fpath.name}: {missing}")
                continue
            commands[cmd_idx].append(df)
        except Exception as e:
            print(f"  [!] Error leyendo {fpath.name}: {e}")

    n_cmds  = len(commands)
    n_trials = sum(len(v) for v in commands.values())
    print(f"  → {n_cmds} comandos, {n_trials} trials cargados de {category_path.name}")
    return commands


# ─────────────────────────────────────────────
#  2. PREPROCESAMIENTO DE SEÑAL
# ─────────────────────────────────────────────
def bandpass_filter(signal, lowcut=BP_LOW, highcut=BP_HIGH, fs=FS, order=4):
    """Filtro Butterworth pasabanda."""
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


def baseline_correct(signal, n_baseline=None):
    """
    Sustrae la media de los primeros n_baseline muestras (pre-estímulo).
    Por defecto usa BASELINE_MS ms.
    """
    if n_baseline is None:
        n_baseline = int(BASELINE_MS * FS / 1000)
    baseline = signal[:n_baseline].mean(axis=0)
    return signal - baseline


def downsample_signal(signal, factor=DOWNSAMPLE):
    """Submuestreo simple tras filtrado (el filtro previene aliasing)."""
    return signal[::factor]


# En p300_classifier.py — agregar estas constantes:
STIM_MS       = 500   # onset del estímulo (ms)
P300_END_MS   = 1200  # fin de la ventana de interés (ms)
POST_END_MS   = 2000  # fin del trial completo (ms)

# Índices en muestras originales (antes de downsample):
STIM_SAMPLE   = int(STIM_MS    * FS / 1000)   # 125
P300_END_SAMPLE = int(P300_END_MS * FS / 1000) # 300

def preprocess_trial(df, use_erp_window=True):
    raw = df[EEG_COLS].values.astype(np.float64)
    
    # 1. Baseline correction usando pre-estímulo (muestras 0-125)
    sig = baseline_correct(raw)
    
    # 2. Filtrado sobre la señal completa (para evitar artefactos de borde)
    sig = bandpass_filter(sig)
    
    # 3. Recortar a ventana de respuesta evocada (0.5s – 1.2s)
    if use_erp_window:
        sig = sig[STIM_SAMPLE:P300_END_SAMPLE]  # solo 175 muestras → 44 tras downsample
    
    # 4. Downsample
    sig = downsample_signal(sig)
    return sig


# ─────────────────────────────────────────────
#  3. EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),   # nota: limitado por BP_HIGH=30; gamma será ~0
}


def _safe_filtfilt(b, a, signal):
    """filtfilt con padlen adaptativo: nunca falla por señal corta."""
    padlen = 3 * (max(len(a), len(b)) - 1)
    n_samples = signal.shape[0]
    if n_samples <= padlen:
        # Señal demasiado corta: bajar orden hasta que quepa
        for order in [2, 1]:
            b2, a2 = butter(order,
                            [float(b[0]), float(b[-1])] if len(b) > 2 else b,
                            btype='band') if False else (b, a)
            # Recalcular con orden reducido no es trivial aquí;
            # en su lugar usar sosfilt que no necesita padlen explícito
            break
        from scipy.signal import sosfilt, sosfiltfilt
        try:
            sos = butter(2,
                         None,   # placeholder; se llama desde band_power con parámetros
                         output='sos')
        except Exception:
            pass
        # Fallback seguro: devolver varianza de la señal original por canal
        return signal
    return filtfilt(b, a, signal, axis=0)


def band_power(signal, band, fs):
    """Potencia absoluta en banda. Usa sosfiltfilt para ser robusto ante señales cortas."""
    from scipy.signal import sosfiltfilt
    low, high = band
    nyq = fs / 2.0
    if high >= nyq:
        high = nyq * 0.99
    if low <= 0:
        low = 0.01
    n_samples = signal.shape[0]
    # Elegir orden máximo que quepa en la señal: padlen = 3*(2*order)
    for order in [4, 3, 2, 1]:
        padlen = 3 * (2 * order)   # sosfiltfilt usa 3*(2*order) aprox
        if n_samples > padlen:
            break
    try:
        sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
        filtered = sosfiltfilt(sos, signal, axis=0)
    except Exception:
        filtered = signal           # fallback: varianza de señal sin filtrar
    return np.var(filtered, axis=0)


def extract_features(signal):
    """
    Extrae los 7 conjuntos de features de un trial preprocesado.
    signal: ndarray (n_samples_downsampled, n_channels)

    Retorna dict con keys = nombre del set, values = ndarray 1D de features.
    """
    fs_ds = FS / DOWNSAMPLE             # 62.5 Hz tras downsampling
    n_ch  = signal.shape[1]
    ch_names = EEG_COLS

    features = {}

    # ── Estadísticas temporales ──────────────────────────────────────────
    feat_est = []
    for i in range(n_ch):
        ch = signal[:, i]
        feat_est.extend([
            np.mean(ch),
            np.std(ch),
            np.var(ch),
            np.sqrt(np.mean(ch**2)),            # RMS
            skew(ch),
            kurtosis(ch),
        ])
    features["Estadisticas_tm"] = np.array(feat_est)

    # ── Potencia absoluta por banda ──────────────────────────────────────
    feat_abs = []
    for band_name, band_range in BANDS.items():
        bp = band_power(signal, band_range, fs_ds)
        feat_abs.extend(bp.tolist())
    features["Frecuencias_Abs"] = np.array(feat_abs)

    # ── Potencia relativa por banda ──────────────────────────────────────
    total_power = np.var(signal, axis=0) + 1e-12
    feat_rel = []
    for band_name, band_range in BANDS.items():
        bp = band_power(signal, band_range, fs_ds)
        feat_rel.extend((bp / total_power).tolist())
    features["Frecuencias_Rel"] = np.array(feat_rel)

    # ── Estadísticas por banda ───────────────────────────────────────────
    feat_fest = []
    for band_name, band_range in BANDS.items():
        from scipy.signal import sosfiltfilt
        nyq = fs_ds / 2.0
        low_b  = max(band_range[0], 0.01)
        high_b = min(band_range[1], nyq * 0.99)
        n_samples = signal.shape[0]
        for order in [4, 3, 2, 1]:
            padlen = 3 * (2 * order)
            if n_samples > padlen:
                break
        try:
            sos = butter(order, [low_b / nyq, high_b / nyq],
                         btype='band', output='sos')
            band_sig = sosfiltfilt(sos, signal, axis=0)
        except Exception:
            band_sig = signal
        for i in range(n_ch):
            ch = band_sig[:, i]
            feat_fest.extend([
                np.mean(ch),
            ])
        if band_name in ["beta", "gamma"]:
            for i in range(n_ch):
                        ch = band_sig[:, i]
                        feat_fest.extend([
                            np.std(ch),
                            np.var(ch),
                            np.sqrt(np.mean(ch**2)),
                            skew(ch),
                            kurtosis(ch),
                        ])
    features["Frecuencias_Est"] = np.array(feat_fest)

        # ── Wavelets (DWT) ───────────────────────────────────────────────────
    # Nivel máximo factible: floor(log2(len(signal))) = 5 requiere al menos 32 muestras
    feat_wav = []
    if n_samples < 32:
        # No se puede hacer DWT nivel 5: 4 estadísticos * (1 cA5 + 5 cD) = 24 por canal
        feat_wav.extend([0.0] * (4 * (5+1) * n_ch))
    else:
        for i in range(n_ch):
            ch = signal[:, i]
            coeffs = pywt.wavedec(ch, 'db4', level=5)
            cA5 = coeffs[0]
            feat_wav.extend([np.mean(cA5), np.std(cA5), np.var(cA5),
                             np.sqrt(np.mean(cA5**2))])
            for cD in coeffs[1:]:
                feat_wav.extend([np.mean(cD), np.std(cD), np.var(cD),
                                 np.sqrt(np.mean(cD**2))])
    features["Wavelets"] = np.array(feat_wav)

    # ── Frecuencias_Todas = Abs + Rel + Est concatenadas ─────────────────
    features["Frecuencias_Todas"] = np.concatenate([
        features["Frecuencias_Abs"],
        features["Frecuencias_Rel"],
        features["Frecuencias_Est"],
    ])
        # ── Welch (potencia en Beta y Gamma) ──────────────────────────────────────
    # La señal ya está filtrada entre 0.5 y 30 Hz. Calculamos PSD con Welch
    # y extraemos estadísticos únicamente en la banda de interés (13-30 Hz).
    feat_welch = []
    low_welch, high_welch = 13.0, 30.0   # dentro del límite del filtro original
    
    # Parámetros adaptativos según longitud de la señal
    n_samples = signal.shape[0]
    # Longitud mínima para calcular Welch: al menos 4 muestras por segmento
    # y se necesita al menos 2 segmentos (no es estricto, pero para evitar errores)
    if n_samples < 8:
        # Señal demasiado corta: rellenar con ceros
        feat_welch.extend([0.0] * (6 * n_ch))
    else:
        # Calcular PSD para cada canal
        for i in range(n_ch):
            ch = signal[:, i]
            # Elegir nperseg: no mayor que la señal, mínimo 4, preferible potencia de 2
            nperseg = min(256, len(ch) // 2) if len(ch) > 32 else len(ch)
            if nperseg < 4:
                feat_welch.extend([0.0] * 6)
                continue
            try:
                f_welch, Pxx = welch(ch, fs=fs_ds, nperseg=nperseg, axis=-1)
                # Seleccionar solo frecuencias dentro de [13, 30] Hz
                band_mask = (f_welch >= low_welch) & (f_welch <= high_welch)
                if not np.any(band_mask):
                    feat_welch.extend([0.0] * 6)
                    continue
                Pxx_band = Pxx[band_mask]
                # Estadísticos del PSD en la banda
                feat_welch.extend([
                    float(np.mean(Pxx_band)),
                    float(np.std(Pxx_band)),
                    float(np.var(Pxx_band)),
                    float(np.sqrt(np.mean(Pxx_band**2))),
                    float(kurtosis(Pxx_band)),
                    float(skew(Pxx_band)),
                ])
            except Exception:
                feat_welch.extend([0.0] * 6)
    
    features["welch"] = np.array(feat_welch)   
    
    features["Estadisticas"] = np.concatenate([
        features["Estadisticas_tm"],
        features["Frecuencias_Est"],
        features["welch"]
    ])
    # ── TODAS ─────────────────────────────────────────────────────────────
    features["TODAS"] = np.concatenate([
        features["Estadisticas_tm"],
        features["Frecuencias_Todas"],
        features["Wavelets"],
        features["welch"]
    ])
    return features


# ─────────────────────────────────────────────
#  4. CONSTRUCCIÓN DEL DATASET BINARIO
# ─────────────────────────────────────────────
def build_dataset(commands_dict, feature_set="TODAS"):
    """
    A partir de {cmd_idx: [df, ...]} construye X, y para clasificación binaria.

    Estrategia de labels:
      - Cada trial de cmd_i  → TARGET (1)    para el clasificador del cmd_i
      - Cada trial de cmd_j≠i → NON-TARGET (0)

    Con N comandos y T trials/comando:
      Muestras positivas: N*T  (cada trial es target de su propio comando)
      Muestras negativas: N*(N-1)*T  (cada trial es non-target de los otros)

    Para evitar un dataset gigantesco con muchos comandos, sub-muestreamos
    non-targets (max_neg_ratio):
    """
    MAX_NEG_RATIO = 5           # máx 5 non-targets por cada target

    all_cmd_idxs = sorted(commands_dict.keys())
    n_commands   = len(all_cmd_idxs)

    if n_commands < 2:
        print("  [!] Se necesitan al menos 2 comandos para entrenar.")
        return None, None, None

    print(f"  → Comandos detectados: {n_commands}  ({all_cmd_idxs})")

    X_pos, X_neg = [], []

    for target_cmd in all_cmd_idxs:
        target_trials = commands_dict[target_cmd]

        # Features de trials TARGET
        for df in target_trials:
            sig  = preprocess_trial(df)
            feat = extract_features(sig)[feature_set]
            X_pos.append(feat)

        # Features de trials NON-TARGET (de otros comandos)
        neg_pool = []
        for other_cmd in all_cmd_idxs:
            if other_cmd == target_cmd:
                continue
            for df in commands_dict[other_cmd]:
                sig  = preprocess_trial(df)
                feat = extract_features(sig)[feature_set]
                neg_pool.append(feat)

        # Sub-muestreo para balancear ratio
        max_neg = len(target_trials) * MAX_NEG_RATIO
        if len(neg_pool) > max_neg:
            idx = np.random.choice(len(neg_pool), max_neg, replace=False)
            neg_pool = [neg_pool[i] for i in idx]
        X_neg.extend(neg_pool)

    X = np.array(X_pos + X_neg)
    y = np.array([1]*len(X_pos) + [0]*len(X_neg))

    print(f"  → Dataset: {len(X_pos)} targets | {len(X_neg)} non-targets | "
          f"{X.shape[1]} features")
    return X, y, all_cmd_idxs


# ─────────────────────────────────────────────
#  5. CLASIFICADORES
# ─────────────────────────────────────────────
def get_classifiers():
    return {
        "LDA_shrinkage": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                        C=1.0, gamma="scale")),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.1)),
        ]),
    }


def evaluate_classifiers(X, y, feature_set_name, category_name):
    """
    Evalúa todos los clasificadores con cross-validation estratificada.
    Retorna DataFrame con resultados.
    """
    results = []
    clfs    = get_classifiers()
    sw      = compute_sample_weight("balanced", y)
    cv      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for clf_name, pipeline in clfs.items():
        try:
            scores_acc = cross_val_score(pipeline, X, y, cv=cv,
                                         scoring="accuracy", n_jobs=-1)
            scores_auc = cross_val_score(pipeline, X, y, cv=cv,
                                         scoring="roc_auc", n_jobs=-1)
            results.append({
                "Categoria":    category_name,
                "Feature_Set":  feature_set_name,
                "Clasificador": clf_name,
                "Acc_mean":     scores_acc.mean(),
                "Acc_std":      scores_acc.std(),
                "AUC_mean":     scores_auc.mean(),
                "AUC_std":      scores_auc.std(),
                "N_samples":    len(y),
                "N_targets":    int(y.sum()),
                "N_nontargets": int((y == 0).sum()),
            })
            print(f"    [{clf_name:20s}] Acc={scores_acc.mean():.3f}±{scores_acc.std():.3f} "
                  f"AUC={scores_auc.mean():.3f}±{scores_auc.std():.3f}")
        except Exception as e:
            print(f"    [{clf_name}] ERROR: {e}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def run_pipeline(root_dir, feature_sets=None):
    """
    Ejecuta el pipeline completo sobre todas las carpetas encontradas.

    root_dir: path a la carpeta que contiene Letters/, Numbers/, Controls/
    feature_sets: lista de sets a evaluar. None = evalúa todos.
    """
    root_dir = Path(root_dir)
    if feature_sets is None:
        feature_sets = ["Estadisticas", "Estadisticas_tm", "Frecuencias_Abs", "Frecuencias_Rel",
                        "Frecuencias_Est", "Wavelets","welch", "Frecuencias_Todas", "TODAS"]

    all_results   = []
    global_cmds   = {}    # para evaluación general fusionando todas las categorías

    # ── Detectar carpetas disponibles ────────────────────────────────────
    found_dirs = []
    for cat in CATEGORY_DIRS:
        cat_path = root_dir / cat
        if cat_path.exists() and cat_path.is_dir():
            found_dirs.append((cat, cat_path))
        else:
            print(f"[~] Carpeta no encontrada, omitiendo: {cat_path}")

    if not found_dirs:
        # Si no hay subcarpetas, trata root_dir como una única categoría
        found_dirs = [("General", root_dir)]

    # ── Por categoría ────────────────────────────────────────────────────
    for cat_name, cat_path in found_dirs:
        print(f"\n{'='*60}")
        print(f"  CATEGORÍA: {cat_name}")
        print(f"{'='*60}")

        commands = load_category(cat_path)
        if not commands:
            continue

        # Acumular para evaluación general (prefijamos cmd con categoría)
        for cmd_idx, trials in commands.items():
            global_key = f"{cat_name}_{cmd_idx}"
            global_cmds[global_key] = trials

        for fs_name in feature_sets:
            print(f"\n  [Feature set: {fs_name}]")
            X, y, cmd_list = build_dataset(commands, feature_set=fs_name)
            if X is None:
                continue
            cat_results = evaluate_classifiers(X, y, fs_name, cat_name)
            all_results.append(cat_results)

    # ── Evaluación general (todas las categorías juntas) ─────────────────
    if len(found_dirs) > 1 and global_cmds:
        print(f"\n{'='*60}")
        print(f"  EVALUACIÓN GENERAL (todas las categorías)")
        print(f"{'='*60}")
        for fs_name in feature_sets:
            print(f"\n  [Feature set: {fs_name}]")
            X, y, _ = build_dataset(global_cmds, feature_set=fs_name)
            if X is None:
                continue
            gen_results = evaluate_classifiers(X, y, fs_name, "GENERAL")
            all_results.append(gen_results)

    # ── Consolidar y guardar resultados ──────────────────────────────────
    if not all_results:
        print("\n[!] Sin resultados para guardar.")
        return None

    df_results = pd.concat(all_results, ignore_index=True)

    out_path = root_dir / "resultados_clasificacion.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n✓ Resultados guardados en: {out_path}")

    # ── Resumen por categoría ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESUMEN — Mejor clasificador por categoría y feature set")
    print("="*60)

    summary = (df_results
               .sort_values("AUC_mean", ascending=False)
               .groupby(["Categoria", "Feature_Set"])
               .first()
               .reset_index()
               [["Categoria", "Feature_Set", "Clasificador",
                 "Acc_mean", "Acc_std", "AUC_mean", "AUC_std"]])

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))

    return df_results


# ─────────────────────────────────────────────
#  7. SCORING PARA INFERENCIA (producción)
# ─────────────────────────────────────────────
def train_final_model(commands_dict, feature_set="TODAS", clf_name="LDA_shrinkage"):
    """
    Entrena el modelo final sobre todos los datos disponibles.
    Retorna el pipeline entrenado listo para inferencia.
    """
    X, y, cmd_list = build_dataset(commands_dict, feature_set=feature_set)
    if X is None:
        return None

    clf = get_classifiers()[clf_name]
    sw  = compute_sample_weight("balanced", y)
    clf.fit(X, y, **{f"{clf.steps[-1][0]}__sample_weight": sw}
            if clf_name != "LDA_shrinkage" else {})
    print(f"✓ Modelo entrenado: {clf_name} | features: {feature_set} | "
          f"muestras: {len(y)}")
    return clf


def predict_command(clf, candidate_dfs, feature_set="TODAS"):
    """
    Dado un modelo entrenado y una lista de DataFrames (uno por comando candidato),
    retorna el índice del comando más probable (el de mayor score acumulado).

    candidate_dfs: lista de listas [[trial_df, ...], ...]
                   una lista de trials por cada comando candidato.
    """
    scores = []
    for cmd_trials in candidate_dfs:
        cmd_score = 0.0
        for df in cmd_trials:
            sig   = preprocess_trial(df)
            feat  = extract_features(sig)[feature_set]
            prob  = clf.predict_proba([feat])[0][1]   # P(target)
            cmd_score += prob
        scores.append(cmd_score / max(len(cmd_trials), 1))

    predicted_idx = int(np.argmax(scores))
    print(f"Scores por comando: {[f'{s:.3f}' for s in scores]}")
    print(f"→ Comando predicho: índice {predicted_idx} (score={scores[predicted_idx]:.3f})")
    return predicted_idx, scores


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Uso: python p300_classifier.py <ruta_root>
    # Ejemplo: python p300_classifier.py ./datos/UserArcane
    if len(sys.argv) < 2:
        print("Uso: python p300_classifier.py <ruta_carpeta_raiz>")
        print("  La carpeta debe contener subcarpetas: Letters/, Numbers/, Controls/")
        print("  o directamente archivos CSV con patrón User{name}_{cmd}_{trial}.csv")
        sys.exit(1)

    root = sys.argv[1]

    # Evalúa todos los feature sets por defecto
    # Para evaluar solo algunos: feature_sets=["Estadisticas", "Wavelets"]
    results = run_pipeline(
        root_dir      = root,
        feature_sets  = ["Estadisticas","Estadisticas_tm", "Frecuencias_Abs", "Frecuencias_Rel","Frecuencias_Est", "Wavelets","welch", "Frecuencias_Todas"],
    )
    #"Estadisticas","Estadisticas_tm", "Frecuencias_Abs", "Frecuencias_Rel","Frecuencias_Est", "Wavelets","welch", "Frecuencias_Todas", "TODAS"