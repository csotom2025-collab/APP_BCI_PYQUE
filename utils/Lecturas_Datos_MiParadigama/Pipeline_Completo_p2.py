"""
Pipeline LDA + Clasificador — Sin Data Leakage
===============================================
Reemplaza el flujo de dos pasos (optimizacion_lda.py → clasificador_lda.py)
por un Pipeline de sklearn donde LDA y clasificador se ajustan SOLO con
datos de entrenamiento, garantizando evaluación honesta.

Estructura de entrada esperada (igual que antes):
    BASE_RESULTS/
    └── {Usuario}/
        ├── Digit/features/    ← CSV: {Usuario}_{Clase}_{Trial}_features.csv
        ├── Char/features/
        └── Comando/features/

Salidas:
    BASE_RESULTS/
    └── {Usuario}/
        └── {Tipo}/
            └── Resultados_Pipeline/
                ├── resumen_{Tipo}.csv
                ├── heatmap_accuracy_{Tipo}.png
                ├── heatmap_f1_{Tipo}.png
                └── confusion_{mejor_modelo}_{Tipo}.png

Uso:
    python pipeline_lda_clasificador.py
    python pipeline_lda_clasificador.py --base D:/EEG_Python/results
    python pipeline_lda_clasificador.py --usuarios UserPony --tipos Char
    python pipeline_lda_clasificador.py --grupos Estadisticas Wavelets TODAS
"""

import os
import re
import copy
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
BASE_RESULTS      = r"C:/Users/crist/OneDrive/Escritorio/APP_BCI_PYQUE/results"
TIPOS_VALIDOS     = ["Digit", "Char", "Comando"]
CARPETA_FEATURES  = "features"
CARPETA_SALIDA    = "Resultados_Pipeline"

TRAIN_SIZE  = 0.80
RANDOM_SEED = 42
CV_FOLDS    = 5        # folds para StratifiedKFold interno
N_COMP_LDA  = None     # None → min(n_clases-1, n_features) automático

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]


# ══════════════════════════════════════════════════════════════
# GRUPOS DE FEATURES (mismos que optimizacion_lda.py)
# ══════════════════════════════════════════════════════════════
def get_feature_sets(all_cols):
    wA_keys  = ["wA5"]
    wD_keys  = ["wD1","wD2","wD3","wD4","wD5"]
    wav_keys = wA_keys + wD_keys

    sets = {
        "Estadisticas": [c for c in all_cols if any(k in c for k in
                         ["mean","std","var","rms","skewness","kurtosis"])
                         and not any(k in c for k in
                         ["delta_","theta_","alpha_","beta_","gamma_","wA","wD"])],

        "Frecuencias_Abs": [c for c in all_cols if any(k in c for k in
                            ["delta_Abs","theta_Abs","alpha_Abs","beta_Abs","gamma_Abs"])],

        "Frecuencias_Rel": [c for c in all_cols if any(k in c for k in
                            ["delta_rel","theta_rel","alpha_rel","beta_rel","gamma_rel"])],

        "Frecuencias_Est": [c for c in all_cols if any(k in c for k in
                            ["delta_mean","theta_mean","alpha_mean","beta_mean","gamma_mean",
                             "beta_std","gamma_std","beta_var","gamma_var",
                             "beta_rms","gamma_rms","beta_skewness","gamma_skewness",
                             "beta_kurtosis","gamma_kurtosis"])],

        "Wavelets": [c for c in all_cols if any(k in c for k in wav_keys)],

        "Frecuencias_Todas": [c for c in all_cols if any(k in c for k in
                              ["delta_","theta_","alpha_","beta_","gamma_"])],

        "TODAS": list(all_cols),
    }
    return {k: v for k, v in sets.items() if v}


# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════
def load_features_folder(folder_path):
    """
    Lee todos los CSV de features. La etiqueta (clase) se extrae del
    nombre del archivo: {Usuario}_{Clase}_{Trial}_features.csv
    """
    pattern = re.compile(r"^(.+?)_(\w+)_(\d+)(?:_features)?\.csv$", re.IGNORECASE)
    records, skipped = [], []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".csv"):
            continue
        m = pattern.match(fname)
        if not m:
            skipped.append(fname)
            continue
        clase, trial = m.group(2), int(m.group(3))
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            df["label"] = clase
            df["trial"] = trial
            records.append(df)
        except Exception as e:
            skipped.append(f"{fname} ({e})")

    if skipped:
        print(f"    ⚠  Omitidos: {', '.join(skipped[:5])}")
    if not records:
        raise ValueError(f"Sin CSV válidos en {folder_path}")

    data = pd.concat(records, ignore_index=True)
    meta = ["label", "trial"]
    feat_cols = [c for c in data.columns if c not in meta]

    # Eliminar features con NaN en cualquier fila
    feat_cols = [c for c in feat_cols if data[c].notna().all()]
    return data, feat_cols


# ══════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE PIPELINES
# ══════════════════════════════════════════════════════════════
def build_pipelines(n_clases):
    """
    Cada pipeline: StandardScaler → LDA → Clasificador.
    LDA recibe n_components=None para que sklearn calcule
    automáticamente min(n_clases-1, n_features).
    """
    lda_kw = dict(n_components=N_COMP_LDA)   # None = automático

    return {
        "LDA + Reg. Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis(**lda_kw)),
            ("clf",    LogisticRegression(max_iter=1000, C=1.0,
                                          solver="lbfgs",
                                          random_state=RANDOM_SEED)),
        ]),
        "LDA + SVM Lineal": Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis(**lda_kw)),
            ("clf",    SVC(kernel="linear", probability=True,
                           random_state=RANDOM_SEED)),
        ]),
        "LDA + SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis(**lda_kw)),
            ("clf",    SVC(kernel="rbf", probability=True,
                           random_state=RANDOM_SEED)),
        ]),
        "LDA + MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("lda",    LinearDiscriminantAnalysis(**lda_kw)),
            ("clf",    MLPClassifier(hidden_layer_sizes=(256, 128),
                                     max_iter=500,
                                     random_state=RANDOM_SEED)),
        ]),
    }


# ══════════════════════════════════════════════════════════════
# EVALUACIÓN DE UN PIPELINE
# ══════════════════════════════════════════════════════════════
def evaluar_pipeline(pipe_template, X_train, X_test, y_train, y_test,
                     etiquetas, n_clases, le):
    """
    1. Valida con StratifiedKFold sobre X_train (LDA nunca ve X_test).
    2. Re-entrena con todo X_train.
    3. Evalúa una sola vez sobre X_test.
    """
    pipe = copy.deepcopy(pipe_template)

    # ── Validación cruzada (solo sobre train) ──────────────────
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=skf, scoring="accuracy", n_jobs=-1)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    # ── Entrenamiento final ────────────────────────────────────
    pipe.fit(X_train, y_train)

    # ── Evaluación en test (una sola vez) ─────────────────────
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm  = confusion_matrix(y_test, y_pred, labels=etiquetas)

    try:
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba    = pipe.predict_proba(X_test)
            y_test_bin = label_binarize(
                le.transform(y_test), classes=np.arange(n_clases)
            )
            auc = (roc_auc_score(y_test_bin, y_proba[:, 1])
                   if n_clases == 2
                   else roc_auc_score(y_test_bin, y_proba,
                                      multi_class="ovr", average="weighted"))
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    return {
        "cv_mean": cv_mean, "cv_std": cv_std,
        "accuracy": acc, "f1": f1, "auc": auc,
        "cm": cm, "y_pred": y_pred, "y_test": y_test,
        "pipe": pipe,
    }


# ══════════════════════════════════════════════════════════════
# PROCESAMIENTO POR TRABAJO (usuario + tipo)
# ══════════════════════════════════════════════════════════════
def procesar_trabajo(trabajo, grupos_filter=None):
    usuario       = trabajo["usuario"]
    tipo          = trabajo["tipo"]
    features_path = trabajo["features_path"]
    output_path   = trabajo["output_path"]
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# {usuario} / {tipo}")
    print(f"{'#'*70}")

    # ── Carga ──────────────────────────────────────────────────
    try:
        data, feat_cols = load_features_folder(features_path)
    except ValueError as e:
        print(f"  ✗ {e}")
        return []

    etiquetas = sorted(data["label"].unique().tolist())
    n_clases  = len(etiquetas)
    le        = LabelEncoder()
    le.fit(etiquetas)

    print(f"  Clases ({n_clases}): {etiquetas}")
    print(f"  Total muestras   : {len(data)}")
    print(f"  Features totales : {len(feat_cols)}\n")

    if n_clases < 2:
        print("  ⚠  Necesita al menos 2 clases. Omitiendo.")
        return []

    # ── Grupos de features ─────────────────────────────────────
    all_sets = get_feature_sets(feat_cols)
    if grupos_filter:
        all_sets = {k: v for k, v in all_sets.items() if k in grupos_filter}
    if not all_sets:
        print("  ⚠  Sin grupos de features válidos.")
        return []

    # ── Pipelines ─────────────────────────────────────────────
    pipelines = build_pipelines(n_clases)

    summary_rows = []
    all_results  = {}   # {grupo: {modelo: resultado}}

    for grupo, cols in all_sets.items():
        if not cols:
            continue

        print(f"\n  {'·'*58}")
        print(f"  Grupo: {grupo}  ({len(cols)} features)")
        print(f"  {'·'*58}")

        X_raw = data[cols].values.astype(float)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = data["label"].values

        # ── Split ANTES de cualquier transformación ────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_raw,
                train_size=TRAIN_SIZE,
                stratify=y_raw,
                random_state=RANDOM_SEED,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_raw,
                train_size=TRAIN_SIZE,
                random_state=RANDOM_SEED,
            )

        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

        grupo_results = {}
        for nombre_pipe, pipe_template in pipelines.items():
            try:
                res = evaluar_pipeline(
                    pipe_template,
                    X_train, X_test, y_train, y_test,
                    etiquetas, n_clases, le,
                )
                grupo_results[nombre_pipe] = res
                auc_str = f"{res['auc']:.2%}" if not np.isnan(res['auc']) else "N/A"
                print(
                    f"    {nombre_pipe:<25}"
                    f"  CV: {res['cv_mean']:.3f}±{res['cv_std']:.3f}"
                    f"  Test Acc: {res['accuracy']:.2%}"
                    f"  F1: {res['f1']:.2%}"
                    f"  AUC: {auc_str}"
                )
                summary_rows.append({
                    "Grupo":    grupo,
                    "Pipeline": nombre_pipe,
                    "CV_mean":  res["cv_mean"],
                    "CV_std":   res["cv_std"],
                    "Accuracy": res["accuracy"],
                    "F1":       res["f1"],
                    "AUC":      res["auc"] if not np.isnan(res["auc"]) else None,
                })
            except Exception as e:
                print(f"    {nombre_pipe:<25}  ERROR: {e}")
                grupo_results[nombre_pipe] = None

        all_results[grupo] = grupo_results

    if not summary_rows:
        return []

    df_sum = pd.DataFrame(summary_rows)

    # ── Guardar resumen CSV ────────────────────────────────────
    csv_path = os.path.join(output_path, f"resumen_{tipo}.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"\n  Resumen guardado → {csv_path}")

    # ── Heatmaps ──────────────────────────────────────────────
    _plot_heatmap(df_sum, output_path, tipo, "Accuracy", usuario)
    _plot_heatmap(df_sum, output_path, tipo, "F1",       usuario)

    # ── Matriz de confusión del mejor modelo global ────────────
    best = df_sum.loc[df_sum["Accuracy"].idxmax()]
    best_grupo = best["Grupo"]
    best_pipe  = best["Pipeline"]
    best_res   = all_results.get(best_grupo, {}).get(best_pipe)
    if best_res:
        _plot_confusion(
            best_res["cm"], etiquetas,
            best_pipe, best_grupo, tipo,
            best_res["accuracy"], output_path, usuario,
        )

    # Resumen por tipo
    print(f"\n  {'='*58}")
    print(f"  Mejor pipeline  : {best_pipe}")
    print(f"  Grupo           : {best_grupo}")
    print(f"  CV Accuracy     : {best['CV_mean']:.3f} ± {best['CV_std']:.3f}")
    print(f"  Test Accuracy   : {best['Accuracy']:.2%}")
    print(f"  Test F1         : {best['F1']:.2%}")
    print(f"  {'='*58}")

    # Añadir metadatos para el resumen global
    for row in summary_rows:
        row["usuario"] = usuario
        row["tipo"]    = tipo

    return summary_rows


# ══════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════
def _plot_heatmap(df_sum, output_path, tipo, metric, usuario):
    try:
        pivot = df_sum.pivot(index="Pipeline", columns="Grupo", values=metric)
        fig_w = max(8, len(pivot.columns) * 1.4)
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5, linecolor="gray",
                    ax=ax, cbar_kws={"label": metric})
        ax.set_title(f"{metric} — Pipeline LDA + Clf\n{usuario} | {tipo}",
                     fontsize=13, fontweight="bold", pad=14)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = os.path.join(output_path, f"heatmap_{metric.lower()}_{tipo}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Heatmap {metric} → {out}")
    except Exception as e:
        print(f"  ⚠  Heatmap {metric} falló: {e}")


def _plot_confusion(cm, etiquetas, pipe_name, grupo, tipo,
                    acc, output_path, usuario):
    try:
        n = len(etiquetas)
        fig_sz = max(8, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=etiquetas)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        if n > 15:
            ax.tick_params(axis="both", labelsize=7)
            for t in ax.texts:
                t.set_fontsize(6)
        ax.set_title(
            f"Matriz de Confusión\n{pipe_name} | Grupo: {grupo}\n"
            f"Test Accuracy: {acc:.2%} | {usuario} | {tipo}",
            fontsize=11, fontweight="bold", pad=12,
        )
        plt.tight_layout()
        safe = pipe_name.replace(" ", "_").replace("+", "").replace(".", "")
        out  = os.path.join(output_path, f"confusion_{safe}_{tipo}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusión → {out}")
    except Exception as e:
        print(f"  ⚠  Confusión falló: {e}")


# ══════════════════════════════════════════════════════════════
# LDA GENERAL — todos los tipos juntos por usuario
# ══════════════════════════════════════════════════════════════
def procesar_lda_general(usuario, trabajos_usuario, base_results, grupos_filter=None):
    """
    Concatena Digit + Char + Comando (todos los tipos del usuario) en un
    único dataset. La etiqueta combina tipo y clase: Digit_0, Char_A, etc.
    Aplica el mismo Pipeline(StandardScaler → LDA → Clf) con split honesto.

    Salida: BASE_RESULTS/{usuario}/Pipeline_General/
    """
    output_path = os.path.join(base_results, usuario, "Pipeline_General")
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  PIPELINE GENERAL — {usuario}")
    print(f"  Tipos combinados: {[t['tipo'] for t in trabajos_usuario]}")
    print(f"{'═'*70}\n")

    # ── 1. Cargar y concatenar todos los tipos ──────────────────
    all_dfs = []
    for trabajo in trabajos_usuario:
        tipo          = trabajo["tipo"]
        features_path = trabajo["features_path"]
        try:
            data_tipo, feat_cols_tipo = load_features_folder(features_path)
            # Prefijar etiqueta: "A" de Char → "Char_A"
            data_tipo["label"]    = tipo + "_" + data_tipo["label"].astype(str)
            data_tipo["tipo_src"] = tipo
            all_dfs.append(data_tipo)
            print(f"  ✔ {tipo:<10} → {len(data_tipo)} muestras")
        except ValueError as e:
            print(f"  ⚠  {tipo}: {e}")

    if not all_dfs:
        print(f"  ✗ Sin datos para LDA General de {usuario}. Omitiendo.")
        return []

    data_total = pd.concat(all_dfs, ignore_index=True)
    meta_cols  = ["label", "trial", "tipo_src"]
    feat_cols  = [c for c in data_total.columns if c not in meta_cols]
    # Eliminar features con NaN
    feat_cols  = [c for c in feat_cols if data_total[c].notna().all()]

    etiquetas = sorted(data_total["label"].unique().tolist())
    n_clases  = len(etiquetas)
    le        = LabelEncoder()
    le.fit(etiquetas)

    print(f"\n  Clases totales ({n_clases}): {etiquetas}")
    print(f"  Muestras totales          : {len(data_total)}")
    print(f"  Features comunes          : {len(feat_cols)}\n")

    if n_clases < 2:
        print("  ⚠  Necesita al menos 2 clases. Omitiendo.")
        return []

    # ── 2. Grupos de features ───────────────────────────────────
    all_sets = get_feature_sets(feat_cols)
    if grupos_filter:
        all_sets = {k: v for k, v in all_sets.items() if k in grupos_filter}
    if not all_sets:
        print("  ⚠  Sin grupos de features válidos.")
        return []

    pipelines    = build_pipelines(n_clases)
    summary_rows = []
    all_results  = {}

    for grupo, cols in all_sets.items():
        if not cols:
            continue

        print(f"\n  {'·'*58}")
        print(f"  Grupo: {grupo}  ({len(cols)} features)")
        print(f"  {'·'*58}")

        X_raw = data_total[cols].values.astype(float)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = data_total["label"].values

        # Split ANTES de cualquier transformación
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_raw,
                train_size=TRAIN_SIZE,
                stratify=y_raw,
                random_state=RANDOM_SEED,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y_raw,
                train_size=TRAIN_SIZE,
                random_state=RANDOM_SEED,
            )

        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

        grupo_results = {}
        for nombre_pipe, pipe_template in pipelines.items():
            try:
                res = evaluar_pipeline(
                    pipe_template,
                    X_train, X_test, y_train, y_test,
                    etiquetas, n_clases, le,
                )
                grupo_results[nombre_pipe] = res
                auc_str = f"{res['auc']:.2%}" if not np.isnan(res['auc']) else "N/A"
                print(
                    f"    {nombre_pipe:<25}"
                    f"  CV: {res['cv_mean']:.3f}±{res['cv_std']:.3f}"
                    f"  Test Acc: {res['accuracy']:.2%}"
                    f"  F1: {res['f1']:.2%}"
                    f"  AUC: {auc_str}"
                )
                summary_rows.append({
                    "usuario":  usuario,
                    "tipo":     "General",
                    "Grupo":    grupo,
                    "Pipeline": nombre_pipe,
                    "CV_mean":  res["cv_mean"],
                    "CV_std":   res["cv_std"],
                    "Accuracy": res["accuracy"],
                    "F1":       res["f1"],
                    "AUC":      res["auc"] if not np.isnan(res["auc"]) else None,
                })
            except Exception as e:
                print(f"    {nombre_pipe:<25}  ERROR: {e}")
                grupo_results[nombre_pipe] = None

        all_results[grupo] = grupo_results

    if not summary_rows:
        return []

    df_sum = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_path, f"resumen_General.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"\n  Resumen General guardado → {csv_path}")

    _plot_heatmap(df_sum, output_path, "General", "Accuracy", usuario)
    _plot_heatmap(df_sum, output_path, "General", "F1",       usuario)

    best     = df_sum.loc[df_sum["Accuracy"].idxmax()]
    best_res = all_results.get(best["Grupo"], {}).get(best["Pipeline"])
    if best_res:
        _plot_confusion(
            best_res["cm"], etiquetas,
            best["Pipeline"], best["Grupo"], "General",
            best_res["accuracy"], output_path, usuario,
        )

    print(f"\n  {'='*58}")
    print(f"  Mejor pipeline General : {best['Pipeline']}")
    print(f"  Grupo                  : {best['Grupo']}")
    print(f"  CV Accuracy            : {best['CV_mean']:.3f} ± {best['CV_std']:.3f}")
    print(f"  Test Accuracy          : {best['Accuracy']:.2%}")
    print(f"  Test F1                : {best['F1']:.2%}")
    print(f"  {'='*58}")

    return summary_rows


# ══════════════════════════════════════════════════════════════
# DESCUBRIMIENTO DE ESTRUCTURA
# ══════════════════════════════════════════════════════════════
def discover_structure(base_results, usuarios_filter=None, tipos_filter=None):
    if not os.path.isdir(base_results):
        raise FileNotFoundError(f"No existe: {base_results}")

    tipos_filter = tipos_filter or TIPOS_VALIDOS
    candidatos   = sorted([d for d in os.listdir(base_results)
                           if os.path.isdir(os.path.join(base_results, d))])
    if usuarios_filter:
        candidatos = [u for u in candidatos if u in usuarios_filter]

    trabajos = []
    for usuario in candidatos:
        user_path = os.path.join(base_results, usuario)
        for tipo in sorted(os.listdir(user_path)):
            if tipo not in tipos_filter:
                continue
            feat_path = os.path.join(user_path, tipo, CARPETA_FEATURES)
            if not os.path.isdir(feat_path):
                continue
            csv_count = len([f for f in os.listdir(feat_path)
                             if f.lower().endswith(".csv")])
            if csv_count == 0:
                continue
            trabajos.append({
                "usuario":       usuario,
                "tipo":          tipo,
                "features_path": feat_path,
                "output_path":   os.path.join(user_path, tipo, CARPETA_SALIDA),
                "csv_count":     csv_count,
            })
    return trabajos


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline LDA + Clasificador sin data leakage — EEG BCI"
    )
    parser.add_argument("--base",     default=BASE_RESULTS)
    parser.add_argument("--usuarios", nargs="+", default=None)
    parser.add_argument("--tipos",    nargs="+", default=None)
    parser.add_argument("--grupos",   nargs="+", default=None)
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"#  PIPELINE LDA + CLASIFICADOR — SIN DATA LEAKAGE")
    print(f"{'#'*70}")
    print(f"\n  Base      : {args.base}")
    print(f"  Usuarios  : {args.usuarios or 'todos'}")
    print(f"  Tipos     : {args.tipos    or TIPOS_VALIDOS}")
    print(f"  Grupos    : {args.grupos   or 'todos'}")
    print(f"  CV folds  : {CV_FOLDS}")
    print(f"  Train/Test: {int(TRAIN_SIZE*100)}/{int((1-TRAIN_SIZE)*100)}\n")

    trabajos = discover_structure(
        args.base,
        usuarios_filter=args.usuarios,
        tipos_filter=args.tipos,
    )
    if not trabajos:
        print("✗ No se encontraron datos. Verifica la ruta base.")
        return

    print(f"  Trabajos encontrados: {len(trabajos)}")
    for t in trabajos:
        print(f"    • {t['usuario']:<15} / {t['tipo']:<10} ({t['csv_count']} CSV)")

    resumen_global = []
    for trabajo in trabajos:
        rows = procesar_trabajo(trabajo, grupos_filter=args.grupos)
        resumen_global.extend(rows)

    # ── Pipeline General por usuario (todos los tipos juntos) ──
    usuarios_unicos = sorted(set(t["usuario"] for t in trabajos))
    for usuario in usuarios_unicos:
        trabajos_usuario = [t for t in trabajos if t["usuario"] == usuario]
        rows_gen = procesar_lda_general(
            usuario, trabajos_usuario, args.base,
            grupos_filter=args.grupos,
        )
        resumen_global.extend(rows_gen)

    # ── Resumen global ─────────────────────────────────────────
    if resumen_global:
        df_global = pd.DataFrame(resumen_global)
        cols = ["usuario","tipo","Grupo","Pipeline",
                "CV_mean","CV_std","Accuracy","F1","AUC"]
        df_global = df_global[[c for c in cols if c in df_global.columns]]
        df_global = df_global.sort_values(
            ["usuario","tipo","Accuracy"], ascending=[True, True, False]
        )

        out_global = os.path.join(args.base, "resumen_global_pipeline.csv")
        df_global.to_csv(out_global, index=False)

        print(f"\n{'#'*70}")
        print(f"#  RESUMEN GLOBAL")
        print(f"{'#'*70}")
        print(df_global.to_string(index=False))
        print(f"\n  Resumen global → {out_global}")

    print(f"\n{'#'*70}")
    print(f"#  FINALIZADO")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()

# ── Ejemplos de uso ────────────────────────────────────────────────────────────
# Todos los usuarios y tipos:
#   python pipeline_lda_clasificador.py
#
# Un solo usuario:
#   python pipeline_lda_clasificador.py --usuarios UserPony
#
# Usuario + tipo específico:
#   python pipeline_lda_clasificador.py --usuarios UserPony --tipos Char
#
# Solo algunos grupos de features:
#   python pipeline_lda_clasificador.py --grupos Estadisticas Wavelets TODAS