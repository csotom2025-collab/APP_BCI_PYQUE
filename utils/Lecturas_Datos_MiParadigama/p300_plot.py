"""
p300_plots.py  v2
=================
Correcciones principales vs v1:
  - build_all_feature_sets se llama UNA sola vez por categoría; el resultado
    (X_all, y_ref, cache) se pasa como argumento a TODAS las funciones de gráficas.
    Ninguna función recalcula features por su cuenta.
  - evaluate_all siempre recibe los 7 feature sets completos.
  - plot_cmd_confusion_scoring/binary usan el cache de features pre-calculado.
  - Firma unificada: todas las funciones de plot reciben X_all, y_ref, cache.

Uso:
  python p300_plots.py <ruta_root> [ruta_salida]
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from p300_classifier import (
    load_category, preprocess_trial, extract_features,
    CATEGORY_DIRS, EEG_COLS, FEATURE_SETS
)

# ─────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────
N_FOLDS    = 5
MAX_NEG    = 5          # ratio máx non-target por target
FEAT_ORDER = [          # orden visual para heatmap
    "Estadisticas", "Frecuencias_Abs", "Frecuencias_Rel",
    "Frecuencias_Est", "Wavelets", "Frecuencias_Todas", "TODAS"
]

COMMAND_MAP = {
    'Letters': {
        'A':{'label':'Letra A'},  'B':{'label':'Letra B'},
        'C':{'label':'Letra C'},  'D':{'label':'Letra D'},
        'E':{'label':'Letra E'},  'F':{'label':'Letra F'},
        'G':{'label':'Letra G'},  'H':{'label':'Letra H'},
        'I':{'label':'Letra I'},  'J':{'label':'Letra J'},
        'K':{'label':'Letra K'},  'L':{'label':'Letra L'},
        'M':{'label':'Letra M'},  'N':{'label':'Letra N'},
        'O':{'label':'Letra O'},  'P':{'label':'Letra P'},
        'Q':{'label':'Letra Q'},  'R':{'label':'Letra R'},
        'S':{'label':'Letra S'},  'T':{'label':'Letra T'},
        'U':{'label':'Letra U'},  'V':{'label':'Letra V'},
        'W':{'label':'Letra W'},  'X':{'label':'Letra X'},
        'Y':{'label':'Letra Y'},  'Z':{'label':'Letra Z'},
    },
    'Numbers': {
        '1':{'label':'Número 1'}, '2':{'label':'Número 2'},
        '3':{'label':'Número 3'}, '4':{'label':'Número 4'},
        '5':{'label':'Número 5'}, '6':{'label':'Número 6'},
        '7':{'label':'Número 7'}, '8':{'label':'Número 8'},
        '9':{'label':'Número 9'},
    },
    'Controls': {
        '───':{'label':'Espacio'},
        '⟵':  {'label':'Backspace'},
        '↩':  {'label':'Enter'},
    },
}

CAT_COLORS = {
    'Letters': '#534AB7', 'Numbers': '#0F6E56',
    'Controls': '#993C1D', 'GENERAL': '#185FA5',
}


# ─────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────
def get_classifiers():
    return {
        "LDA": Pipeline([("sc", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis(
                             solver="eigen", shrinkage="auto"))]),
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", probability=True,
                                     class_weight="balanced",
                                     C=1.0, gamma="scale"))]),
        "GBM": Pipeline([("sc", StandardScaler()),
                         ("clf", GradientBoostingClassifier(
                             n_estimators=80, max_depth=3,
                             learning_rate=0.1))]),
    }


def cmd_label(cmd_key, category):
    """Devuelve la abreviación del comando (ej 'A', '1', 'Enter')."""
    cat_map = COMMAND_MAP.get(category, {})
    key_str = str(cmd_key)
    if key_str in cat_map:
        return cat_map[key_str]['label'].split()[-1]
    if isinstance(cmd_key, int):
        keys = list(cat_map.keys())
        if cmd_key < len(keys):
            return cat_map[keys[cmd_key]]['label'].split()[-1]
    return str(cmd_key)


# ─────────────────────────────────────────
#  CONSTRUCCIÓN DEL DATASET (UNA SOLA VEZ)
# ─────────────────────────────────────────
def build_dataset_all(commands_dict):
    """
    Preprocesa y extrae features de todos los trials UNA sola vez.

    Retorna:
      cache    : {cmd -> [feat_dict, ...]}   features por trial
      X_all    : {fs_name -> (X ndarray, y ndarray)}  datasets listos
      y_ref    : y del primer feature set (todos iguales)
      cmd_list : lista ordenada de comandos disponibles
    """
    cmd_list = sorted(commands_dict.keys())
    n_cmds   = len(cmd_list)
    if n_cmds < 2:
        return None, None, None, cmd_list

    print(f"    Preprocesando {sum(len(v) for v in commands_dict.values())} trials...")

    # ── Paso 1: extraer features (un solo paso por trial) ─────────────
    cache = {}
    for cmd in cmd_list:
        cache[cmd] = []
        for df in commands_dict[cmd]:
            try:
                sig  = preprocess_trial(df)
                feat = extract_features(sig)   # dict con los 7 sets
                cache[cmd].append(feat)
            except Exception as e:
                print(f"    [!] Error preprocesando trial de cmd={cmd}: {e}")

    # ── Paso 2: construir X, y para cada feature set ───────────────────
    X_all = {}
    for fs_name in FEATURE_SETS:
        pos_X, neg_X = [], []
        for target_cmd in cmd_list:
            # Positivos
            for feat in cache[target_cmd]:
                pos_X.append(feat[fs_name])
            # Negativos (sub-muestreados)
            neg_pool = [feat[fs_name]
                        for other in cmd_list if other != target_cmd
                        for feat in cache[other]]
            max_neg = len(cache[target_cmd]) * MAX_NEG
            if len(neg_pool) > max_neg:
                idx = np.random.choice(len(neg_pool), max_neg, replace=False)
                neg_pool = [neg_pool[i] for i in idx]
            neg_X.extend(neg_pool)

        X = np.array(pos_X + neg_X)
        y = np.array([1]*len(pos_X) + [0]*len(neg_X))
        X_all[fs_name] = (X, y)

    y_ref = X_all[FEATURE_SETS[0]][1]
    n_pos = int(y_ref.sum())
    n_neg = int((y_ref == 0).sum())
    print(f"    Dataset: {n_pos} targets | {n_neg} non-targets | "
          f"{len(cmd_list)} comandos | "
          f"{X_all[FEATURE_SETS[0]][0].shape[1]}–"
          f"{X_all['TODAS'][0].shape[1]} features/set")
    return cache, X_all, y_ref, cmd_list


# ─────────────────────────────────────────
#  EVALUACIÓN (usa X_all pre-calculado)
# ─────────────────────────────────────────
def evaluate_classifiers(X_all, y_ref):
    """
    Evalúa los 3 clasificadores × 7 feature sets.
    X_all: {fs_name: (X, y)}  — viene de build_dataset_all
    y_ref: array y compartido
    Retorna DataFrame con resultados.
    """
    rows = []
    cv   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fs_name in FEATURE_SETS:
        if fs_name not in X_all:
            continue
        X, y = X_all[fs_name]
        for clf_name, pipeline in get_classifiers().items():
            try:
                acc = cross_val_score(pipeline, X, y, cv=cv,
                                      scoring="accuracy", n_jobs=-1)
                auc = cross_val_score(pipeline, X, y, cv=cv,
                                      scoring="roc_auc",  n_jobs=-1)
                rows.append({
                    "Clasificador": clf_name,
                    "Feature_Set":  fs_name,
                    "AUC_mean": round(auc.mean(), 4),
                    "AUC_std":  round(auc.std(),  4),
                    "Acc_mean": round(acc.mean(), 4),
                    "Acc_std":  round(acc.std(),  4),
                    "N_pos": int(y.sum()),
                    "N_neg": int((y == 0).sum()),
                })
                print(f"    [{clf_name:4s}|{fs_name:20s}] "
                      f"Acc={acc.mean():.3f}±{acc.std():.3f}  "
                      f"AUC={auc.mean():.3f}±{auc.std():.3f}")
            except Exception as e:
                print(f"    [!] {clf_name}/{fs_name}: {e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
#  PLOT 1: HEATMAP clf × feature_set
# ─────────────────────────────────────────
def plot_heatmap(results_df, category_name, ax, metric="AUC_mean"):
    pivot     = results_df.pivot(index="Clasificador", columns="Feature_Set",
                                  values=metric)
    pivot_std = results_df.pivot(index="Clasificador", columns="Feature_Set",
                                  values=metric.replace("mean", "std"))

    ordered = [c for c in FEAT_ORDER if c in pivot.columns]
    pivot     = pivot[ordered]
    pivot_std = pivot_std[ordered]

    annot = pd.DataFrame(
        [[f"{pivot.loc[r,c]:.3f}\n±{pivot_std.loc[r,c]:.3f}"
          for c in ordered] for r in pivot.index],
        index=pivot.index, columns=ordered)

    cat_color = CAT_COLORS.get(category_name, '#185FA5')
    cmap      = sns.light_palette(cat_color, as_cmap=True)

    sns.heatmap(pivot, ax=ax, annot=annot, fmt="", cmap=cmap,
                vmin=0.4, vmax=1.0, linewidths=0.5, linecolor="#e0e0e0",
                cbar_kws={"shrink": 0.8, "label": metric},
                annot_kws={"size": 7.5})

    max_val = pivot.values.max()
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(ordered):
            if abs(pivot.loc[row, col] - max_val) < 1e-6:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor='gold', lw=2.5, clip_on=False))

    ax.set_title(f"{category_name} — {metric} (todos los feature sets)",
                 fontsize=11, fontweight='bold', color=cat_color, pad=8)
    ax.set_xlabel("Feature Set", fontsize=9)
    ax.set_ylabel("Clasificador", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=0)


# ─────────────────────────────────────────
#  PLOT 2: CONFUSION MATRIX BINARIA (Target vs Non-Target)
# ─────────────────────────────────────────
def plot_confusion_binary(X_all, best_clf, best_fs, category_name, ax):
    """Matriz 2×2 Target/Non-Target del mejor clasificador."""
    if best_fs not in X_all:
        ax.text(0.5, 0.5, f"Feature set '{best_fs}' no disponible",
                ha='center', va='center', transform=ax.transAxes); return

    X, y  = X_all[best_fs]
    clf   = get_classifiers()[best_clf]
    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        y_pred_all.extend(clf.predict(X[te]))
        y_true_all.extend(y[te])

    cm      = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cat_color = CAT_COLORS.get(category_name, '#185FA5')
    cmap      = sns.light_palette(cat_color, as_cmap=True)

    sns.heatmap(cm_norm, ax=ax, annot=True, fmt=".2%", cmap=cmap,
                xticklabels=["Non-Target","Target"],
                yticklabels=["Non-Target","Target"],
                linewidths=0.5, linecolor="#e0e0e0",
                cbar_kws={"shrink":0.7,"label":"Proporción"},
                vmin=0, vmax=1, annot_kws={"size":11})

    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.72, f"n={cm[i,j]}",
                    ha='center', va='center', fontsize=8, color='#555')

    ax.set_title(f"{category_name} — Binaria (Target/Non-Target)\n"
                 f"{best_clf} · {best_fs}",
                 fontsize=10, fontweight='bold', color=cat_color, pad=8)
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Real", fontsize=9)


# ─────────────────────────────────────────
#  PLOT 3: CONFUSION MATRIX NxN — SCORING P300 POR COMANDO
# ─────────────────────────────────────────
def plot_confusion_scoring(cache, X_all, best_clf, best_fs,
                            category_name, ax):
    """
    Matriz NxN: fila=comando real, col=comando predicho por scoring P300.
    Usa el cache de features ya calculado — NO re-extrae nada.
    """
    cmd_list = sorted(cache.keys())
    n_cmds   = len(cmd_list)
    if n_cmds < 3:
        ax.text(0.5, 0.5, "Mínimo 3 comandos", ha='center', va='center',
                transform=ax.transAxes); return

    y_true, y_pred = [], []

    for test_cmd in cmd_list:
        # Construir dataset de entrenamiento con todos menos test_cmd
        X_tr, y_tr = [], []
        for cmd in cmd_list:
            label = 1 if cmd == test_cmd else 0
            for feat in cache[cmd]:
                X_tr.append(feat[best_fs]); y_tr.append(label)
        X_tr = np.array(X_tr); y_tr = np.array(y_tr)
        if y_tr.sum() == 0 or (y_tr==0).sum() == 0:
            continue

        clf = get_classifiers()[best_clf]
        clf.fit(X_tr, y_tr)

        # Scoring acumulado: promedio P(target) de cada candidato
        scores = {
            c: np.mean([clf.predict_proba([f[best_fs]])[0][1]
                        for f in cache[c]])
            for c in cmd_list
        }
        y_true.append(test_cmd)
        y_pred.append(max(scores, key=scores.get))

    idx_map = {c: i for i, c in enumerate(cmd_list)}
    ldisp   = [cmd_label(k, category_name) for k in cmd_list]
    cm      = np.zeros((n_cmds, n_cmds), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx_map[t], idx_map[p]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)
    acc      = np.trace(cm) / max(cm.sum(), 1)

    cat_color = CAT_COLORS.get(category_name, '#185FA5')
    cmap      = sns.light_palette(cat_color, as_cmap=True)
    fs_val    = 7 if n_cmds <= 12 else 5

    sns.heatmap(cm_norm, ax=ax, annot=True, fmt=".0%", cmap=cmap,
                xticklabels=ldisp, yticklabels=ldisp,
                linewidths=0.4, linecolor="#e8e8e8", vmin=0, vmax=1,
                cbar_kws={"shrink":0.7,"label":"Proporción"},
                annot_kws={"size": fs_val})

    for i in range(n_cmds):
        ax.add_patch(plt.Rectangle((i,i), 1, 1, fill=False,
                     edgecolor=cat_color, lw=1.5))

    ax.set_title(f"{category_name} — Scoring P300 por comando (NxN)\n"
                 f"{best_clf} · {best_fs} · Acc={acc:.1%}",
                 fontsize=10, fontweight='bold', color=cat_color, pad=8)
    ax.set_xlabel("Comando predicho", fontsize=9)
    ax.set_ylabel("Comando real",     fontsize=9)
    tk = 7 if n_cmds <= 16 else 5
    ax.tick_params(axis='x', labelsize=tk, rotation=45)
    ax.tick_params(axis='y', labelsize=tk, rotation=0)


# ─────────────────────────────────────────
#  PLOT 4: CONFUSION MATRIX Nx2 — DETECCIÓN BINARIA POR COMANDO
# ─────────────────────────────────────────
def plot_confusion_binary_per_cmd(cache, X_all, best_clf, best_fs,
                                   category_name, ax):
    """
    Matriz Nx2: fila=comando real, col=[falso_neg, verdadero_pos].
    Muestra qué comandos son difíciles de detectar.
    Usa el cache de features ya calculado.
    """
    cmd_list = sorted(cache.keys())
    n_cmds   = len(cmd_list)
    if n_cmds < 2:
        ax.text(0.5, 0.5, "Mínimo 2 comandos", ha='center', va='center',
                transform=ax.transAxes); return

    # Construir dataset completo con tag de comando por muestra
    X_list, y_list, cmd_tag = [], [], []
    for target_cmd in cmd_list:
        for feat in cache[target_cmd]:
            X_list.append(feat[best_fs]); y_list.append(1)
            cmd_tag.append(target_cmd)
        neg_pool = [(feat[best_fs], o)
                    for o in cmd_list if o != target_cmd
                    for feat in cache[o]]
        mx = len(cache[target_cmd]) * MAX_NEG
        if len(neg_pool) > mx:
            idx = np.random.choice(len(neg_pool), mx, replace=False)
            neg_pool = [neg_pool[i] for i in idx]
        for feat, src in neg_pool:
            X_list.append(feat); y_list.append(0)
            cmd_tag.append(f"__neg__{src}")

    X_arr   = np.array(X_list)
    y_arr   = np.array(y_list)
    cmd_arr = np.array(cmd_tag, dtype=object)

    cv          = StratifiedKFold(n_splits=min(N_FOLDS, n_cmds),
                                   shuffle=True, random_state=42)
    y_pred_full = np.full(len(y_arr), -1, dtype=int)
    for tr, te in cv.split(X_arr, y_arr):
        clf = get_classifiers()[best_clf]
        clf.fit(X_arr[tr], y_arr[tr])
        y_pred_full[te] = clf.predict(X_arr[te])

    # Matriz Nx2 solo para trials TARGET
    ldisp = [cmd_label(k, category_name) for k in cmd_list]
    mat   = np.zeros((n_cmds, 2), dtype=int)
    for i, cmd in enumerate(cmd_list):
        mask = (cmd_arr == cmd) & (y_arr == 1)
        if mask.sum() == 0: continue
        preds = y_pred_full[mask]
        mat[i, 0] = (preds == 0).sum()   # FN
        mat[i, 1] = (preds == 1).sum()   # TP

    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.where(row_sums > 0, mat / row_sums, 0.0)
    det_rate = mat[:, 1].sum() / max(mat.sum(), 1)

    cat_color = CAT_COLORS.get(category_name, '#185FA5')
    cmap      = sns.light_palette(cat_color, as_cmap=True)

    sns.heatmap(mat_norm, ax=ax, annot=True, fmt=".0%", cmap=cmap,
                xticklabels=["Non-Target\n(Falso neg.)",
                              "Target\n(Verdadero pos.)"],
                yticklabels=ldisp,
                linewidths=0.4, linecolor="#e8e8e8", vmin=0, vmax=1,
                cbar_kws={"shrink":0.7,"label":"Tasa"},
                annot_kws={"size": 8 if n_cmds <= 16 else 6})

    for i in range(n_cmds):
        ax.add_patch(plt.Rectangle((1,i), 1, 1, fill=False,
                     edgecolor=cat_color, lw=1.2))

    ax.set_title(f"{category_name} — Detección Target por comando\n"
                 f"{best_clf} · {best_fs} · Det={det_rate:.1%}",
                 fontsize=10, fontweight='bold', color=cat_color, pad=8)
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Comando real", fontsize=9)
    tk = 7 if n_cmds <= 16 else 5
    ax.tick_params(axis='x', labelsize=9,  rotation=0)
    ax.tick_params(axis='y', labelsize=tk, rotation=0)


# ─────────────────────────────────────────
#  PLOT 5: AUC POR FOLD CV
# ─────────────────────────────────────────
def plot_auc_folds(X_all, best_clf, best_fs, category_name, ax):
    """Barras horizontales AUC por fold del mejor clf/fs."""
    if best_fs not in X_all:
        return
    X, y = X_all[best_fs]
    clf  = get_classifiers()[best_clf]
    cv   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    auc_vals = np.clip(
        cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1), 0, 1)

    cat_color = CAT_COLORS.get(category_name, '#185FA5')
    y_pos     = np.arange(len(auc_vals))
    colors    = [cat_color if v >= 0.7 else '#B0A8D0'
                 if v >= 0.55 else '#E0D8F0' for v in auc_vals]
    bars = ax.barh(y_pos, auc_vals, color=colors, height=0.6,
                   edgecolor='white', linewidth=0.5)

    ax.axvline(0.5,  color='#aaa', lw=1, ls='--', label='Azar (0.5)')
    ax.axvline(0.7,  color='#FF9800', lw=1, ls='-.', label='Bueno (0.7)')
    ax.axvline(0.85, color='#4CAF50', lw=1, ls='-',  label='Excelente (0.85)')

    for bar, val in zip(bars, auc_vals):
        ax.text(min(val+0.01, 0.97), bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va='center', ha='left', fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Fold {i+1}" for i in range(len(auc_vals))], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC-ROC", fontsize=9)
    ax.set_title(f"{category_name} — AUC por fold CV\n{best_clf} · {best_fs}",
                 fontsize=10, fontweight='bold', color=cat_color, pad=8)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.spines[['top','right']].set_visible(False)


# ─────────────────────────────────────────
#  GENERADOR DE FIGURAS
# ─────────────────────────────────────────
def generate_figures(cache, X_all, y_ref, results_df, commands_dict,
                      category_name, output_dir):
    """
    Genera y guarda las 3 figuras para una categoría:
      Fig 1: heatmap + confusion binaria + AUC folds
      Fig 2: confusion NxN scoring P300
      Fig 3: confusion Nx2 detección binaria por comando
    """
    best_row = results_df.loc[results_df["AUC_mean"].idxmax()]
    best_clf = best_row["Clasificador"]
    best_fs  = best_row["Feature_Set"]
    best_auc = best_row["AUC_mean"]
    cat_color = CAT_COLORS.get(category_name, '#185FA5')

    print(f"\n  ★ Mejor: {best_clf} + {best_fs}  →  AUC={best_auc:.4f}")

    # ── Fig 1: overview ─────────────────────────────────────────────────
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle(
        f"P300 Speller — {category_name}\n"
        f"Mejor: {best_clf} · {best_fs} · AUC={best_auc:.3f}",
        fontsize=14, fontweight='bold', color=cat_color, y=0.99)
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.45, wspace=0.35)

    plot_heatmap(results_df, category_name, fig1.add_subplot(gs1[0, :]))
    plot_confusion_binary(X_all, best_clf, best_fs, category_name,
                          fig1.add_subplot(gs1[1, 0]))
    plot_auc_folds(X_all, best_clf, best_fs, category_name,
                   fig1.add_subplot(gs1[1, 1]))

    f1_path = output_dir / f"reporte_{category_name}.png"
    fig1.savefig(f1_path, dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"  ✓ {f1_path.name}")

    # ── Fig 2+3: matrices por comando ───────────────────────────────────
    n_cmds = len(cache)
    h      = max(6, min(n_cmds * 0.55 + 3, 22))
    fig2, axes = plt.subplots(1, 2, figsize=(16, h))
    fig2.suptitle(
        f"P300 Speller — Matrices por comando  |  {category_name}\n"
        f"{best_clf} + {best_fs}  ·  AUC={best_auc:.3f}",
        fontsize=13, fontweight='bold', color=cat_color, y=1.01)

    plot_confusion_scoring(cache, X_all, best_clf, best_fs,
                            category_name, axes[0])
    plot_confusion_binary_per_cmd(cache, X_all, best_clf, best_fs,
                                   category_name, axes[1])

    plt.tight_layout()
    f2_path = output_dir / f"confusion_comandos_{category_name}.png"
    fig2.savefig(f2_path, dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"  ✓ {f2_path.name}")

    return best_clf, best_fs, best_auc


# ─────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────
def run_full_report(root_dir, output_dir=None):
    root_dir   = Path(root_dir)
    output_dir = Path(output_dir) if output_dir else root_dir / "graficas"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    all_cache    = {}   # para GENERAL
    all_commands = {}

    # ── Detectar carpetas ───────────────────────────────────────────────
    found = [(c, root_dir/c) for c in CATEGORY_DIRS if (root_dir/c).exists()]
    if not found:
        found = [("General", root_dir)]

    all_results = []

    for cat_name, cat_path in found:
        print(f"\n{'='*60}")
        print(f"  CATEGORÍA: {cat_name}")
        print(f"{'='*60}")

        commands = load_category(cat_path)
        if len(commands) < 2:
            print(f"  [!] Menos de 2 comandos, omitiendo.")
            continue

        # Acumular para GENERAL
        for k, v in commands.items():
            all_commands[f"{cat_name}_{k}"] = v

        # ── Build dataset (una sola vez) ────────────────────────────────
        cache, X_all, y_ref, cmd_list = build_dataset_all(commands)
        if X_all is None:
            continue

        # ── Evaluar todos clf × 7 feature sets ─────────────────────────
        print(f"\n  Evaluando clasificadores × {len(FEATURE_SETS)} feature sets...")
        results_df = evaluate_classifiers(X_all, y_ref)
        results_df["Categoria"] = cat_name
        all_results.append(results_df)
        results_df.to_csv(output_dir / f"metricas_{cat_name}.csv", index=False)

        # ── Generar figuras ─────────────────────────────────────────────
        generate_figures(cache, X_all, y_ref, results_df, commands,
                          cat_name, output_dir)

        # Guardar cache para GENERAL
        all_cache.update({f"{cat_name}_{k}": v for k, v in cache.items()})

    # ── EVALUACIÓN GENERAL ──────────────────────────────────────────────
    if len(all_commands) >= 4:
        print(f"\n{'='*60}")
        print(f"  EVALUACIÓN GENERAL ({len(all_commands)} comandos totales)")
        print(f"{'='*60}")

        # Reusar cache ya calculado: solo construir X_all por feature set
        print("  Construyendo datasets generales desde cache...")
        cmd_list_g = sorted(all_cache.keys())
        X_all_g = {}
        for fs_name in FEATURE_SETS:
            pos_X, neg_X = [], []
            for target_cmd in cmd_list_g:
                for feat in all_cache[target_cmd]:
                    pos_X.append(feat[fs_name])
                neg_pool = [feat[fs_name]
                            for o in cmd_list_g if o != target_cmd
                            for feat in all_cache[o]]
                mx = len(all_cache[target_cmd]) * MAX_NEG
                if len(neg_pool) > mx:
                    idx = np.random.choice(len(neg_pool), mx, replace=False)
                    neg_pool = [neg_pool[i] for i in idx]
                neg_X.extend(neg_pool)
            X = np.array(pos_X + neg_X)
            y = np.array([1]*len(pos_X) + [0]*len(neg_X))
            X_all_g[fs_name] = (X, y)

        y_ref_g = X_all_g[FEATURE_SETS[0]][1]
        print(f"  Dataset general: "
              f"{int(y_ref_g.sum())} targets | "
              f"{int((y_ref_g==0).sum())} non-targets")

        print(f"\n  Evaluando clasificadores × {len(FEATURE_SETS)} feature sets...")
        results_g = evaluate_classifiers(X_all_g, y_ref_g)
        results_g["Categoria"] = "GENERAL"
        all_results.append(results_g)
        results_g.to_csv(output_dir / "metricas_GENERAL.csv", index=False)

        generate_figures(all_cache, X_all_g, y_ref_g, results_g,
                          all_commands, "GENERAL", output_dir)

    # ── CSV consolidado ─────────────────────────────────────────────────
    if all_results:
        df_full = pd.concat(all_results, ignore_index=True)
        df_full.to_csv(output_dir / "metricas_consolidadas.csv", index=False)
        print(f"\n✓ CSV consolidado: metricas_consolidadas.csv")

    print(f"\n✓ Todos los reportes en: {output_dir}")


# ─────────────────────────────────────────
#  ENTRADA
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python p300_plots.py <ruta_carpeta_raiz> [ruta_salida]")
        sys.exit(1)
    run_full_report(
        root_dir   = sys.argv[1],
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    )