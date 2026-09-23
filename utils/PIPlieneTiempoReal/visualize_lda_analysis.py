# -*- coding: utf-8 -*-
"""
VISUALIZACIÓN COMPLETA DEL ANÁLISIS LDA JERÁRQUICO
====================================================
Genera para cada usuario:
  1. Proyección LDA 2D - Super-clase (Letters / Numbers / Controls)
  2. Proyección LDA 2D - Letters
  3. Proyección LDA 2D - Numbers
  4. Proyección LDA 2D - Controls
  5. Tabla de "Feature Set ganador" por nivel (barplot comparativo)
  6. Curvas de Aprendizaje (accuracy vs. tamaño de entrenamiento) por nivel

Uso:
    python visualize_lda_analysis.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # sin GUI – graba archivos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

import config
from eeg_features import get_feature_sets, log_transform_power_columns
from train_hierarchical_optimized import build_pipeline_hierarchical, build_classifiers

# ──────────────────────────────────────────────────────────────────────────────
# PALETAS Y ESTILO
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")

PALETTE_SUPER   = ["#00D9FF", "#FF6B6B", "#B5FF4D"]          # 3 clases
PALETTE_LETTERS = sns.color_palette("husl", 26)               # 26 letras
PALETTE_NUMBERS = sns.color_palette("Set2", 10)               # 10 dígitos
PALETTE_CONTROLS = ["#FF9F43", "#EE5A24", "#1289A7"]          # 3 controles

FEATURE_SET_COLORS = {
    "Estadisticas":     "#F8EFBA",
    "Frecuencias_Abs":  "#78E08F",
    "Frecuencias_Rel":  "#38ADA9",
    "Frecuencias_Est":  "#079992",
    "Wavelets":         "#E55039",
    "Frecuencias_Todas":"#5758BB",
    "TODAS":            "#F0932B",
}

GROUP_PRETTY = {
    "super_clase": "Super-Clase\n(Letters / Numbers / Controls)",
    "Letters":     "Letters (A-Z)",
    "Numbers":     "Numbers (0-9)",
    "Controls":    "Controls",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _lda2d_from_bundle(bundle_level, X_raw, y_encoded, classes_str):
    """
    Extrae los 2 primeros componentes LDA del pipeline guardado en el bundle.
    Si ya tiene lda1/lda2 o lda, los usa directamente.
    Devuelve X_lda (N, 2) o None si <2 componentes disponibles.
    """
    pipe = bundle_level["pipeline"]

    # Reconstruimos la parte de pre-proceso hasta el último LDA
    preproc_steps = []
    lda_step = None
    for name, step in pipe.steps[:-1]:          # excluir clf
        preproc_steps.append((name, step))
        if name in ("lda", "lda2"):
            lda_step = (name, step)

    if lda_step is None:
        return None

    # Transformar con todo lo anterior excepto el último LDA
    X_t = X_raw.copy()
    for name, step in preproc_steps:
        if name == lda_step[0]:
            break
        X_t = step.transform(X_t)

    lda_fitted = lda_step[1]

    # Nos aseguramos de tener al menos 2 componentes
    n_comp_avail = lda_fitted.scalings_.shape[1]
    if n_comp_avail < 2:
        # proyecto al único componente disponible + 0
        X_lda_1 = lda_fitted.transform(X_t)     # (N, 1)
        X_lda = np.hstack([X_lda_1, np.zeros((len(X_lda_1), 1))])
    else:
        X_lda = lda_fitted.transform(X_t)[:, :2]

    return X_lda


def _fit_lda2d_fresh(X_raw, y, n_components=2, k_best=250):
    """
    Ajusta un LDA 2D fresco sobre X_raw, y para visualización.
    """
    steps = [("scaler", StandardScaler())]
    n_feat = X_raw.shape[1]
    if k_best is not None and k_best < n_feat:
        steps.append(("sel", SelectKBest(f_classif, k=k_best)))
        eff = k_best
    else:
        eff = n_feat

    n_cls = len(np.unique(y))
    n_comp = min(n_components, n_cls - 1, eff - 1)
    n_comp = max(1, n_comp)
    steps.append(("lda", LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto", n_components=n_comp)))

    pipe = Pipeline(steps)
    pipe.fit(X_raw, y)
    X_lda = pipe.named_steps["lda"].transform(
        pipe[:-1].transform(X_raw)
    )
    if X_lda.shape[1] < 2:
        X_lda = np.hstack([X_lda, np.zeros((len(X_lda), 1))])
    return X_lda[:, :2]


def _scatter_lda(ax, X_lda, y, class_names, palette, title,
                  evr=None, max_scatter_pts=3000, alpha=0.65,
                  show_centroids=True, show_ellipses=True):
    """Scatter LDA 2D estilizado."""
    rng = np.random.default_rng(42)

    xlabel = "LD1"
    ylabel = "LD2"
    if evr is not None and not np.any(np.isnan(evr)):
        xlabel += f" ({evr[0]:.1f}%)"
        ylabel += f" ({evr[1]:.1f}%)"

    for i, cls_name in enumerate(class_names):
        mask = y == i
        Xc = X_lda[mask]
        if len(Xc) == 0:
            continue

        color = palette[i % len(palette)]

        if len(Xc) > max_scatter_pts:
            idx = rng.choice(len(Xc), max_scatter_pts, replace=False)
            Xc_plot = Xc[idx]
        else:
            Xc_plot = Xc

        ax.scatter(Xc_plot[:, 0], Xc_plot[:, 1],
                   c=[color], alpha=alpha, s=12, linewidths=0,
                   label=cls_name, zorder=2)

        if show_centroids:
            mu = Xc.mean(axis=0)
            ax.scatter(*mu, marker="*", s=220, c=[color],
                       edgecolors="white", linewidths=0.8, zorder=5)
            if len(class_names) <= 12:
                ax.annotate(cls_name, mu, fontsize=7, color="white",
                            ha="center", va="bottom",
                            fontweight="bold",
                            xytext=(0, 6), textcoords="offset points",
                            zorder=6)

        if show_ellipses and len(Xc) >= 5:
            cov = np.cov(Xc.T)
            if cov.ndim == 2 and cov.shape == (2, 2):
                _draw_ellipse(ax, Xc.mean(axis=0), cov, color, alpha=0.18)

    ax.set_xlabel(xlabel, color="#AAAAAA", fontsize=9)
    ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_facecolor("#111111")
    ax.grid(True, alpha=0.08, lw=0.5)


def _draw_ellipse(ax, mean, cov, color, alpha=0.15, n_std=2):
    """Dibuja una elipse de confianza 2D."""
    from matplotlib.patches import Ellipse
    import numpy.linalg as la
    try:
        vals, vecs = la.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * n_std * np.sqrt(np.abs(vals))
        ell = Ellipse(xy=mean, width=w, height=h, angle=theta,
                      facecolor=color, alpha=alpha, edgecolor=color,
                      linewidth=0.8)
        ax.add_patch(ell)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 1: PROYECCIONES LDA  (2×2 subplot)
# ──────────────────────────────────────────────────────────────────────────────

def plot_lda_projections(df_usuario, bundle, out_dir, usuario):
    """Genera figura con 4 subplots de proyección LDA 2D."""
    fig = plt.figure(figsize=(20, 17), facecolor="#0D0D0D")
    fig.suptitle(
        f"Proyecciones LDA – Usuario: {usuario}",
        fontsize=16, fontweight="bold", color="white", y=0.97
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── 1. SUPER-CLASE ──────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, 0])
    super_b = bundle["super_clase"]
    df_s = df_usuario.copy()
    df_s["label"] = df_s["tpComando"]
    fc_s = super_b["feature_columns"]
    df_s[fc_s] = log_transform_power_columns(df_s[fc_s], fc_s)

    le_s = LabelEncoder()
    y_s = le_s.fit_transform(df_s["label"])
    X_s = np.nan_to_num(df_s[fc_s].to_numpy(), nan=0.0)

    X_lda_s = _lda2d_from_bundle(super_b, X_s, y_s, le_s.classes_)
    if X_lda_s is None:
        X_lda_s = _fit_lda2d_fresh(X_s, y_s, n_components=2, k_best=250)

    evr_s = None
    for name, step in super_b["pipeline"].steps:
        if name in ("lda", "lda2"):
            if hasattr(step, "explained_variance_ratio_") and \
               step.explained_variance_ratio_ is not None and \
               len(step.explained_variance_ratio_) >= 1:
                evr_s = step.explained_variance_ratio_[:2] * 100
            break

    _scatter_lda(ax_s, X_lda_s, y_s, le_s.classes_,
                 PALETTE_SUPER,
                 f"Super-Clase  ·  {super_b['feature_set']}\n"
                 f"Clf: {super_b['clasificador']}  |  "
                 f"f1={super_b.get('f1_macro_cv', 0):.3f}",
                 evr=evr_s, show_ellipses=True, alpha=0.70)

    handles = [mpatches.Patch(facecolor=PALETTE_SUPER[i], label=cls)
               for i, cls in enumerate(le_s.classes_)]
    ax_s.legend(handles=handles, fontsize=8, loc="upper right",
                framealpha=0.25, labelcolor="white")

    # ── 2. LETTERS ──────────────────────────────────────────────────────────
    ax_l = fig.add_subplot(gs[0, 1])
    let_b = bundle["por_grupo"]["Letters"]
    df_l = df_usuario[df_usuario["tpComando"] == "Letters"].reset_index(drop=True)
    fc_l = let_b["feature_columns"]
    df_l = df_l.copy()
    df_l[fc_l] = log_transform_power_columns(df_l[fc_l], fc_l)

    le_l = LabelEncoder()
    y_l = le_l.fit_transform(df_l["label"])
    X_l = np.nan_to_num(df_l[fc_l].to_numpy(), nan=0.0)

    X_lda_l = _lda2d_from_bundle(let_b, X_l, y_l, le_l.classes_)
    if X_lda_l is None:
        X_lda_l = _fit_lda2d_fresh(X_l, y_l, n_components=2, k_best=250)

    pal_l = [PALETTE_LETTERS[i % len(PALETTE_LETTERS)]
             for i in range(len(le_l.classes_))]
    _scatter_lda(ax_l, X_lda_l, y_l, le_l.classes_, pal_l,
                 f"Letters  ·  {let_b['feature_set']}\n"
                 f"Clf: {let_b['clasificador']}  |  "
                 f"f1={let_b.get('f1_macro_cv', 0):.3f}",
                 show_ellipses=False, show_centroids=True, alpha=0.55)
    if len(le_l.classes_) <= 14:
        handles_l = [mpatches.Patch(facecolor=pal_l[i], label=cls)
                     for i, cls in enumerate(le_l.classes_)]
        ax_l.legend(handles=handles_l, fontsize=6, ncol=2, loc="upper right",
                    framealpha=0.2, labelcolor="white")

    # ── 3. NUMBERS ──────────────────────────────────────────────────────────
    ax_n = fig.add_subplot(gs[1, 0])
    num_b = bundle["por_grupo"]["Numbers"]
    df_n = df_usuario[df_usuario["tpComando"] == "Numbers"].reset_index(drop=True)
    fc_n = num_b["feature_columns"]
    df_n = df_n.copy()
    df_n[fc_n] = log_transform_power_columns(df_n[fc_n], fc_n)

    le_n = LabelEncoder()
    y_n = le_n.fit_transform(df_n["label"])
    X_n = np.nan_to_num(df_n[fc_n].to_numpy(), nan=0.0)

    X_lda_n = _lda2d_from_bundle(num_b, X_n, y_n, le_n.classes_)
    if X_lda_n is None:
        X_lda_n = _fit_lda2d_fresh(X_n, y_n, n_components=2, k_best=250)

    pal_n = [PALETTE_NUMBERS[i % len(PALETTE_NUMBERS)]
             for i in range(len(le_n.classes_))]
    _scatter_lda(ax_n, X_lda_n, y_n, le_n.classes_, pal_n,
                 f"Numbers  ·  {num_b['feature_set']}\n"
                 f"Clf: {num_b['clasificador']}  |  "
                 f"f1={num_b.get('f1_macro_cv', 0):.3f}",
                 show_ellipses=True, show_centroids=True, alpha=0.70)
    handles_n = [mpatches.Patch(facecolor=pal_n[i], label=cls)
                 for i, cls in enumerate(le_n.classes_)]
    ax_n.legend(handles=handles_n, fontsize=8, ncol=2, loc="upper right",
                framealpha=0.2, labelcolor="white")

    # ── 4. CONTROLS ─────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ctrl_b = bundle["por_grupo"]["Controls"]
    df_c = df_usuario[df_usuario["tpComando"] == "Controls"].reset_index(drop=True)
    fc_c = ctrl_b["feature_columns"]
    df_c = df_c.copy()
    df_c[fc_c] = log_transform_power_columns(df_c[fc_c], fc_c)

    le_c = LabelEncoder()
    y_c = le_c.fit_transform(df_c["label"])
    X_c = np.nan_to_num(df_c[fc_c].to_numpy(), nan=0.0)

    X_lda_c = _lda2d_from_bundle(ctrl_b, X_c, y_c, le_c.classes_)
    if X_lda_c is None:
        X_lda_c = _fit_lda2d_fresh(X_c, y_c, n_components=2, k_best=250)

    pal_c = [PALETTE_CONTROLS[i % len(PALETTE_CONTROLS)]
             for i in range(len(le_c.classes_))]
    _scatter_lda(ax_c, X_lda_c, y_c, le_c.classes_, pal_c,
                 f"Controls  ·  {ctrl_b['feature_set']}\n"
                 f"Clf: {ctrl_b['clasificador']}  |  "
                 f"f1={ctrl_b.get('f1_macro_cv', 0):.3f}",
                 show_ellipses=True, show_centroids=True, alpha=0.75)
    handles_c = [mpatches.Patch(facecolor=pal_c[i], label=cls)
                 for i, cls in enumerate(le_c.classes_)]
    ax_c.legend(handles=handles_c, fontsize=9, loc="upper right",
                framealpha=0.2, labelcolor="white")

    out_path = os.path.join(out_dir, f"lda_projections_{usuario}.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    print(f"   [OK] LDA projections -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 2: FEATURE SET GANADOR POR NIVEL
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_winner_analysis(results_df, bundle, out_dir, usuario):
    """
    3 visualizaciones en una figura:
      - Barplot top Feature Sets por nivel (f1_macro_mean)
      - Heatmap Clf x FeatureSet para super-clase
      - Tabla resumen de ganadores
    """
    fig = plt.figure(figsize=(22, 14), facecolor="#0D0D0D")
    fig.suptitle(
        f"Análisis de Feature Sets y Clasificadores – Usuario: {usuario}",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38)

    niveles = ["super_clase", "Letters", "Numbers", "Controls"]
    grupo_map = {
        "super_clase": "Letters/Numbers/Controls",
        "Letters": "Letters",
        "Numbers": "Numbers",
        "Controls": "Controls",
    }

    # ── Panel A: Barplot F1 por Feature Set (agrupado por nivel) ───────────
    ax_bar = fig.add_subplot(gs[0, :])

    plot_data = []
    for nivel in niveles:
        grp = grupo_map[nivel]
        sub = results_df[results_df["grupo"] == grp].copy()
        if sub.empty:
            continue
        agg = sub.groupby("feature_set")["f1_macro_mean"].max().reset_index()
        agg["nivel"] = GROUP_PRETTY.get(nivel, nivel)
        plot_data.append(agg)

    if plot_data:
        plot_df = pd.concat(plot_data, ignore_index=True)
        feature_sets_ordered = plot_df.groupby("feature_set")["f1_macro_mean"].mean()\
                                      .sort_values(ascending=False).index.tolist()
        niveles_uniq = plot_df["nivel"].unique().tolist()

        width = 0.18
        colors_nivel = ["#00D9FF", "#B5FF4D", "#FF6B6B", "#FFD700"]

        for i, nivel_str in enumerate(niveles_uniq):
            sub_n = plot_df[plot_df["nivel"] == nivel_str]
            vals = [sub_n[sub_n["feature_set"] == fs]["f1_macro_mean"].values[0]
                    if fs in sub_n["feature_set"].values else 0
                    for fs in feature_sets_ordered]
            xpos = np.arange(len(feature_sets_ordered)) + i * width
            bars = ax_bar.bar(xpos, vals, width=width * 0.9,
                              color=colors_nivel[i % len(colors_nivel)],
                              alpha=0.82, label=nivel_str, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.008,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=6, color="white", rotation=45)

        center_x = np.arange(len(feature_sets_ordered)) + width * (len(niveles_uniq) - 1) / 2
        ax_bar.set_xticks(center_x)
        ax_bar.set_xticklabels(feature_sets_ordered, rotation=22, ha="right",
                               fontsize=9, color="white")
        ax_bar.set_ylabel("F1-Macro (max por feature set)", color="#AAAAAA", fontsize=10)
        ax_bar.set_title("Mejor F1-Macro por Feature Set en cada Nivel",
                         fontsize=12, fontweight="bold", color="white")
        ax_bar.legend(fontsize=9, framealpha=0.2, labelcolor="white")
        ax_bar.set_facecolor("#111111")
        ax_bar.grid(axis="y", alpha=0.12, lw=0.6)
        ax_bar.tick_params(colors="#888888")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#333333")

    # ── Panel B: Heatmap Clf × FeatureSet (super-clase) ─────────────────────
    ax_heat = fig.add_subplot(gs[1, 0])
    sub_s = results_df[results_df["grupo"] == "Letters/Numbers/Controls"]
    if not sub_s.empty:
        pivot = sub_s.pivot_table(index="clasificador", columns="feature_set",
                                  values="f1_macro_mean", aggfunc="max")
        cmap = sns.color_palette("viridis", as_cmap=True)
        sns.heatmap(pivot, ax=ax_heat, cmap=cmap, annot=True, fmt=".2f",
                    linewidths=0.5, linecolor="#1A1A1A",
                    annot_kws={"size": 7, "color": "white"},
                    cbar_kws={"shrink": 0.8})
        ax_heat.set_title("Heatmap F1-Macro: Clf × Feature Set\n(Super-Clase)",
                          fontsize=10, fontweight="bold", color="white")
        ax_heat.set_xlabel("Feature Set", fontsize=8, color="#AAAAAA")
        ax_heat.set_ylabel("Clasificador", fontsize=8, color="#AAAAAA")
        ax_heat.tick_params(colors="white", labelsize=7)
        ax_heat.set_facecolor("#111111")

    # ── Panel C: Tabla resumen de ganadores ──────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis("off")

    winner_rows = []
    for nivel in niveles:
        grp = grupo_map[nivel]
        sub = results_df[results_df["grupo"] == grp]
        if sub.empty:
            continue
        best = sub.loc[sub["f1_macro_mean"].idxmax()]
        winner_rows.append([
            GROUP_PRETTY.get(nivel, nivel).replace("\n", " "),
            best["feature_set"],
            f"{best['n_features']}",
            best["clasificador"][:18],
            f"{best['f1_macro_mean']:.3f}",
            f"{best['accuracy_mean']:.3f}",
        ])

    if winner_rows:
        col_labels = ["Nivel", "Feature Set", "N.feat", "Clasificador", "F1", "Acc"]
        table = ax_tbl.table(
            cellText=winner_rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("#1A1A2E" if row % 2 == 0 else "#16213E")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333333")
            if row == 0:
                cell.set_facecolor("#0F3460")
                cell.set_text_props(fontweight="bold", color="#00D9FF")
        ax_tbl.set_title("Resumen de Ganadores por Nivel",
                         fontsize=10, fontweight="bold", color="white", pad=12)

    out_path = os.path.join(out_dir, f"feature_winner_analysis_{usuario}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    print(f"   [OK] Feature winner analysis -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 3: CURVAS DE APRENDIZAJE
# ──────────────────────────────────────────────────────────────────────────────

def _compute_learning_curve(df_subset, feature_cols, n_components, clf,
                             clf_name, group_label, k_best=250, cv_splits=5,
                             n_points=8):
    """Calcula learning curve (train_sizes, train_scores, val_scores)."""
    from sklearn.base import clone

    le = LabelEncoder()
    y = le.fit_transform(df_subset["label"])
    X = np.nan_to_num(df_subset[feature_cols].to_numpy(), nan=0.0)
    n_classes = len(np.unique(y))

    if X.shape[0] < 30 or n_classes < 2:
        return None, None, None, None

    class_counts = pd.Series(y).value_counts()
    min_class = class_counts.min()
    n_splits = max(2, min(cv_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = build_pipeline_hierarchical(clone(clf), n_components, X.shape[1], k_best)

    min_sz = max(n_splits * n_classes, 20)
    max_sz = int(X.shape[0] * (n_splits - 1) / n_splits)
    if min_sz >= max_sz:
        return None, None, None, None

    train_sizes = np.unique(np.linspace(min_sz, max_sz, n_points, dtype=int))

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipe, X, y,
            train_sizes=train_sizes,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
            return_times=False,
        )
        return train_sizes_abs, train_scores, val_scores, le.classes_
    except Exception as e:
        print(f"   [LC ERROR] {group_label} x {clf_name}: {e}")
        return None, None, None, None


def _prep_superclase(df):
    df2 = df.copy()
    df2["label"] = df2["tpComando"]
    return df2


def _style_lc_ax(ax):
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, alpha=0.10, lw=0.5)


def plot_learning_curves(df_usuario, bundle, results_df, out_dir, usuario):
    """4 subplots de curvas de aprendizaje, uno por nivel."""
    classifiers = build_classifiers(40)

    grupos_config = {
        "super_clase": {
            "df_fn": _prep_superclase,
            "bundle_key": "super_clase",
            "is_nested": False,
            "n_comp": 2,
        },
        "Letters": {
            "df_fn": lambda df: df[df["tpComando"] == "Letters"].copy().reset_index(drop=True),
            "bundle_key": "Letters",
            "is_nested": True,
            "n_comp": None,
        },
        "Numbers": {
            "df_fn": lambda df: df[df["tpComando"] == "Numbers"].copy().reset_index(drop=True),
            "bundle_key": "Numbers",
            "is_nested": True,
            "n_comp": None,
        },
        "Controls": {
            "df_fn": lambda df: df[df["tpComando"] == "Controls"].copy().reset_index(drop=True),
            "bundle_key": "Controls",
            "is_nested": True,
            "n_comp": None,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor="#0D0D0D")
    fig.suptitle(
        f"Curvas de Aprendizaje (F1-Macro CV) – Usuario: {usuario}",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )
    axes = axes.flatten()

    for ax_idx, (grupo_key, cfg) in enumerate(grupos_config.items()):
        ax = axes[ax_idx]
        ax.set_facecolor("#111111")

        # Obtener bundle correcto
        bkey = cfg["bundle_key"]
        if cfg["is_nested"]:
            bnd = bundle["por_grupo"][bkey]
        else:
            bnd = bundle[bkey]

        clf_name = bnd["clasificador"]
        feature_cols = bnd["feature_columns"]

        # Preparar df
        df_sub = cfg["df_fn"](df_usuario)
        fc = [c for c in feature_cols if c in df_sub.columns]
        if not fc:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            continue

        df_sub[fc] = log_transform_power_columns(df_sub[fc], fc)

        n_clases = df_sub["label"].nunique()
        n_comp = cfg["n_comp"] if cfg["n_comp"] else max(1, min(n_clases - 1, 50))

        clf = classifiers.get(clf_name)
        if clf is None:
            clf = list(classifiers.values())[0]

        ts, tr_sc, val_sc, classes = _compute_learning_curve(
            df_sub, fc, n_comp, clf, clf_name,
            group_label=grupo_key, cv_splits=5, n_points=9
        )

        if ts is None:
            ax.text(0.5, 0.5, "Datos insuficientes para LC",
                    ha="center", va="center", color="#FF6B6B",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(GROUP_PRETTY.get(grupo_key, grupo_key),
                         color="white", fontsize=11, fontweight="bold")
            _style_lc_ax(ax)
            continue

        tr_mean = tr_sc.mean(axis=1)
        tr_std  = tr_sc.std(axis=1)
        val_mean = val_sc.mean(axis=1)
        val_std  = val_sc.std(axis=1)

        ax.plot(ts, tr_mean, "o-", color="#00D9FF", lw=2.0,
                markersize=5, label="Entrenamiento", zorder=4)
        ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.18, color="#00D9FF")

        ax.plot(ts, val_mean, "s--", color="#FF6B6B", lw=2.0,
                markersize=5, label="Validación (CV)", zorder=4)
        ax.fill_between(ts, val_mean - val_std, val_mean + val_std,
                        alpha=0.18, color="#FF6B6B")

        best_val = val_mean.max()
        best_ts  = ts[val_mean.argmax()]
        ax.axhline(best_val, ls=":", color="#B5FF4D", lw=1.2, alpha=0.7)
        ax.annotate(f"Best: {best_val:.3f}", xy=(best_ts, best_val),
                    xytext=(10, -14), textcoords="offset points",
                    fontsize=8, color="#B5FF4D", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#B5FF4D", lw=0.8))

        ax.set_xlabel("# Muestras de Entrenamiento", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("F1-Macro", color="#AAAAAA", fontsize=9)
        f1_cv = bnd.get("f1_macro_cv", 0.0)
        ax.set_title(
            f"{GROUP_PRETTY.get(grupo_key, grupo_key)}\n"
            f"FS: {bnd['feature_set']} | Clf: {clf_name[:20]} | F1-cv={f1_cv:.3f}",
            fontsize=9, fontweight="bold", color="white"
        )
        ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
        ax.set_ylim(0, 1.05)
        _style_lc_ax(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"learning_curves_{usuario}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    print(f"   [OK] Learning curves -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 4: PANEL RESUMEN EJECUTIVO
# ──────────────────────────────────────────────────────────────────────────────

def plot_executive_summary(bundle, results_df, out_dir, usuario):
    """Panel con barras horizontales de accuracy & F1 + tabla de ganadores."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0D0D0D")
    fig.suptitle(
        f"Panel Resumen Ejecutivo – {usuario}",
        fontsize=14, fontweight="bold", color="white"
    )

    ax = axes[0]
    ax.set_facecolor("#111111")

    nivel_labels, accs, f1s = [], [], []
    nivel_map = {
        "super_clase": "Letters/Numbers/Controls",
        "Letters": "Letters",
        "Numbers": "Numbers",
        "Controls": "Controls",
    }
    for nivel, grp in nivel_map.items():
        sub = results_df[results_df["grupo"] == grp]
        if sub.empty:
            continue
        best = sub.loc[sub["f1_macro_mean"].idxmax()]
        nivel_labels.append(GROUP_PRETTY.get(nivel, nivel).replace("\n", " "))
        accs.append(best["accuracy_mean"])
        f1s.append(best["f1_macro_mean"])

    y = np.arange(len(nivel_labels))
    bar_h = 0.35
    bars_acc = ax.barh(y + bar_h / 2, accs, bar_h, color="#00D9FF",
                       alpha=0.82, label="Accuracy", zorder=3)
    bars_f1  = ax.barh(y - bar_h / 2, f1s,  bar_h, color="#FF6B6B",
                       alpha=0.82, label="F1-Macro", zorder=3)

    for bar, v in zip(bars_acc, accs):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9,
                color="white", fontweight="bold")
    for bar, v in zip(bars_f1, f1s):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=9,
                color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(nivel_labels, color="white", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score", color="#AAAAAA", fontsize=9)
    ax.set_title("Accuracy y F1-Macro (mejor combinación por nivel)",
                 color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(axis="x", alpha=0.12, lw=0.6)
    ax.axvline(0.5, color="#555555", lw=0.8, ls="--")

    # Panel B: tabla detallada
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    ax2.axis("off")

    winner_info = []
    for nivel, grp in nivel_map.items():
        sub = results_df[results_df["grupo"] == grp]
        if sub.empty:
            continue
        best = sub.loc[sub["f1_macro_mean"].idxmax()]
        winner_info.append({
            "Nivel": GROUP_PRETTY.get(nivel, nivel).replace("\n", " "),
            "Feature Set": best["feature_set"],
            "N. Features": int(best["n_features"]),
            "Clasificador": best["clasificador"][:20],
            "F1-Macro": f"{best['f1_macro_mean']:.4f}",
            "Accuracy": f"{best['accuracy_mean']:.4f}",
        })

    if winner_info:
        tbl_df = pd.DataFrame(winner_info)
        col_labels = list(tbl_df.columns)
        cell_text  = tbl_df.values.tolist()

        table = ax2.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.15, 1.0, 0.7],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (row, col), cell in table.get_celld().items():
            bg = "#1A1A2E" if row % 2 == 0 else "#16213E"
            cell.set_facecolor(bg)
            cell.set_text_props(color="white")
            cell.set_edgecolor("#333333")
            if row == 0:
                cell.set_facecolor("#0F3460")
                cell.set_text_props(fontweight="bold", color="#00D9FF", fontsize=9)
        ax2.set_title("Ganadores por Nivel – Detalle Completo",
                      color="white", fontsize=10, fontweight="bold", pad=14)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"executive_summary_{usuario}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    print(f"   [OK] Executive summary -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*90)
    print("VISUALIZACION LDA JERARQUICO – ANALISIS COMPLETO")
    print("="*90)

    print(f"\n1) Cargando dataset: {config.FEATURES_CSV} ...")
    df = pd.read_csv(config.FEATURES_CSV)
    meta_cols = ["usuario", "tpComando", "letra", "trial", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = log_transform_power_columns(df[feature_cols], feature_cols)
    print(f"   OK {df.shape[0]} trials | {len(feature_cols)} features | "
          f"Usuarios: {df['usuario'].unique().tolist()}")

    usuarios = sorted(df["usuario"].unique())

    for usuario in usuarios:
        print(f"\n{'='*90}")
        print(f"USUARIO: {usuario}")
        print("="*90)

        usuario_models_dir = os.path.join(config.MODELS_DIR, usuario)
        bundle_path = os.path.join(usuario_models_dir, "hierarchical_bundle_optimizado.joblib")
        results_csv_path = os.path.join(config.OUTPUT_DIR, usuario, "evaluacion_jerarquica_completa.csv")

        if not os.path.exists(bundle_path):
            print(f"   Bundle no encontrado: {bundle_path}  — saltando.")
            continue
        if not os.path.exists(results_csv_path):
            print(f"   CSV evaluacion no encontrado: {results_csv_path}  — saltando.")
            continue

        print(f"\n2) Cargando bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)

        print(f"3) Cargando resultados: {results_csv_path}")
        results_df = pd.read_csv(results_csv_path)

        df_usuario = df[df["usuario"] == usuario].reset_index(drop=True)
        print(f"   Trials del usuario: {len(df_usuario)}")

        out_dir = os.path.join(config.FIGURES_DIR, usuario)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n4) Generando figuras en: {out_dir}")

        # Figura 1: Proyecciones LDA
        print("   -> Proyecciones LDA 2D ...")
        try:
            plot_lda_projections(df_usuario, bundle, out_dir, usuario)
        except Exception as e:
            print(f"   [ERROR] LDA projections: {e}")
            import traceback; traceback.print_exc()

        # Figura 2: Feature Winner Analysis
        print("   -> Analisis de Feature Set ganador ...")
        try:
            plot_feature_winner_analysis(results_df, bundle, out_dir, usuario)
        except Exception as e:
            print(f"   [ERROR] Feature winner: {e}")
            import traceback; traceback.print_exc()

        # Figura 3: Curvas de Aprendizaje
        print("   -> Curvas de Aprendizaje (puede tardar ~2-5 min) ...")
        try:
            plot_learning_curves(df_usuario, bundle, results_df, out_dir, usuario)
        except Exception as e:
            print(f"   [ERROR] Learning curves: {e}")
            import traceback; traceback.print_exc()

        # Figura 4: Panel Resumen Ejecutivo
        print("   -> Panel resumen ejecutivo ...")
        try:
            plot_executive_summary(bundle, results_df, out_dir, usuario)
        except Exception as e:
            print(f"   [ERROR] Executive summary: {e}")
            import traceback; traceback.print_exc()

        print(f"\n  Todas las figuras guardadas en: {out_dir}")

    print("\n" + "="*90)
    print("VISUALIZACION COMPLETADA")
    print("="*90)


if __name__ == "__main__":
    main()
