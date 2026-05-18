"""
Optimización de Features EEG con LDA + Random Forest
======================================================
Adaptado a archivos de features con formato: {Usuario}_{Clase}_{Trial}_features.csv
Cada CSV = una época ya procesada con 688 features por época.

Features detectados por canal (16 canales × 43 features):
  Estadísticas : mean, std, var, rms, skewness, kurtosis
  Frec. Abs    : delta_Abs, theta_Abs, alpha_Abs, beta_Abs, gamma_Abs
  Frec. Est    : delta_mean/std/var... theta_mean... alpha_mean... beta_*/gamma_*
  Frec. Rel    : delta_rel, theta_rel, alpha_rel, beta_rel, gamma_rel
  Wavelets     : wA5_energy, wA5_rel, wD1..wD5 _energy/_rel

Salidas por cada grupo de features:
  - CSV con componentes LDA + label
  - Reporte de importancia de componentes (Random Forest sobre LDA)
  - Gráficas: varianza explicada, scatter LDA, importancia de componentes

Uso:
    python optimizacion_lda.py --folder D:/EEG_Python/results/Usermar/Digit/Features
    python optimizacion_lda.py --folder ./features --output ./resultados_lda --tipo Digit
"""

import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────
#  Grupos de features (igual que tu ejemplo, adaptado a tus columnas)
# ─────────────────────────────────────────────────────────────
def get_feature_sets(all_cols):
    """
    Devuelve dict con los grupos de columnas por tipo de feature.
    Coincide con la lógica de tu Optimizaciondatos_lda.py original.
    """
    wA_keys  = ['wA5']
    wD_keys  = ['wD1','wD2','wD3','wD4','wD5']
    wav_keys = wA_keys + wD_keys

    sets = {
        "Estadisticas": [c for c in all_cols if any(k in c for k in
                         ['mean','std','var','rms','skewness','kurtosis'])
                         and not any(k in c for k in
                         ['delta_','theta_','alpha_','beta_','gamma_','wA','wD'])],

        "Frecuencias_Abs": [c for c in all_cols if any(k in c for k in
                            ['delta_Abs','theta_Abs','alpha_Abs','beta_Abs','gamma_Abs'])],

        "Frecuencias_Rel": [c for c in all_cols if any(k in c for k in
                            ['delta_rel','theta_rel','alpha_rel','beta_rel','gamma_rel'])],

        "Frecuencias_Est": [c for c in all_cols if any(k in c for k in
                            ['delta_mean','theta_mean','alpha_mean','beta_mean','gamma_mean',
                             'beta_std','gamma_std','beta_var','gamma_var',
                             'beta_rms','gamma_rms','beta_skewness','gamma_skewness',
                             'beta_kurtosis','gamma_kurtosis'])],

        "Wavelets": [c for c in all_cols if any(k in c for k in wav_keys)],

        "Frecuencias_Todas": [c for c in all_cols if any(k in c for k in
                              ['delta_','theta_','alpha_','beta_','gamma_'])],

        "TODAS": list(all_cols),
    }

    # Eliminar sets vacíos
    sets = {k: v for k, v in sets.items() if v}
    return sets


# ─────────────────────────────────────────────────────────────
#  Carga de datos
# ─────────────────────────────────────────────────────────────
def load_features_folder(folder_path, tipo_clase=None):
    """
    Lee todos los CSV de features en la carpeta.
    Nombre esperado: {Usuario}_{Clase}_{Trial}_features.csv
                  o: {Usuario}_{Clase}_{Trial}.csv
    Retorna:
        data_total : DataFrame con todas las features + columna 'label'
        archivos   : lista de rutas de archivos encontrados
        etiquetas  : lista de clases únicas (str)
        file_map   : dict {label -> lista de paths}
    """
    pattern = re.compile(r"^(.+?)_(\w+)_(\d+)(?:_features)?\.csv$", re.IGNORECASE)

    records  = []
    file_map = {}
    skipped  = []

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No hay CSV en: {folder_path}")

    print(f"\n{'='*60}")
    print(f"  OPTIMIZACIÓN LDA — EEG Features")
    print(f"{'='*60}")
    print(f"\n📂 Carpeta : {folder_path}")
    print(f"📄 Archivos: {len(csv_files)}\n")

    for fname in csv_files:
        m = pattern.match(fname)
        if not m:
            skipped.append(fname)
            continue

        usuario, clase, trial = m.group(1), m.group(2), int(m.group(3))
        fpath = os.path.join(folder_path, fname)

        try:
            df    = pd.read_csv(fpath)
            df['label']   = clase
            df['usuario'] = usuario
            df['trial']   = trial
            records.append(df)

            if clase not in file_map:
                file_map[clase] = []
            file_map[clase].append(fpath)

        except Exception as e:
            skipped.append(f"{fname} ({e})")

    if skipped:
        print(f"⚠️  Omitidos: {', '.join(skipped[:5])}")
    if not records:
        raise ValueError("No se cargó ningún archivo válido.")

    data_total = pd.concat(records, ignore_index=True)
    etiquetas  = sorted(data_total['label'].unique().tolist())

    print(f"✅ Épocas cargadas : {len(data_total)}")
    print(f"📊 Clases          : {etiquetas}")
    print(f"🧠 Features totales: {data_total.shape[1] - 3}  (sin label/usuario/trial)\n")

    return data_total, file_map, etiquetas


# ─────────────────────────────────────────────────────────────
#  LDA + Random Forest sobre un conjunto de features
# ─────────────────────────────────────────────────────────────
def run_lda_rf(X, y, feature_names, nombre_set, n_classes):
    """
    1. Aplica LDA para reducir dimensionalidad.
    2. Entrena Random Forest sobre componentes LDA.
    3. Retorna lda_model, X_lda, importancias ordenadas.
    """
    # LDA: máximo n_classes-1 componentes discriminantes
    n_components = min(n_classes - 1, X.shape[1], X.shape[0] - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    X_lda = lda.fit_transform(X, y)
    var_ratio = lda.explained_variance_ratio_

    print(f"  📐 Features entrada  : {X.shape[1]}")
    print(f"  📉 Componentes LDA   : {X_lda.shape[1]}")
    print(f"  📊 Varianza explicada: {[f'{v:.3f}' for v in var_ratio]}")
    print(f"  📊 Varianza acum.    : {var_ratio.cumsum()[-1]:.4f}\n")

    # Random Forest sobre componentes LDA
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_lda, y)

    importancias = rf.feature_importances_
    idx_ord      = np.argsort(importancias)[::-1]
    comp_names   = [f"LD{i+1}" for i in range(X_lda.shape[1])]
    top_n        = min(10, len(comp_names))

    print(f"  🌲 Top {top_n} componentes más importantes (Random Forest):")
    for i in range(top_n):
        c = idx_ord[i]
        print(f"     {i+1:2}. LD{c+1}  →  Importancia: {importancias[c]:.4f}  "
              f"| Varianza LDA: {var_ratio[c]:.4f}")

    # CV accuracy con LDA como clasificador directo
    lda_clf = LinearDiscriminantAnalysis()
    if len(np.unique(y)) > 1 and len(X) >= 5:
        cv_scores = cross_val_score(lda_clf, X, y, cv=min(5, len(X)//len(np.unique(y))),
                                    scoring='accuracy')
        print(f"\n  ✅ CV Accuracy LDA ({nombre_set}): "
              f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    return lda, X_lda, importancias, idx_ord, var_ratio, comp_names


# ─────────────────────────────────────────────────────────────
#  Guardar CSV con componentes LDA por archivo original
# ─────────────────────────────────────────────────────────────
def save_lda_csvs(file_map, features, lda_model, nombre_set, output_path, tipo_clase):
    """
    Para cada archivo original, transforma sus features con el LDA entrenado
    y guarda el CSV con componentes LD1, LD2... + label.
    """
    out_dir = os.path.join(output_path, tipo_clase, nombre_set)
    os.makedirs(out_dir, exist_ok=True)

    n_comp = lda_model.scalings_.shape[1]

    for label, paths in file_map.items():
        for fpath in paths:
            try:
                df   = pd.read_csv(fpath)
                # Verificar que tiene las columnas necesarias
                cols_ok = [c for c in features if c in df.columns]
                if len(cols_ok) < len(features):
                    print(f"  ⚠️  {os.path.basename(fpath)}: faltan {len(features)-len(cols_ok)} columnas")
                    cols_ok = [c for c in features if c in df.columns]

                X_file  = df[cols_ok].values.astype(float)
                X_trans = lda_model.transform(X_file)

                df_out  = pd.DataFrame(X_trans,
                                       columns=[f"LD{i+1}" for i in range(n_comp)])
                df_out['label'] = label

                # Nombre de salida
                base    = os.path.splitext(os.path.basename(fpath))[0]
                out_csv = os.path.join(out_dir, f"{base}_lda.csv")
                df_out.to_csv(out_csv, index=False)

            except Exception as e:
                print(f"  ❌ Error procesando {os.path.basename(fpath)}: {e}")

    print(f"  💾 CSVs LDA guardados en: {out_dir}")


# ─────────────────────────────────────────────────────────────
#  Visualizaciones
# ─────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

def plot_lda_results(X_lda, y, var_ratio, importancias, idx_ord,
                     comp_names, nombre_set, output_path, tipo_clase, label_encoder):
    """
    Genera 3 gráficas en un solo PNG:
      1. Varianza explicada por cada componente LDA
      2. Scatter LD1 vs LD2 (o LD1 solo si hay 1 componente)
      3. Importancia de componentes (Random Forest)
    """
    out_dir = os.path.join(output_path, tipo_clase, nombre_set)
    os.makedirs(out_dir, exist_ok=True)

    n_comp   = X_lda.shape[1]
    clases   = label_encoder.classes_
    y_labels = label_encoder.inverse_transform(y)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"LDA — {tipo_clase} | Grupo: {nombre_set}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── 1. Varianza explicada ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    comps = [f"LD{i+1}" for i in range(n_comp)]
    bars  = ax1.bar(comps, var_ratio * 100,
                    color=PALETTE[:n_comp], edgecolor="white")
    ax1.plot(comps, np.cumsum(var_ratio) * 100,
             "o--", color="black", linewidth=1.5, markersize=5, label="Acumulada")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_title("Varianza por componente")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, var_ratio * 100):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)

    # ── 2. Scatter LDA ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for i, cls in enumerate(clases):
        mask = y_labels == cls
        if n_comp >= 2:
            ax2.scatter(X_lda[mask, 0], X_lda[mask, 1],
                        label=f"Clase {cls}", color=PALETTE[i % len(PALETTE)],
                        alpha=0.75, edgecolors="white", linewidths=0.5, s=60)
            ax2.set_xlabel("LD1")
            ax2.set_ylabel("LD2")
        else:
            ax2.scatter(X_lda[mask, 0],
                        np.random.normal(i, 0.05, mask.sum()),
                        label=f"Clase {cls}", color=PALETTE[i % len(PALETTE)],
                        alpha=0.75, edgecolors="white", linewidths=0.5, s=60)
            ax2.set_xlabel("LD1")
            ax2.set_ylabel("Clase (offset)")

    ax2.set_title("Proyección LDA")
    ax2.legend(fontsize=8, loc="best")
    ax2.spines[["top","right"]].set_visible(False)

    # ── 3. Importancia componentes (RF) ───────────────────
    ax3 = fig.add_subplot(gs[2])
    top_n    = min(10, n_comp)
    top_idx  = idx_ord[:top_n]
    top_imp  = importancias[top_idx]
    top_comp = [comp_names[i] for i in top_idx]

    ax3.barh(top_comp[::-1], top_imp[::-1],
             color=[PALETTE[i % len(PALETTE)] for i in range(top_n)],
             edgecolor="white")
    ax3.set_xlabel("Importancia (Random Forest)")
    ax3.set_title("Importancia de componentes LDA")
    for i, (comp, val) in enumerate(zip(top_comp[::-1], top_imp[::-1])):
        ax3.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8)
    ax3.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"lda_{nombre_set}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Gráfica: lda_{nombre_set}.png")


def plot_resumen_cv(resumen, output_path, tipo_clase):
    """Gráfica de barras con CV Accuracy por grupo de features."""
    grupos = [r["grupo"] for r in resumen if not np.isnan(r["cv_acc"])]
    accs   = [r["cv_acc"] for r in resumen if not np.isnan(r["cv_acc"])]
    stds   = [r["cv_std"] for r in resumen if not np.isnan(r["cv_acc"])]

    if not grupos:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(grupos, accs, xerr=stds,
                   color=PALETTE[:len(grupos)], edgecolor="white",
                   height=0.55, capsize=4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CV Accuracy (LDA, 5-Fold)", fontsize=12)
    ax.set_title(f"Comparación de grupos de features — {tipo_clase}\n"
                 f"(LDA como clasificador)", fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(output_path, tipo_clase, "resumen_cv_accuracy.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Guardada: resumen_cv_accuracy.png")


def save_resumen_csv(resumen, output_path, tipo_clase):
    df = pd.DataFrame(resumen)
    p  = os.path.join(output_path, tipo_clase, "resumen_lda.csv")
    df.to_csv(p, index=False)
    print(f"  💾 resumen_lda.csv")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Optimización LDA + Random Forest sobre features EEG."
    )
    parser.add_argument("--folder", type=str,
                        default=r"D:/EEG_Python/results/Usermar/Char/features",
                        help="Carpeta con los CSV de features (Usuario_Clase_Trial_features.csv)")
    parser.add_argument("--output", type=str,
                        default=r"D:/EEG_Python/results/Usermar/Char/Resultados_LDA",
                        help="Carpeta de salida")
    parser.add_argument("--tipo",   type=str, default="Char",
                        help="Nombre del tipo de clase (default: Digit)")
    parser.add_argument("--grupos", nargs="+", default=None,
                        help="Seleccionar grupos específicos, ej: --grupos Estadisticas Wavelets")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, args.tipo), exist_ok=True)

    # 1. Cargar datos
    data_total, file_map, etiquetas = load_features_folder(args.folder, args.tipo)

    # Separar X / y
    meta_cols  = ['label', 'usuario', 'trial']
    feat_cols  = [c for c in data_total.columns if c not in meta_cols]
    X_total    = data_total[feat_cols].values.astype(float)
    y_raw      = data_total['label'].values

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_cls = len(le.classes_)

    # 2. Grupos de features
    all_feature_sets = get_feature_sets(feat_cols)
    if args.grupos:
        all_feature_sets = {k: v for k, v in all_feature_sets.items() if k in args.grupos}
        if not all_feature_sets:
            raise ValueError(f"Grupos no encontrados. Disponibles: {list(get_feature_sets(feat_cols).keys())}")

    print(f"🔍 Grupos a evaluar: {list(all_feature_sets.keys())}\n")

    # 3. Loop por grupo
    resumen = []
    for nombre_set, features in all_feature_sets.items():
        print(f"\n{'─'*60}")
        print(f"  GRUPO: {nombre_set}  ({len(features)} features)")
        print(f"{'─'*60}")

        if len(features) == 0:
            print("  (vacío, omitiendo)")
            continue

        X_sub = data_total[features].values.astype(float)

        # Manejo de NaN/Inf
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

        # LDA + RF
        lda_model, X_lda, importancias, idx_ord, var_ratio, comp_names = run_lda_rf(
            X_sub, y_enc, features, nombre_set, n_cls
        )

        # CV accuracy
        cv_acc, cv_std = np.nan, np.nan
        if len(np.unique(y_enc)) > 1 and len(X_sub) >= max(5, n_cls):
            lda_clf = LinearDiscriminantAnalysis()
            n_splits = min(5, len(X_sub) // n_cls)
            if n_splits >= 2:
                cv = cross_val_score(lda_clf, X_sub, y_enc,
                                     cv=n_splits, scoring='accuracy')
                cv_acc, cv_std = cv.mean(), cv.std()

        resumen.append({
            "grupo":       nombre_set,
            "n_features":  len(features),
            "n_comp_lda":  X_lda.shape[1],
            "var_acum":    round(float(var_ratio.sum()), 4),
            "cv_acc":      round(float(cv_acc), 4) if not np.isnan(cv_acc) else np.nan,
            "cv_std":      round(float(cv_std), 4) if not np.isnan(cv_std) else np.nan,
            "top_comp":    ", ".join([f"LD{idx_ord[i]+1}" for i in range(min(5, len(idx_ord)))]),
        })

        # Guardar CSVs transformados
        save_lda_csvs(file_map, features, lda_model, nombre_set, args.output, args.tipo)

        # Gráficas por grupo
        plot_lda_results(X_lda, y_enc, var_ratio, importancias, idx_ord,
                         comp_names, nombre_set, args.output, args.tipo, le)

    # 4. Resumen global
    print(f"\n{'='*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*60}")
    df_res = pd.DataFrame(resumen)
    print(df_res.to_string(index=False))

    save_resumen_csv(resumen, args.output, args.tipo)
    plot_resumen_cv(resumen, args.output, args.tipo)

    # Mejor grupo
    validos = [r for r in resumen if not np.isnan(r["cv_acc"])]
    if validos:
        mejor = max(validos, key=lambda r: r["cv_acc"])
        print(f"\n🏆 Mejor grupo: {mejor['grupo']}  "
              f"(CV Acc = {mejor['cv_acc']:.4f} ± {mejor['cv_std']:.4f})")

    print(f"\n✅ Todo guardado en: {args.output}/{args.tipo}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
