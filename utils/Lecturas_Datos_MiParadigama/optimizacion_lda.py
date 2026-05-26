"""
Optimización de Features EEG con LDA + Random Forest
=====================================================
Recorre automáticamente la estructura de carpetas:

    BASE_RESULTS/
    └── {Usuario}/               ← detectado automáticamente
        ├── Digit/
        │   └── features/        ← CSV: {Usuario}_{Clase}_{Trial}_features.csv
        ├── Char/
        │   └── features/
        └── Comando/
            └── features/

El script:
  1. Detecta todos los usuarios dentro de BASE_RESULTS
  2. Por cada usuario detecta los tipos de clase (Digit, Char, Comando, etc.)
  3. Busca la carpeta features/ dentro de cada tipo
  4. Lee los CSV, detecta las clases automáticamente desde el nombre del archivo
  5. Aplica LDA para N clases (lo que haya) por cada grupo de features
  6. Guarda CSV transformados + gráficas + resumen

Salidas:
    BASE_RESULTS/
    └── {Usuario}/
        └── {TipoClase}/
            └── Resultados_LDA/
                ├── {GrupoFeatures}/
                │   ├── {Usuario}_{Clase}_{Trial}_features_lda.csv
                │   └── lda_{GrupoFeatures}.png
                ├── resumen_lda.csv
                └── resumen_cv_accuracy.png
    └── resumen_lda_global.csv   ← resumen de todos los usuarios y tipos

Uso:
    python optimizacion_lda.py
    python optimizacion_lda.py --base D:/EEG_Python/results
    python optimizacion_lda.py --usuarios Usermar UserArcane --tipos Digit Char
    python optimizacion_lda.py --grupos Estadisticas Wavelets
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────────────────────
#  Configuración de rutas
# ─────────────────────────────────────────────────────────────
BASE_RESULTS     = r"D:/EEG_Python/results"
TIPOS_VALIDOS    = ["Digit", "Char", "Comando"]
CARPETA_FEATURES = "features"
CARPETA_LDA      = "Resultados_LDA"

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]


# ─────────────────────────────────────────────────────────────
#  Grupos de features
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  Descubrimiento automático de carpetas
# ─────────────────────────────────────────────────────────────
def discover_structure(base_results, usuarios_filter=None, tipos_filter=None):
    """
    Recorre:  base_results / {usuario} / {tipo} / features/
    Retorna lista de trabajos pendientes.
    """
    if not os.path.isdir(base_results):
        raise FileNotFoundError(f"No existe la carpeta base: {base_results}")

    candidatos = sorted([
        d for d in os.listdir(base_results)
        if os.path.isdir(os.path.join(base_results, d))
    ])
    if usuarios_filter:
        candidatos = [u for u in candidatos if u in usuarios_filter]

    trabajos = []
    for usuario in candidatos:
        usuario_path = os.path.join(base_results, usuario)

        tipos = sorted([
            d for d in os.listdir(usuario_path)
            if os.path.isdir(os.path.join(usuario_path, d))
            and d in (tipos_filter or TIPOS_VALIDOS)
        ])

        for tipo in tipos:
            features_path = os.path.join(usuario_path, tipo, CARPETA_FEATURES)
            output_path   = os.path.join(usuario_path, tipo, CARPETA_LDA)

            if not os.path.isdir(features_path):
                print(f"  ⚠️  Sin carpeta features/: {features_path}")
                continue

            csv_count = len([f for f in os.listdir(features_path)
                             if f.lower().endswith(".csv")])
            if csv_count == 0:
                print(f"  ⚠️  Sin CSV en: {features_path}")
                continue

            trabajos.append({
                "usuario":       usuario,
                "tipo":          tipo,
                "features_path": features_path,
                "output_path":   output_path,
                "csv_count":     csv_count,
            })

    return trabajos


# ─────────────────────────────────────────────────────────────
#  Carga de datos
# ─────────────────────────────────────────────────────────────
def load_features_folder(folder_path):
    """
    Lee todos los CSV de features en la carpeta.
    Nombre: {Usuario}_{Clase}_{Trial}_features.csv  o  {Usuario}_{Clase}_{Trial}.csv
    La etiqueta (label) = campo Clase del nombre del archivo.
    Funciona con cualquier número de clases.
    """
    pattern = re.compile(r"^(.+?)_(\w+)_(\d+)(?:_features)?\.csv$", re.IGNORECASE)

    records  = []
    file_map = {}
    skipped  = []

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".csv")])

    for fname in csv_files:
        m = pattern.match(fname)
        if not m:
            skipped.append(fname)
            continue

        usuario, clase, trial = m.group(1), m.group(2), int(m.group(3))
        fpath = os.path.join(folder_path, fname)

        try:
            df            = pd.read_csv(fpath)
            df["label"]   = clase
            df["usuario"] = usuario
            df["trial"]   = trial
            records.append(df)
            file_map.setdefault(clase, []).append(fpath)
        except Exception as e:
            skipped.append(f"{fname} ({e})")

    if skipped:
        print(f"    ⚠️  Omitidos: {', '.join(skipped[:5])}")
    if not records:
        raise ValueError(f"No se cargó ningún archivo válido en {folder_path}")

    data_total = pd.concat(records, ignore_index=True)
    etiquetas  = sorted(data_total["label"].unique().tolist())
    meta_cols  = ["label", "usuario", "trial"]
    feat_cols  = [c for c in data_total.columns if c not in meta_cols]

    return data_total, file_map, etiquetas, feat_cols


# ─────────────────────────────────────────────────────────────
#  LDA + Random Forest
# ─────────────────────────────────────────────────────────────
def run_lda_rf(X, y_enc, nombre_set, n_classes):
    n_components = min(n_classes - 1, X.shape[1], X.shape[0] - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    X_lda     = lda.fit_transform(X, y_enc)
    var_ratio = lda.explained_variance_ratio_

    print(f"    📐 Features entrada  : {X.shape[1]}")
    print(f"    📉 Componentes LDA   : {X_lda.shape[1]}  (máx. N_clases-1 = {n_classes-1})")
    print(f"    📊 Varianza explicada: {[f'{v:.3f}' for v in var_ratio]}")
    print(f"    📊 Varianza acumulada: {var_ratio.cumsum()[-1]:.4f}\n")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_lda, y_enc)

    importancias = rf.feature_importances_
    idx_ord      = np.argsort(importancias)[::-1]
    comp_names   = [f"LD{i+1}" for i in range(X_lda.shape[1])]
    top_n        = min(10, len(comp_names))

    print(f"    🌲 Top {top_n} componentes más importantes (Random Forest):")
    for i in range(top_n):
        c = idx_ord[i]
        print(f"       {i+1:2}. LD{c+1}  →  Importancia: {importancias[c]:.4f}  "
              f"| Varianza LDA: {var_ratio[c]:.4f}")

    return lda, X_lda, importancias, idx_ord, var_ratio, comp_names


# ─────────────────────────────────────────────────────────────
#  Guardar CSV LDA por archivo original
# ─────────────────────────────────────────────────────────────
def save_lda_csvs(file_map, features, lda_model, nombre_set, output_path):
    out_dir = os.path.join(output_path, nombre_set)
    os.makedirs(out_dir, exist_ok=True)
    n_comp = lda_model.scalings_.shape[1]

    for label, paths in file_map.items():
        for fpath in paths:
            try:
                df      = pd.read_csv(fpath)
                cols_ok = [c for c in features if c in df.columns]
                if len(cols_ok) < len(features):
                    print(f"    ⚠️  {os.path.basename(fpath)}: "
                          f"faltan {len(features)-len(cols_ok)} columnas")

                X_file  = df[cols_ok].values.astype(float)
                X_file  = np.nan_to_num(X_file, nan=0.0, posinf=0.0, neginf=0.0)
                X_trans = lda_model.transform(X_file)

                df_out          = pd.DataFrame(X_trans,
                                               columns=[f"LD{i+1}" for i in range(n_comp)])
                df_out["label"] = label

                base    = os.path.splitext(os.path.basename(fpath))[0]
                out_csv = os.path.join(out_dir, f"{base}_lda.csv")
                df_out.to_csv(out_csv, index=False)

            except Exception as e:
                print(f"    ❌ {os.path.basename(fpath)}: {e}")

    print(f"    💾 CSVs LDA → {out_dir}")


# ─────────────────────────────────────────────────────────────
#  Visualizaciones
# ─────────────────────────────────────────────────────────────
def plot_lda_results(X_lda, y_enc, var_ratio, importancias, idx_ord,
                     comp_names, nombre_set, output_path,
                     tipo_clase, label_encoder, usuario):

    out_dir = os.path.join(output_path, nombre_set)
    os.makedirs(out_dir, exist_ok=True)

    n_comp   = X_lda.shape[1]
    clases   = label_encoder.classes_
    y_labels = label_encoder.inverse_transform(y_enc)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"LDA — {usuario} | {tipo_clase} | Grupo: {nombre_set}  "
                 f"({len(clases)} clases)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # 1. Varianza explicada
    ax1         = fig.add_subplot(gs[0])
    comp_labels = [f"LD{i+1}" for i in range(n_comp)]
    bars        = ax1.bar(comp_labels, var_ratio * 100,
                          color=PALETTE[:n_comp], edgecolor="white")
    ax1.plot(comp_labels, np.cumsum(var_ratio) * 100,
             "o--", color="black", linewidth=1.5, markersize=5, label="Acumulada")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_title("Varianza por componente")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, var_ratio * 100):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax1.spines[["top","right"]].set_visible(False)
    if n_comp > 8:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # 2. Scatter LDA (LD1 vs LD2)
    ax2 = fig.add_subplot(gs[1])
    for i, cls in enumerate(clases):
        mask = y_labels == cls
        if n_comp >= 2:
            ax2.scatter(X_lda[mask, 0], X_lda[mask, 1],
                        label=str(cls), color=PALETTE[i % len(PALETTE)],
                        alpha=0.75, edgecolors="white", linewidths=0.5, s=55)
            ax2.set_xlabel("LD1");  ax2.set_ylabel("LD2")
        else:
            ax2.scatter(X_lda[mask, 0],
                        np.random.normal(i, 0.05, mask.sum()),
                        label=str(cls), color=PALETTE[i % len(PALETTE)],
                        alpha=0.75, edgecolors="white", linewidths=0.5, s=55)
            ax2.set_xlabel("LD1");  ax2.set_ylabel("Clase (offset)")

    ax2.set_title(f"Proyección LDA  ({len(clases)} clases)")
    ax2.legend(fontsize=7, loc="best", ncol=max(1, len(clases)//10), title="Clase")
    ax2.spines[["top","right"]].set_visible(False)

    # 3. Importancia componentes (RF)
    ax3      = fig.add_subplot(gs[2])
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
    print(f"    📊 Gráfica: lda_{nombre_set}.png")


def plot_resumen_cv(resumen, output_path, tipo_clase, usuario):
    grupos = [r["grupo"] for r in resumen if not np.isnan(r["cv_acc"])]
    accs   = [r["cv_acc"] for r in resumen if not np.isnan(r["cv_acc"])]
    stds   = [r["cv_std"] for r in resumen if not np.isnan(r["cv_acc"])]

    if not grupos:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(grupos) * 0.6)))
    bars = ax.barh(grupos, accs, xerr=stds,
                   color=PALETTE[:len(grupos)], edgecolor="white",
                   height=0.55, capsize=4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CV Accuracy (LDA, 5-Fold)", fontsize=12)
    ax.set_title(f"Comparación de grupos de features\n{usuario} — {tipo_clase}  "
                 f"(LDA como clasificador)", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(output_path, "resumen_cv_accuracy.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    📊 Guardada: resumen_cv_accuracy.png")


def save_resumen_csv(resumen, output_path):
    df = pd.DataFrame(resumen)
    p  = os.path.join(output_path, "resumen_lda.csv")
    df.to_csv(p, index=False)
    print(f"    💾 resumen_lda.csv")


# ─────────────────────────────────────────────────────────────
#  Procesamiento de un trabajo (un usuario + un tipo)
# ─────────────────────────────────────────────────────────────
def procesar_trabajo(trabajo, grupos_filter=None):
    usuario       = trabajo["usuario"]
    tipo          = trabajo["tipo"]
    features_path = trabajo["features_path"]
    output_path   = trabajo["output_path"]

    print(f"\n  {'─'*56}")
    print(f"  📂 {usuario} / {tipo}")
    print(f"  {'─'*56}")
    print(f"  Features : {features_path}")
    print(f"  Salida   : {output_path}\n")

    os.makedirs(output_path, exist_ok=True)

    # Cargar datos
    try:
        data_total, file_map, etiquetas, feat_cols = load_features_folder(features_path)
    except ValueError as e:
        print(f"    ❌ {e}")
        return None

    n_cls = len(etiquetas)
    print(f"    ✅ Épocas: {len(data_total)}  |  "
          f"Clases ({n_cls}): {etiquetas}")
    print(f"    🧠 Features totales: {len(feat_cols)}\n")

    if n_cls < 2:
        print(f"    ⚠️  Se necesitan al menos 2 clases para LDA. Omitiendo.")
        return None

    # Codificar etiquetas
    le    = LabelEncoder()
    y_enc = le.fit_transform(data_total["label"].values)

    # Grupos de features
    all_feature_sets = get_feature_sets(feat_cols)
    if grupos_filter:
        all_feature_sets = {k: v for k, v in all_feature_sets.items()
                            if k in grupos_filter}
    if not all_feature_sets:
        print(f"    ⚠️  Sin grupos de features válidos. Omitiendo.")
        return None

    print(f"    🔍 Grupos a evaluar: {list(all_feature_sets.keys())}\n")

    # Loop por grupo de features
    resumen = []
    for nombre_set, features in all_feature_sets.items():
        print(f"\n    {'·'*52}")
        print(f"    GRUPO: {nombre_set}  ({len(features)} features)")
        print(f"    {'·'*52}")

        if not features:
            print("    (vacío, omitiendo)")
            continue

        X_sub = data_total[features].values.astype(float)
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            lda_model, X_lda, importancias, idx_ord, var_ratio, comp_names = \
                run_lda_rf(X_sub, y_enc, nombre_set, n_cls)
        except Exception as e:
            print(f"    ❌ LDA falló: {e}")
            continue

        # CV accuracy
        cv_acc, cv_std = np.nan, np.nan
        if len(X_sub) >= max(5, n_cls * 2):
            n_splits = min(5, len(X_sub) // n_cls)
            if n_splits >= 2:
                try:
                    cv = cross_val_score(LinearDiscriminantAnalysis(),
                                         X_sub, y_enc,
                                         cv=n_splits, scoring="accuracy")
                    cv_acc, cv_std = cv.mean(), cv.std()
                    print(f"\n    ✅ CV Accuracy LDA: {cv_acc:.4f} ± {cv_std:.4f}\n")
                except Exception as e:
                    print(f"    ⚠️  CV falló: {e}")

        resumen.append({
            "grupo":      nombre_set,
            "n_features": len(features),
            "n_clases":   n_cls,
            "n_comp_lda": X_lda.shape[1],
            "var_acum":   round(float(var_ratio.sum()), 4),
            "cv_acc":     round(float(cv_acc), 4) if not np.isnan(cv_acc) else np.nan,
            "cv_std":     round(float(cv_std), 4) if not np.isnan(cv_std) else np.nan,
            "top_comp":   ", ".join([f"LD{idx_ord[i]+1}"
                                     for i in range(min(5, len(idx_ord)))]),
        })

        save_lda_csvs(file_map, features, lda_model, nombre_set, output_path)
        plot_lda_results(X_lda, y_enc, var_ratio, importancias, idx_ord,
                         comp_names, nombre_set, output_path, tipo, le, usuario)

    # Resumen del trabajo
    if resumen:
        print(f"\n    {'='*52}")
        print(f"    RESUMEN — {usuario} / {tipo}")
        print(f"    {'='*52}")
        print(pd.DataFrame(resumen).to_string(index=False))

        save_resumen_csv(resumen, output_path)
        plot_resumen_cv(resumen, output_path, tipo, usuario)

        validos = [r for r in resumen if not np.isnan(r["cv_acc"])]
        if validos:
            mejor = max(validos, key=lambda r: r["cv_acc"])
            print(f"\n    🏆 Mejor grupo: {mejor['grupo']}  "
                  f"(CV Acc = {mejor['cv_acc']:.4f} ± {mejor['cv_std']:.4f})")

    print(f"\n    ✅ Guardado en: {output_path}\n")
    return resumen


# ─────────────────────────────────────────────────────────────
#  LDA GENERAL — todas las clases de todos los tipos juntas
# ─────────────────────────────────────────────────────────────
def procesar_lda_general(usuario, trabajos_usuario, base_results, grupos_filter=None):
    """
    Carga TODOS los CSV de features de todos los tipos (Digit + Char + Comando + ...)
    del mismo usuario y aplica un LDA general con todas las clases combinadas.

    La etiqueta de cada época será:  {tipo}_{clase}
    Ej: Digit_0, Digit_1, Char_A, Char_C, Comando_izq, ...

    Salida:
        BASE_RESULTS/{usuario}/LDA_General/
            ├── {GrupoFeatures}/
            │   ├── {usuario}_*_lda.csv
            │   └── lda_{GrupoFeatures}.png
            ├── resumen_lda.csv
            └── resumen_cv_accuracy.png
    """
    output_path = os.path.join(base_results, usuario, "LDA_General")
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  🌐 LDA GENERAL — {usuario}")
    print(f"     Tipos combinados: {[t['tipo'] for t in trabajos_usuario]}")
    print(f"{'═'*62}\n")

    # ── 1. Cargar y concatenar todos los tipos ──────────────
    all_dfs   = []
    file_map  = {}   # etiqueta_combinada -> [paths]

    for trabajo in trabajos_usuario:
        tipo          = trabajo["tipo"]
        features_path = trabajo["features_path"]

        try:
            data_tipo, fm_tipo, etiquetas_tipo, _ = load_features_folder(features_path)
        except ValueError as e:
            print(f"  ⚠️  {tipo}: {e}")
            continue

        # Prefixar etiqueta con el tipo para distinguir clases entre tipos
        # Ej: clase "0" de Digit → "Digit_0", clase "A" de Char → "Char_A"
        data_tipo["label"]    = tipo + "_" + data_tipo["label"].astype(str)
        data_tipo["tipo_src"] = tipo
        all_dfs.append(data_tipo)

        for label_orig, paths in fm_tipo.items():
            label_comb = f"{tipo}_{label_orig}"
            file_map.setdefault(label_comb, []).extend(paths)

    if not all_dfs:
        print(f"  ❌ No se cargó ningún tipo para {usuario}. Omitiendo LDA General.")
        return None

    data_total = pd.concat(all_dfs, ignore_index=True)
    etiquetas  = sorted(data_total["label"].unique().tolist())
    n_cls      = len(etiquetas)
    meta_cols  = ["label", "usuario", "trial", "tipo_src"]
    feat_cols  = [c for c in data_total.columns if c not in meta_cols]

    print(f"  ✅ Total épocas    : {len(data_total)}")
    print(f"  📊 Clases totales  : {n_cls}  → {etiquetas}")
    print(f"  🧠 Features totales: {len(feat_cols)}\n")

    if n_cls < 2:
        print(f"  ⚠️  Se necesitan al menos 2 clases. Omitiendo.")
        return None

    # Verificar que las columnas de features son consistentes entre tipos
    # (si un tipo tiene más features que otro, usar solo las comunes)
    feat_cols_common = [c for c in feat_cols
                        if data_total[c].notna().sum() == len(data_total)]
    if len(feat_cols_common) < len(feat_cols):
        print(f"  ⚠️  Features con NaN en algunos tipos: "
              f"{len(feat_cols)-len(feat_cols_common)} eliminadas. "
              f"Usando {len(feat_cols_common)} features comunes.\n")
        feat_cols = feat_cols_common

    # ── 2. Codificar etiquetas ──────────────────────────────
    le    = LabelEncoder()
    y_enc = le.fit_transform(data_total["label"].values)

    # ── 3. Grupos de features ───────────────────────────────
    all_feature_sets = get_feature_sets(feat_cols)
    if grupos_filter:
        all_feature_sets = {k: v for k, v in all_feature_sets.items()
                            if k in grupos_filter}
    if not all_feature_sets:
        print(f"  ⚠️  Sin grupos de features válidos.")
        return None

    print(f"  🔍 Grupos a evaluar: {list(all_feature_sets.keys())}\n")

    # ── 4. Loop por grupo de features ──────────────────────
    resumen = []
    for nombre_set, features in all_feature_sets.items():
        print(f"\n  {'·'*56}")
        print(f"  GRUPO: {nombre_set}  ({len(features)} features)")
        print(f"  {'·'*56}")

        if not features:
            print("  (vacío, omitiendo)")
            continue

        X_sub = data_total[features].values.astype(float)
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            lda_model, X_lda, importancias, idx_ord, var_ratio, comp_names = \
                run_lda_rf(X_sub, y_enc, nombre_set, n_cls)
        except Exception as e:
            print(f"  ❌ LDA falló: {e}")
            continue

        # CV accuracy
        cv_acc, cv_std = np.nan, np.nan
        if len(X_sub) >= max(5, n_cls * 2):
            n_splits = min(5, len(X_sub) // n_cls)
            if n_splits >= 2:
                try:
                    cv = cross_val_score(LinearDiscriminantAnalysis(),
                                         X_sub, y_enc,
                                         cv=n_splits, scoring="accuracy")
                    cv_acc, cv_std = cv.mean(), cv.std()
                    print(f"\n  ✅ CV Accuracy LDA General: {cv_acc:.4f} ± {cv_std:.4f}\n")
                except Exception as e:
                    print(f"  ⚠️  CV falló: {e}")

        resumen.append({
            "grupo":      nombre_set,
            "n_features": len(features),
            "n_clases":   n_cls,
            "n_comp_lda": X_lda.shape[1],
            "var_acum":   round(float(var_ratio.sum()), 4),
            "cv_acc":     round(float(cv_acc), 4) if not np.isnan(cv_acc) else np.nan,
            "cv_std":     round(float(cv_std), 4) if not np.isnan(cv_std) else np.nan,
            "top_comp":   ", ".join([f"LD{idx_ord[i]+1}"
                                     for i in range(min(5, len(idx_ord)))]),
        })

        # Guardar CSVs (con etiqueta tipo_clase combinada)
        save_lda_csvs(file_map, features, lda_model, nombre_set, output_path)

        # Gráfica con título especial para LDA General
        plot_lda_results(X_lda, y_enc, var_ratio, importancias, idx_ord,
                         comp_names, nombre_set, output_path,
                         "General", le, usuario)

    # ── 5. Resumen LDA General ──────────────────────────────
    if resumen:
        print(f"\n  {'='*56}")
        print(f"  RESUMEN LDA GENERAL — {usuario}")
        print(f"  {'='*56}")
        print(pd.DataFrame(resumen).to_string(index=False))

        save_resumen_csv(resumen, output_path)
        plot_resumen_cv(resumen, output_path, "General (todos los tipos)", usuario)

        validos = [r for r in resumen if not np.isnan(r["cv_acc"])]
        if validos:
            mejor = max(validos, key=lambda r: r["cv_acc"])
            print(f"\n  🏆 Mejor grupo (LDA General): {mejor['grupo']}  "
                  f"(CV Acc = {mejor['cv_acc']:.4f} ± {mejor['cv_std']:.4f})")

    print(f"\n  ✅ LDA General guardado en: {output_path}\n")
    return resumen


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Optimización LDA + Random Forest — multi-usuario, multi-tipo, N clases."
    )
    parser.add_argument("--base", type=str,
                        default=BASE_RESULTS,
                        help=f"Carpeta raíz con los usuarios (default: {BASE_RESULTS})")
    parser.add_argument("--usuarios", nargs="+", default=None,
                        help="Filtrar usuarios, ej: --usuarios Usermar UserArcane")
    parser.add_argument("--tipos", nargs="+", default=None,
                        help="Filtrar tipos de clase, ej: --tipos Digit Char Comando")
    parser.add_argument("--grupos", nargs="+", default=None,
                        help="Filtrar grupos de features, ej: --grupos Estadisticas Wavelets")
    args = parser.parse_args()

    print(f"\n{'#'*62}")
    print(f"#  OPTIMIZACIÓN LDA — MULTI-USUARIO / MULTI-TIPO / N CLASES")
    print(f"{'#'*62}")
    print(f"\n  Base       : {args.base}")
    print(f"  Usuarios   : {args.usuarios or 'todos'}")
    print(f"  Tipos      : {args.tipos    or TIPOS_VALIDOS}")
    print(f"  Grupos feat: {args.grupos   or 'todos'}\n")

    # Descubrir estructura
    trabajos = discover_structure(
        args.base,
        usuarios_filter=args.usuarios,
        tipos_filter=args.tipos or TIPOS_VALIDOS,
    )

    if not trabajos:
        print("❌ No se encontró ninguna combinación usuario/tipo con datos.")
        return

    print(f"\n  Trabajos encontrados: {len(trabajos)}")
    for t in trabajos:
        print(f"    • {t['usuario']:<15} / {t['tipo']:<10}  ({t['csv_count']} CSV)")

    # Procesar cada trabajo (LDA por tipo)
    resumen_global = []
    for trabajo in trabajos:
        res = procesar_trabajo(trabajo, grupos_filter=args.grupos)
        if res:
            for r in res:
                r["usuario"] = trabajo["usuario"]
                r["tipo"]    = trabajo["tipo"]
                resumen_global.append(r)

    # ── LDA GENERAL por usuario (todas las clases de todos los tipos juntas) ──
    usuarios_unicos = sorted(set(t["usuario"] for t in trabajos))
    for usuario in usuarios_unicos:
        trabajos_usuario = [t for t in trabajos if t["usuario"] == usuario]
        res_gen = procesar_lda_general(
            usuario, trabajos_usuario, args.base,
            grupos_filter=args.grupos
        )
        if res_gen:
            for r in res_gen:
                r["usuario"] = usuario
                r["tipo"]    = "LDA_General"
                resumen_global.append(r)

    # Resumen global
    if resumen_global:
        print(f"\n{'#'*62}")
        print(f"#  RESUMEN GLOBAL")
        print(f"{'#'*62}")
        df_global = pd.DataFrame(resumen_global)
        cols_show = ["usuario","tipo","grupo","n_clases","n_comp_lda",
                     "var_acum","cv_acc","cv_std"]
        print(df_global[[c for c in cols_show if c in df_global.columns]]
              .sort_values(["usuario","tipo","cv_acc"], ascending=[True,True,False])
              .to_string(index=False))

        out_global = os.path.join(args.base, "resumen_lda_global.csv")
        df_global.to_csv(out_global, index=False)
        print(f"\n  💾 Resumen global: {out_global}")

    print(f"\n{'#'*62}")
    print(f"#  FINALIZADO")
    print(f"{'#'*62}\n")


if __name__ == "__main__":
    main()
# Solo un usuario
##python optimizacion_lda.py --base "D:/EEG_Python/results" --usuarios Usermar
# Varios usuarios y solo algunos tipos
#python optimizacion_lda.py --base "D:/EEG_Python/results" --usuarios Usermar UserArcane --tipos Digit Char
# Solo algunos grupos de features
#python optimizacion_lda.py --base "D:/EEG_Python/results" --grupos Estadisticas Wavelets TODAS

# d:\EEG_Python\.venv\Scripts\python.exe d:/EEG_Python/Lecturas_Datos_MiParadigama/optimizacion_lda.py --usuarios UserOmar