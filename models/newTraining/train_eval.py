# -*- coding: utf-8 -*-
"""
Arquitectura jerárquica de 4 modelos para clasificación EEG:
Evaluación multi-clasificador y multi-set de características por grupos.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

import config
from eeg_features import log_transform_power_columns


# ---------------------------------------------------------------------------
# Configuración por etapa
# ---------------------------------------------------------------------------
STAGE_CONFIG = {
    "super_clase": {"k_best": 250, "cv_splits": 5, "n_rep": 1},
    "Letters":     {"k_best": 150, "cv_splits": 5, "n_rep": 3},
    "Numbers":     {"k_best": 250, "cv_splits": 5, "n_rep": 3},
    "Controls":    {"k_best": 100, "cv_splits": 5, "n_rep": 1},
}


def build_classifiers(n_classes):
    """Retorna un diccionario con las instancias de los clasificadores a evaluar."""
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=1e-3, max_iter=2000, random_state=42)
    svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm_rbf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
    logreg = LogisticRegression(max_iter=2000, random_state=42)

    ensemble_mlp_svmrbf = VotingClassifier(
        estimators=[("mlp", mlp), ("svm_rbf", svm_rbf)],
        voting="soft"
    )
    ensemble_svmlin_logreg = VotingClassifier(
        estimators=[("svm_lin", svm_lin), ("logreg", logreg)],
        voting="soft"
    )

    return {
        "MLP": mlp,
        "SVM_Lineal": svm_lin,
        "SVM_RBF": svm_rbf,
        "LogReg": logreg,
        "Ensamble_MLP_SVMRBF": ensemble_mlp_svmrbf,
        "Ensamble_SVMLin_LogReg": ensemble_svmlin_logreg,
    }


def get_feature_sets(all_cols):
    """Genera y retorna los grupos/subconjuntos de características a evaluar."""
    wav_keys = ["wA", "wD"]
    return {
        "Estadisticas": [c for c in all_cols if any(k in c for k in
                         ["mean", "std", "var", "rms", "skewness", "kurtosis"])
                         and not any(k in c for k in
                         ["delta_", "theta_", "alpha_", "beta_", "gamma_", "wA", "wD"])],

        "Frecuencias_Abs": [c for c in all_cols if any(k in c for k in
                            ["delta_Abs", "theta_Abs", "alpha_Abs", "beta_Abs", "gamma_Abs"])],

        "Frecuencias_Rel": [c for c in all_cols if any(k in c for k in
                            ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"])],

        "Frecuencias_Est": [c for c in all_cols if any(k in c for k in
                            ["delta_mean", "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
                             "beta_std", "gamma_std", "beta_var", "gamma_var",
                             "beta_rms", "gamma_rms", "beta_skewness", "gamma_skewness",
                             "beta_kurtosis", "gamma_kurtosis"])],

        "Wavelets": [c for c in all_cols if any(k in c for k in wav_keys)],

        "Frecuencias_Todas": [c for c in all_cols if any(k in c for k in
                              ["delta_", "theta_", "alpha_", "beta_", "gamma_"])],

        "TODAS": list(all_cols),
    }


def average_repetitions_stage(df_stage, feature_cols, n_rep, seed=42):
    """Promedia grupos de n_rep trials de la misma clase."""
    if n_rep <= 1:
        return df_stage

    rng = np.random.default_rng(seed)
    rows = []
    for label, g in df_stage.groupby("label"):
        idxs = g.index.to_numpy().copy()
        rng.shuffle(idxs)
        n_groups = len(idxs) // n_rep
        for i in range(n_groups):
            sel = idxs[i * n_rep:(i + 1) * n_rep]
            chunk = df_stage.loc[sel]
            avg = chunk[feature_cols].mean(axis=0).to_dict()
            avg["label"] = label
            for col in ("usuario", "tpComando", "letra"):
                if col in chunk.columns:
                    avg[col] = chunk[col].iloc[0]
            avg["trial"] = f"avg{n_rep}_{i}"
            rows.append(avg)
    return pd.DataFrame(rows)


def build_pipeline(n_components, n_features_in, k_best, clf_instance):
    """Construye el pipeline con el clasificador recibido."""
    steps = [("scaler", StandardScaler())]
    
    if k_best is not None and k_best < n_features_in and k_best > 0:
        steps.append(("select", SelectKBest(f_classif, k=k_best)))
        
    if n_components > 0:
        steps.append(("lda", LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=n_components)))

    steps.append(("clf", clf_instance))
    return Pipeline(steps)


def evaluate_stage_all_combinations(df_stage, all_cols, stage_name, out_dir_figs):
    """
    Evalúa todas las combinaciones de [Set de Características x Clasificador] para un grupo/etapa.
    Retorna el reporte de métricas y guarda el modelo ganador para esa etapa.
    """
    cfg = STAGE_CONFIG[stage_name]
    df_stage = average_repetitions_stage(df_stage, all_cols, cfg.get("n_rep", 1))

    le = LabelEncoder()
    y = le.fit_transform(df_stage["label"])
    n_classes = len(le.classes_)

    class_counts = pd.Series(y).value_counts()
    min_c = class_counts.min()
    n_splits = max(2, min(cfg["cv_splits"], min_c))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_sets = get_feature_sets(all_cols)
    eval_results = []

    best_f1 = -1.0
    best_bundle = None
    best_oof_pred = None
    best_y_true = None

    print(f"\n--- Evaluación por Grupos y Sets: Etapa [{stage_name}] ({n_classes} clases) ---")

    for set_name, cols in feature_sets.items():
        if len(cols) == 0:
            continue

        X = df_stage[cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_features = X.shape[1]
        n_components = min(n_classes - 1, n_features)
        k_use = min(cfg["k_best"], n_features) if cfg["k_best"] else n_features

        classifiers = build_classifiers(n_classes)

        for clf_name, clf_inst in classifiers.items():
            pipe = build_pipeline(n_components, n_features, k_use, clone(clf_inst))

            try:
                acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
                f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)

                acc_m, acc_s = float(acc.mean()), float(acc.std())
                f1_m, f1_s = float(f1.mean()), float(f1.std())

                eval_results.append({
                    "Etapa": stage_name,
                    "Set_Caracteristicas": set_name,
                    "Clasificador": clf_name,
                    "N_Features": n_features,
                    "Acc_Mean": acc_m,
                    "Acc_Std": acc_s,
                    "F1_Macro_Mean": f1_m,
                    "F1_Macro_Std": f1_s
                })

                # Criterio para seleccionar el mejor modelo de la etapa
                if f1_m > best_f1:
                    best_f1 = f1_m
                    y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
                    
                    fitted_pipe = clone(pipe)
                    fitted_pipe.fit(X, y)

                    best_bundle = {
                        "pipeline": fitted_pipe,
                        "label_encoder": le,
                        "feature_columns": cols,
                        "set_name": set_name,
                        "clf_name": clf_name,
                        "n_classes": n_classes,
                        "cv_accuracy_mean": acc_m,
                        "cv_f1_macro_mean": f1_m,
                        "n_rep": cfg.get("n_rep", 1),
                    }
                    best_oof_pred = y_pred_oof
                    best_y_true = y

            except Exception as e:
                print(f"Error evaluando [{stage_name}] | Set: {set_name} | Clf: {clf_name}: {e}")

    # Guardar Matriz de Confusión del mejor modelo
    if best_bundle is not None:
        labels_str = list(le.classes_)
        title = (f"{stage_name} Best: {best_bundle['clf_name']} | {best_bundle['set_name']}\n"
                 f"(acc={best_bundle['cv_accuracy_mean']:.3f}, f1={best_bundle['cv_f1_macro_mean']:.3f})")
        _plot_confusion(best_y_true, best_oof_pred, labels_str, title,
                         os.path.join(out_dir_figs, f"cm_{stage_name}.png"))

        print(f"  --> [GANADOR {stage_name}] Clf: '{best_bundle['clf_name']}' | "
              f"Set: '{best_bundle['set_name']}' | Acc: {best_bundle['cv_accuracy_mean']:.3f} | F1: {best_bundle['cv_f1_macro_mean']:.3f}")

    return eval_results, best_bundle


def _plot_confusion(y_true, y_pred, labels_str, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_str))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig_size = max(6, len(labels_str) * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, xticklabels=labels_str, yticklabels=labels_str,
                cmap="Blues", vmin=0, vmax=1, cbar=True, square=True)
    plt.xlabel("Prediccción")
    plt.ylabel("Real")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_separability_super_clase(df, feature_cols, out_path):
    le = LabelEncoder()
    y = le.fit_transform(df["tpComando"])
    X = df[feature_cols].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cfg = STAGE_CONFIG["super_clase"]
    k_use = min(cfg["k_best"], X.shape[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    selector = SelectKBest(f_classif, k=k_use)
    Xsel = selector.fit_transform(Xs, y)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2)
    X2d = lda.fit_transform(Xsel, y)

    plt.figure(figsize=(8, 7))
    palette = sns.color_palette("Set2", n_colors=len(le.classes_))
    for i, cls in enumerate(le.classes_):
        mask = y == i
        plt.scatter(X2d[mask, 0], X2d[mask, 1], s=35, alpha=0.7,
                    color=palette[i], label=cls, edgecolor="white", linewidth=0.3)
    plt.xlabel("Componente LDA 1")
    plt.ylabel("Componente LDA 2")
    plt.title("Separabilidad entre Letters / Numbers / Controls\n(proyección LDA 2D)")
    plt.legend(title="tpComando")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_for_user(df_usuario, all_cols, usuario, figures_dir, models_dir):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*70}\nUSUARIO: {usuario}  ({len(df_usuario)} trials)\n{'='*70}")

    plot_separability_super_clase(
        df_usuario, all_cols,
        os.path.join(figures_dir, "separabilidad_super_clase.png")
    )

    all_evaluations = []

    # 1) SUPER-CLASE
    df_super = df_usuario.copy()
    df_super["label"] = df_super["tpComando"]
    evals_super, bundle_super = evaluate_stage_all_combinations(
        df_super, all_cols, "super_clase", figures_dir
    )
    all_evaluations.extend(evals_super)

    # 2) SUBGRUPOS (Letters, Numbers, Controls)
    bundles_by_group = {}
    for grupo in ["Letters", "Numbers", "Controls"]:
        df_grupo = df_usuario[df_usuario["tpComando"] == grupo].reset_index(drop=True)
        if df_grupo.empty:
            print(f"\n[AVISO] Usuario '{usuario}' no tiene trials de grupo '{grupo}'.")
            continue

        evals_g, bundle_g = evaluate_stage_all_combinations(
            df_grupo, all_cols, grupo, figures_dir
        )
        all_evaluations.extend(evals_g)
        bundles_by_group[grupo] = bundle_g

    # Exportar reporte detallado de este usuario
    eval_df = pd.DataFrame(all_evaluations)
    eval_df["Usuario"] = usuario
    eval_path = os.path.join(figures_dir, f"evaluacion_completa_{usuario}.csv")
    eval_df.to_csv(eval_path, index=False)

    # Guardar jerarquía con los mejores clasificadores seleccionados
    hierarchical_bundle = {
        "usuario": usuario,
        "super_clase": bundle_super,
        "por_grupo": bundles_by_group,
        "channel_names": config.CHANNEL_NAMES,
        "fs": config.FS,
    }
    out_path = os.path.join(models_dir, "hierarchical_bundle.joblib")
    joblib.dump(hierarchical_bundle, out_path)

    return {
        "usuario": usuario,
        "acc_super": bundle_super["cv_accuracy_mean"],
        "f1_super": bundle_super["cv_f1_macro_mean"],
        "best_clf_super": bundle_super["clf_name"],
        "best_set_super": bundle_super["set_name"],
        **{f"acc_{g}": b["cv_accuracy_mean"] for g, b in bundles_by_group.items()}
    }


def main():
    print("1) Cargando dataset de características...")
    df = pd.read_csv(config.FEATURES_CSV)

    meta_cols = ["usuario", "tpComando", "letra", "trial", "label"]
    all_cols = [c for c in df.columns if c not in meta_cols]
    df[all_cols] = log_transform_power_columns(df[all_cols], all_cols)

    usuarios = sorted(df["usuario"].unique())

    resumen_global = []
    for usuario in usuarios:
        df_usuario = df[df["usuario"] == usuario].reset_index(drop=True)
        figures_dir = os.path.join(config.FIGURES_DIR, usuario)
        models_dir = os.path.join(config.MODELS_DIR, usuario)

        info = train_for_user(df_usuario, all_cols, usuario, figures_dir, models_dir)
        resumen_global.append(info)

    resumen_df = pd.DataFrame(resumen_global)
    resumen_path = os.path.join(config.OUTPUT_DIR, "resumen_por_usuario.csv")
    resumen_df.to_csv(resumen_path, index=False)
    print(f"\n{'='*70}\nResumen guardado en: {resumen_path}")
    print(resumen_df.to_string(index=False))


if __name__ == "__main__":
    main()