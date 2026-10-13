# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO JERÁRQUICO CON COLOR Y MODELOS LDA POR COLOR

Flujo:
1. Evalúa combinaciones de clasificadores y features con validación cruzada
2. Guarda tabla completa de resultados (evaluacion_jerarquica_completa.csv)
3. Identifica modelos ganadores para color, super-grupo y comando fino por color
4. Entrena la cascada: color → super-grupo del color → comando fino del color/grupo
5. Guarda bundle final listo para tiempo real (hierarchical_bundle_optimizado.joblib)

Protocolo asumido:
- 2 flashes por grupo en 2 segundos de grabación
- 14 canales EEG (F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4)
- fs=128 Hz
"""

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier 
from lda_utils import SafeLDA
import config
from eeg_features import get_feature_sets, log_transform_power_columns


def normalize_string_columns(df, columns):
    """Convierte columnas de metadatos a texto para evitar mezclas int/str en LabelEncoder."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else str(x))
    return df


def validate_color_labels(df):
    """Verifica que el dataset ya incluya colores asignados por data_loader."""
    if "color" not in df.columns:
        raise ValueError(
            "El dataset no tiene la columna 'color'. Regeneralo con data_loader.py."
        )
    invalid_colors = sorted(set(df["color"].dropna().astype(str)) - {"Green", "Blue"})
    if invalid_colors:
        raise ValueError(
            "El dataset contiene colores no validos: "
            f"{invalid_colors}. Regeneralo con data_loader.py."
        )


# ===========================================================================================
# PASO 1: DEFINIR CLASIFICADORES (6 total)
# ===========================================================================================
def build_classifiers(n_classes):
    """6 clasificadores a evaluar"""
    ramdomState = config.RANDOM_STATE
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=1e-3, max_iter=2000, random_state=ramdomState)
    svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=ramdomState)
    svm_rbf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=ramdomState)
    logreg = LogisticRegression(max_iter=2000, random_state=ramdomState)
    rff= RandomForestClassifier(n_estimators=100, random_state=ramdomState)
    xgc = XGBClassifier(n_estimators=100, random_state=ramdomState)

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
        "RF":rff,
        "XGBC":xgc,
    }


# ===========================================================================================
# PASO 2: PIPELINE CON LDA OPTIMIZADO PARA ARQUITECTURA JERÁRQUICA
# ===========================================================================================
def build_pipeline_hierarchical(clf, n_components, n_features_in, k_best=None,
                                 use_double_lda=True):
    """
    Pipeline dinámico adaptado para evitar fallos en clasificación binaria (Color).
    Si n_components == 1 (Verde vs Azul), usa 1 solo LDA directo para evitar redundancia.
    Si n_components >= 2 (Comandos finos), activa la cascada doble.
    """
    steps = [("scaler", StandardScaler())]

    # 1. Selección de las mejores características iniciales
    if k_best is not None and k_best < n_features_in:
        steps.append(("select", SelectKBest(f_classif, k=k_best)))
        effective_features = k_best
    else:
        # Protección dinámica: si k_best es None, usamos el 80% de lo que entra
        k_dinamico = max(1, int(n_features_in * 0.8))
        steps.append(("select", SelectKBest(f_classif, k=k_dinamico)))
        effective_features = k_dinamico

    # Asegurar componentes seguros
    n_comp_safe = min(n_components, effective_features - 1, 200)
    n_comp_safe = max(1, n_comp_safe)

    # 2. LÓGICA DINÁMICA DE LDA:
    # Si es clasificación binaria (n_comp_safe == 1), forzamos un solo LDA directo
    if n_comp_safe == 1:
        steps.append(("lda_directo", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=1)))
    
    # Si es multiclase (n_comp_safe >= 2) y está activo el doble LDA, usamos la cascada
    elif use_double_lda and n_comp_safe >= 2:
        # LDA 1 (amplio)
        n_comp_lda1 = min(n_comp_safe * 3, effective_features - 1, 200)
        n_comp_lda1 = max(n_comp_safe + 1, n_comp_lda1)
        steps.append(("lda1", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=n_comp_lda1)))

        # Re-escalar para estabilidad numérica en la cascada
        steps.append(("scaler2", StandardScaler()))

        # LDA 2 (fino)
        steps.append(("lda2", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=n_comp_safe)))
    else:
        # Modo simple de respaldo
        steps.append(("lda_simple", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=n_comp_safe)))

    from sklearn.base import clone
    steps.append(("clf", clone(clf)))

    return Pipeline(steps)


# ===========================================================================================
# PASO 3: EVALUACIÓN DE 1 COMBINACIÓN (clf × feature_set × grupo)
# ===========================================================================================
def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    """Guarda una matriz de confusión normalizada en una imagen."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig_size = max(6, len(labels) * 0.5)

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels,
                cmap="GnBu", vmin=0, vmax=1, cbar=True, square=True)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def evaluate_combination(df_subset, feature_cols, group_name, clf_name, clf,
                        n_components, k_best=None, cv_splits=5, verbose=True):
    """Evalúa una combinación con una sola CV y reutiliza sus predicciones OOF.

    Retorna accuracy_mean, accuracy_std, f1_macro_mean, f1_macro_std,
    y_pred_oof, label_encoder y y_true.
    """
    
    le = LabelEncoder()
    y = le.fit_transform(df_subset["label"].astype(str))
    n_classes = len(le.classes_)
    
    X = df_subset[feature_cols].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if X.shape[0] < 2 or n_classes < 2:
        return None, None, None, None, None, None, None
    
    class_counts = pd.Series(y).value_counts()
    n_splits = max(2, min(cv_splits, class_counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    pipe = build_pipeline_hierarchical(clf, n_components, X.shape[1], k_best)
    
    try:
        y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)

        fold_acc = []
        fold_f1 = []
        for _, test_idx in skf.split(X, y):
            fold_acc.append(accuracy_score(y[test_idx], y_pred_oof[test_idx]))
            fold_f1.append(f1_score(y[test_idx], y_pred_oof[test_idx], average="macro"))
        acc = np.asarray(fold_acc)
        f1 = np.asarray(fold_f1)
        
        if verbose:
            print(f"  {group_name:12s} × {clf_name:22s} | "
                  f"acc={acc.mean():.3f}±{acc.std():.3f} | f1={f1.mean():.3f}±{f1.std():.3f}")
        
        return float(acc.mean()), float(acc.std()), float(f1.mean()), float(f1.std()), y_pred_oof, le, y
    except Exception as e:
        # Siempre imprime el error real para facilitar diagnóstico
        print(f"  [ERROR] {group_name} × {clf_name}: {type(e).__name__}: {e}")
        return None, None, None, None, None, None, None


# ===========================================================================================
# PASO 4: EVALUACIÓN EXHAUSTIVA (todos los classificadores × feature_sets × grupos)
# ===========================================================================================
def evaluate_all_hierarchical(df, verbose=True):
    """
    Evalúa TODAS las combinaciones:
    - 6 clasificadores
    - 7 feature_sets (Estadísticas, Frecuencias_Abs/Rel/Est, Wavelets, Todas)
    - color global y etapas super/fina entrenadas por separado para cada color
    
    Retorna: DataFrame con resultados, dict con mejores modelos por grupo
    """
    
    classifiers = build_classifiers(40)  # 40 clases totales (27 letras + 10 números + 3 controles)
    results = []
    best_per_group = {}
    
    # ============ COLOR (Green/Blue - 2 clases) ============
    if verbose:
        print("\n" + "="*100)
        print("EVALUANDO: COLOR (Green/Blue - 2 clases)")
        print("="*100)

    validate_color_labels(df)
    df_color = df
    feature_cols_all = [c for c in df.columns if c not in
                       ["usuario", "tpComando", "letra", "trial", "label", "color"]]
    df_for_color = df_color.copy()
    df_for_color["label"] = df_for_color["color"]
    df_for_color[feature_cols_all] = log_transform_power_columns(
        df_for_color[feature_cols_all], feature_cols_all
    )
    feature_sets = get_feature_sets(feature_cols_all)

    best_f1_color = -1
    for set_name, cols in feature_sets.items():
        if verbose:
            print(f"\n{set_name} ({len(cols)} features):")
        for clf_name, clf in classifiers.items():
            acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true = evaluate_combination(
                df_for_color, cols, "color", clf_name, clf,
                n_components=1, k_best=250, cv_splits=5, verbose=verbose
            )
            if acc_m is not None:
                results.append({
                    "nivel": "color",
                    "grupo": "Green/Blue",
                    "n_clases": 2,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "clasificador": clf_name,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "f1_macro_mean": f1_m,
                    "f1_macro_std": f1_s,
                })
                if f1_m > best_f1_color:
                    best_f1_color = f1_m
                    best_per_group["color"] = {
                        "feature_set": set_name,
                        "cols": cols,
                        "clasificador": clf_name,
                        "clf": clf,
                        "n_components": 1,
                        "f1_macro": f1_m,
                        "accuracy_mean": acc_m,
                        "accuracy_std": acc_s,
                        "label_encoder": le,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }

    if "color" not in best_per_group:
        raise RuntimeError(
            "No se pudo evaluar el nivel de color; verifica que existan muestras "
            "suficientes para Green y Blue."
        )

    # Cada color tiene un clasificador de super-grupo y modelos finos propios.
    for color in ("Green", "Blue"):
        df_color_group = df_color[df_color["color"] == color].reset_index(drop=True)
        if df_color_group.empty:
            raise ValueError(f"No hay muestras disponibles para el color {color}.")

        # ============ SUPER-GRUPO DENTRO DEL COLOR ============
        if verbose:
            print("\n" + "="*100)
            print(f"EVALUANDO SUPER-GRUPO SOLO PARA {color}")
            print("="*100)

        df_super = df_color_group.copy()
        df_super["label"] = df_super["tpComando"]
        df_super[feature_cols_all] = log_transform_power_columns(
            df_super[feature_cols_all], feature_cols_all
        )
        feature_sets = get_feature_sets(feature_cols_all)
        best_key = f"super_clase_{color}"
        best_f1 = -1

        for set_name, cols in feature_sets.items():
            for clf_name, clf in classifiers.items():
                acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true = evaluate_combination(
                    df_super, cols, f"super-{color}", clf_name, clf,
                    n_components=2, k_best=250, cv_splits=5, verbose=verbose
                )
                if acc_m is None:
                    continue
                results.append({
                    "nivel": "super_clase_por_color",
                    "color": color,
                    "grupo": "Letters/Numbers/Controls",
                    "n_clases": len(le.classes_),
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "clasificador": clf_name,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "f1_macro_mean": f1_m,
                    "f1_macro_std": f1_s,
                })
                if f1_m > best_f1:
                    best_f1 = f1_m
                    best_per_group[best_key] = {
                        "feature_set": set_name,
                        "cols": cols,
                        "clasificador": clf_name,
                        "clf": clf,
                        "n_components": 2,
                        "f1_macro": f1_m,
                        "accuracy_mean": acc_m,
                        "accuracy_std": acc_s,
                        "label_encoder": le,
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }

        if best_key not in best_per_group:
            raise RuntimeError(
                f"No se pudo evaluar el clasificador de super-grupo para {color}; "
                "verifica que haya suficientes muestras por grupo."
            )

        # ============ COMANDO FINO DENTRO DE COLOR + SUPER-GRUPO ============
        for grupo in ("Letters", "Numbers", "Controls"):
            df_fine = df_color_group[
                df_color_group["tpComando"] == grupo
            ].reset_index(drop=True)
            if df_fine.empty:
                raise ValueError(f"No hay muestras para la combinación {color}/{grupo}.")

            fine_key = f"{color}_{grupo}"
            fine_classes = sorted(df_fine["label"].astype(str).unique())
            if len(fine_classes) == 1:
                fixed_label = fine_classes[0]
                print(
                    f"  {color}/{grupo}: solo existe la clase {ascii(fixed_label)}; "
                    "se usará una salida fija, sin evaluación multiclase."
                )
                best_per_group[fine_key] = {
                    "fixed_label": fixed_label,
                    "feature_set": "N/A",
                    "cols": [],
                    "clasificador": "Salida fija",
                    "f1_macro": None,
                    "y_true": None,
                    "y_pred": None,
                }
                continue

            if verbose:
                print("\n" + "="*100)
                print(f"EVALUANDO COMANDOS FINOS: {color}/{grupo}")
                print("="*100)

            df_fine[feature_cols_all] = log_transform_power_columns(
                df_fine[feature_cols_all], feature_cols_all
            )
            feature_sets = get_feature_sets(feature_cols_all)
            n_components = max(1, min(len(fine_classes) - 1, 50))
            best_f1 = -1

            for set_name, cols in feature_sets.items():
                for clf_name, clf in classifiers.items():
                    acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true = evaluate_combination(
                        df_fine, cols, f"{color}-{grupo}", clf_name, clf,
                        n_components=n_components, k_best=250,
                        cv_splits=5, verbose=verbose
                    )
                    if acc_m is None:
                        continue
                    results.append({
                        "nivel": "comando_fino_por_color_grupo",
                        "color": color,
                        "grupo": grupo,
                        "n_clases": len(le.classes_),
                        "feature_set": set_name,
                        "n_features": len(cols),
                        "clasificador": clf_name,
                        "accuracy_mean": acc_m,
                        "accuracy_std": acc_s,
                        "f1_macro_mean": f1_m,
                        "f1_macro_std": f1_s,
                    })
                    if f1_m > best_f1:
                        best_f1 = f1_m
                        best_per_group[fine_key] = {
                            "feature_set": set_name,
                            "cols": cols,
                            "clasificador": clf_name,
                            "clf": clf,
                            "n_components": n_components,
                            "f1_macro": f1_m,
                            "accuracy_mean": acc_m,
                            "accuracy_std": acc_s,
                            "label_encoder": le,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }

            if fine_key not in best_per_group:
                raise RuntimeError(
                    f"No se pudo evaluar el modelo fino para {color}/{grupo}; "
                    "verifica que cada comando tenga suficientes muestras."
                )
    
    if not results:
        raise RuntimeError(
            "Ninguna combinación produjo resultados válidos. "
            "Revisa los errores impresos arriba para ver qué falló en cada combinación."
        )
    results_df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    return results_df, best_per_group


# ===========================================================================================
# PASO 5: ENTRENAR MODELOS GANADORES
# ===========================================================================================
def train_final_models(df, best_per_group):
    """Entrena los modelos de color y los modelos super/fino separados por color."""

    validate_color_labels(df)
    df_color = df
    df_color_stage = df_color.copy()
    df_color_stage["label"] = df_color_stage["color"]

    def fit_bundle(df_subset, model_info):
        if "fixed_label" in model_info:
            return {
                "fixed_label": model_info["fixed_label"],
                "feature_columns": [],
                "feature_set": "N/A",
                "clasificador": "Salida fija",
                "f1_macro_cv": None,
            }

        le = LabelEncoder()
        y = le.fit_transform(df_subset["label"].astype(str))
        X = df_subset[model_info["cols"]].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pipeline = build_pipeline_hierarchical(
            model_info["clf"], model_info["n_components"], X.shape[1], k_best=250
        )
        pipeline.fit(X, y)
        return {
            "pipeline": pipeline,
            "label_encoder": le,
            "feature_columns": model_info["cols"],
            "feature_set": model_info["feature_set"],
            "clasificador": model_info["clasificador"],
            "f1_macro_cv": (
                float(model_info["f1_macro"])
                if model_info.get("f1_macro") is not None else None
            ),
        }

    print("\n" + "="*100)
    print("ENTRENANDO MODELO FINAL: COLOR")
    print("="*100)
    final_bundles = {
        "color": fit_bundle(df_color_stage, best_per_group["color"]),
        "por_color": {},
    }

    for color in ("Green", "Blue"):
        print(f"\nENTRENANDO MODELOS SOLO PARA COLOR: {color}")
        df_color_subset = df_color[df_color["color"] == color].reset_index(drop=True)

        df_super = df_color_subset.copy()
        df_super["label"] = df_super["tpComando"]
        hierarchy = {
            "super_clase": fit_bundle(
                df_super, best_per_group[f"super_clase_{color}"]
            ),
            "por_grupo": {},
        }

        for grupo in ("Letters", "Numbers", "Controls"):
            df_fine = df_color_subset[
                df_color_subset["tpComando"] == grupo
            ].reset_index(drop=True)
            fine_bundle = fit_bundle(
                df_fine, best_per_group[f"{color}_{grupo}"]
            )
            hierarchy["por_grupo"][grupo] = fine_bundle
            if "fixed_label" in fine_bundle:
                print(f"  OK {grupo}: salida fija {ascii(fine_bundle['fixed_label'])}")
            else:
                print(
                    f"  OK {grupo}: {len(fine_bundle['label_encoder'].classes_)} "
                    f"comandos, f1_cv={fine_bundle['f1_macro_cv']:.3f}"
                )
        final_bundles["por_color"][color] = hierarchy

    return final_bundles


# ===========================================================================================
# PASO 6: MAIN
# ===========================================================================================
def save_confusion_matrices_per_user(usuario, best_per_group, figures_dir):
    """Guarda una matriz de confusión por cada nivel ganador del usuario."""
    os.makedirs(figures_dir, exist_ok=True)
    for nivel, info in best_per_group.items():
        if info.get("y_true") is None or info.get("y_pred") is None:
            continue
        labels = [str(lbl) for lbl in info["label_encoder"].classes_]
        out_path = os.path.join(figures_dir, f"cm_{nivel}.png")
        title = (
            f"{usuario} | {nivel}\n"
            f"acc={info.get('accuracy_mean', 0.0):.3f}, "
            f"f1_macro={info.get('f1_macro', 0.0):.3f}"
        )
        plot_confusion_matrix(info["y_true"], info["y_pred"], labels, title, out_path)
        print(f"   ✓ Matriz guardada en: {out_path}")


def main():
    print("\n" + "="*100)
    print("ENTRENAMIENTO JERÁRQUICO COMPLETO CON OPTIMIZACIÓN DE 2 NIVELES DE LDA")
    print("POR USUARIO")
    print("="*100)
    
    # Cargar y preparar datos
    print("\n1) Cargando dataset...")
    df = pd.read_csv(config.FEATURES_CSV)
    df = normalize_string_columns(
        df, ["usuario", "tpComando", "letra", "label", "color"]
    )
    validate_color_labels(df)
    meta_cols = ["usuario", "tpComando", "letra", "trial", "label", "color"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = log_transform_power_columns(df[feature_cols], feature_cols)
    print(f"   ✓ Dataset cargado: {df.shape[0]} trials, {len(feature_cols)} features")
    
    usuarios = sorted(df["usuario"].unique())
    print(f"   ✓ Usuarios detectados: {usuarios}\n")
    
    resumen_global = []
    
    # Entrenar un modelo POR USUARIO
    for usuario in usuarios:
        print("\n" + "="*100)
        print(f"USUARIO: {usuario}")
        print("="*100)
        
        df_usuario = df[df["usuario"] == usuario].reset_index(drop=True)
        print(f"Trials para {usuario}: {len(df_usuario)}")
        
        # Paso 1-4: Evaluación exhaustiva (solo para este usuario)
        print(f"\n2) Evaluando TODAS las combinaciones para {usuario}...")
        print("   Esto puede tomar 5-15 minutos por usuario...")
        ini=datetime.now()  
        print("Incio de evaluacion exhaustiva",ini)
        results_df, best_per_group = evaluate_all_hierarchical(df_usuario, verbose=False)
        fin=datetime.now()
        print("Fin de evaluacion exhaustiva",fin)
        print("Tiempo de evaluacion exhaustiva",fin-ini)
        # Guardar tabla por usuario
        usuario_dir = os.path.join(config.OUTPUT_DIR, usuario)
        os.makedirs(usuario_dir, exist_ok=True)
        
        out_csv = os.path.join(usuario_dir, "evaluacion_jerarquica_completa.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"   ✓ Tabla guardada en: {out_csv}")

        # Guardar matrices de confusión del mejor modelo por nivel
        usuario_figures_dir = os.path.join(config.FIGURES_DIR, usuario)
        save_confusion_matrices_per_user(usuario, best_per_group, usuario_figures_dir)
        
        # Resumen ganadores por grupo
        print(f"\n   GANADOR POR CADA GRUPO ({usuario}):")
        print("   " + "-" * 96)
        for grupo, info in best_per_group.items():
            if info.get("fixed_label") is not None:
                print(f"   {grupo:20s} | salida fija={ascii(info['fixed_label'])}")
                continue
            print(f"   {grupo:20s} | feature_set={info['feature_set']:18s} | "
                  f"clf={info['clasificador']:22s} | f1_cv={info['f1_macro']:.3f}")
        
        # Paso 5: Entrenar modelos finales para este usuario
        print(f"\n3) Entrenando modelos finales por color para {usuario}...")
        final_bundles = train_final_models(df_usuario, best_per_group)
        
        # Guardar bundle final POR USUARIO
        usuario_models_dir = os.path.join(config.MODELS_DIR, usuario)
        os.makedirs(usuario_models_dir, exist_ok=True)
        
        hierarchy_bundle = {
            "usuario": usuario,
            "color": final_bundles["color"],
            "por_color": final_bundles["por_color"],
            "channel_names": config.CHANNEL_NAMES,
            "fs": config.FS,
            "use_p300_window_only": config.USE_P300_WINDOW_ONLY,
            "p300_window_s": config.P300_WINDOW_S,
            "apply_baseline_correction": config.APPLY_BASELINE_CORRECTION,
            "baseline_window_s": config.BASELINE_WINDOW_S,
            "log_transform_power": True,
            "use_flash_segmentation": True,
            "n_flashes": 2,  # 2 flashes por grupo en protocolo real
            # Umbral mínimo de confianza para la Etapa 1 (super-clase).
            # 0.0 = nunca rechazar (default). Configurable en inferencia con set_min_group_confidence().
            "min_group_confidence": 0.0,
        }
        
        out_bundle = os.path.join(usuario_models_dir, "hierarchical_bundle_optimizado.joblib")
        joblib.dump(hierarchy_bundle, out_bundle)
        print(f"   ✓ Bundle guardado en: {out_bundle}")
        
        # Resumen para este usuario
        best_row = results_df.iloc[0]
        resumen_global.append({
            "usuario": usuario,
            "n_trials": len(df_usuario),
            "mejor_nivel": best_row["nivel"],
            "mejor_grupo": best_row["grupo"],
            "mejor_feature_set": best_row["feature_set"],
            "mejor_clasificador": best_row["clasificador"],
            "accuracy_mean": float(best_row["accuracy_mean"]),
            "f1_macro_mean": float(best_row["f1_macro_mean"]),
        })
    
    # Resumen final global
    print("\n" + "="*100)
    print("RESUMEN FINAL: MEJOR COMBINACIÓN POR USUARIO")
    print("="*100)
    resumen_df = pd.DataFrame(resumen_global)
    print(resumen_df.to_string(index=False))
    
    # Guardar resumen
    resumen_path = os.path.join(config.OUTPUT_DIR, "resumen_entrenamiento_por_usuario.csv")
    resumen_df.to_csv(resumen_path, index=False)
    print(f"\n✓ Resumen guardado en: {resumen_path}")
    
    print("\n" + "="*100)
    print("ENTRENAMIENTO COMPLETADO PARA TODOS LOS USUARIOS")
    print("="*100)
    print(f"\nEstructura generada:")
    for usuario in usuarios:
        print(f"\n{usuario}:")
        print(f"  - outputs/{usuario}/evaluacion_jerarquica_completa.csv")
        print(f"  - outputs/figures/{usuario}/cm_super_clase.png")
        print(f"  - outputs/figures/{usuario}/cm_Letters.png")
        print(f"  - outputs/figures/{usuario}/cm_Numbers.png")
        print(f"  - outputs/figures/{usuario}/cm_Controls.png")
        print(f"  - outputs/models/{usuario}/hierarchical_bundle_optimizado.joblib")
    
    print(f"\nPara usar en tiempo real (POR USUARIO):")
    print(f"  from hierarchical_infer import HierarchicalBCIPredictor")
    print(f"  predictor = HierarchicalBCIPredictor(usuario='UserCMSM')  # o el usuario que quieras")
    print(f"  comando, detalle = predictor.predict_from_csv('archivo.csv')")


if __name__ == "__main__":
    main()