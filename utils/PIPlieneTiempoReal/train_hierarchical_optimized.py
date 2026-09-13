# -*- coding: utf-8 -*-
"""
ENTRENAMIENTO JERÁRQUICO COMPLETO CON OPTIMIZACIÓN DE 2 NIVELES DE LDA

Flujo:
1. Evalúa TODAS las combinaciones (6 clf × 7 feature_sets × 4 grupos) con CV
2. Guarda tabla completa de resultados (evaluacion_jerarquica_completa.csv)
3. Identifica ganador por cada grupo (super-clase, Letters, Numbers, Controls)
4. Entrena los 4 modelos ganadores con LDA optimizado en 2 niveles:
   - Nivel 1 (Super-clase): LDA optimizado para separar 3 grupos principales
   - Nivel 2 (Por grupo): LDA optimizado dentro de cada grupo especializado
5. Guarda bundle final listo para tiempo real (hierarchical_bundle_optimizado.joblib)

Protocolo asumido:
- 2 flashes por grupo en 2 segundos de grabación
- 14 canales EEG (F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4)
- fs=128 Hz
"""

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
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


# ===========================================================================================
# PASO 1: DEFINIR CLASIFICADORES (6 total)
# ===========================================================================================
def build_classifiers(n_classes):
    """6 clasificadores a evaluar"""
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=1e-3, max_iter=2000, random_state=42)
    svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm_rbf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    rff= RandomForestClassifier(n_estimators=100, random_state=42)
    xgc = XGBClassifier(n_estimators=100, random_state=42)

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
    Pipeline para arquitectura jerárquica con LDA en cascada doble:
    
    Flujo: StandardScaler → [SelectKBest] → LDA_1 (amplio) → LDA_2 (fino) → Clasificador
    
    - LDA_1 (grueso): reduce a min(n_components*3, n_features-1) componentes
      para capturar la mayor varianza discriminante disponible.
    - LDA_2 (fino): refina a n_components componentes en el espacio ya reducido,
      maximizando la separación para el clasificador final.
    - Clasificador: opera sobre las n_components features más discriminantes.
    
    El doble LDA mejora la separabilidad en espacios de alta dimensión (EEG)
    donde un solo LDA puede no capturar toda la estructura de clase.
    
    use_double_lda: Si False, usa solo 1 LDA (modo legado).
    """
    steps = [("scaler", StandardScaler())]

    if k_best is not None and k_best < n_features_in:
        steps.append(("select", SelectKBest(f_classif, k=k_best)))
        effective_features = k_best
    else:
        effective_features = n_features_in

    n_comp_safe = min(n_components, effective_features - 1, 200)
    n_comp_safe = max(1, n_comp_safe)

    if use_double_lda and n_comp_safe >= 2:
        # LDA 1 (amplio): hasta 3× los componentes finales para no perder info
        n_comp_lda1 = min(n_comp_safe * 3, effective_features - 1, 200)
        n_comp_lda1 = max(n_comp_safe + 1, n_comp_lda1)  # siempre > LDA2
        steps.append(("lda1", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=n_comp_lda1)))

        # Re-escalar entre los dos LDA para estabilidad numérica
        steps.append(("scaler2", StandardScaler()))

        # LDA 2 (fino): refina al número de componentes objetivo
        steps.append(("lda2", SafeLDA(
            solver="eigen", shrinkage="auto", n_components=n_comp_safe)))
    else:
        # Modo simple: 1 solo LDA
        steps.append(("lda", SafeLDA(
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
                cmap="Blues", vmin=0, vmax=1, cbar=True, square=True)
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
    """Evalúa 1 combinación: retorna accuracy_mean, f1_macro_mean, y_pred_oof, label_encoder, y_true"""
    
    le = LabelEncoder()
    y = le.fit_transform(df_subset["label"])
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
        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
        
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
    - 4 niveles (super-clase + Letters + Numbers + Controls)
    
    Retorna: DataFrame con resultados, dict con mejores modelos por grupo
    """
    
    classifiers = build_classifiers(40)  # 40 clases totales (27 letras + 10 números + 3 controles)
    results = []
    best_per_group = {}
    
    # ============ SUPER-CLASE (Letters/Numbers/Controls - 3 clases) ============
    if verbose:
        print("\n" + "="*100)
        print("EVALUANDO: SUPER-CLASE (Letters/Numbers/Controls - 3 clases)")
        print("="*100)
    
    df_super = df.copy()
    df_super["label"] = df_super["tpComando"]
    feature_cols_all = [c for c in df.columns if c not in 
                       ["usuario", "tpComando", "letra", "trial", "label"]]
    df_super[feature_cols_all] = log_transform_power_columns(df_super[feature_cols_all], feature_cols_all)
    feature_sets = get_feature_sets(feature_cols_all)
    
    best_f1_super = -1
    for set_name, cols in feature_sets.items():
        if verbose:
            print(f"\n{set_name} ({len(cols)} features):")
        for clf_name, clf in classifiers.items():
            acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true = evaluate_combination(
                df_super, cols, "super", clf_name, clf,
                n_components=2,  # 3 clases -> máx 2 componentes LDA
                k_best=250, cv_splits=5, verbose=verbose
            )
            if acc_m is not None:
                results.append({
                    "nivel": "super_clase",
                    "grupo": "Letters/Numbers/Controls",
                    "n_clases": 3,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "clasificador": clf_name,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "f1_macro_mean": f1_m,
                    "f1_macro_std": f1_s,
                })
                if f1_m > best_f1_super:
                    best_f1_super = f1_m
                    best_per_group["super_clase"] = {
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
    
    # ============ GRUPOS ESPECIALIZADOS (Letters, Numbers, Controls) ============
    for grupo in ["Letters", "Numbers", "Controls"]:
        if verbose:
            print("\n" + "="*100)
            print(f"EVALUANDO: {grupo}")
            print("="*100)
        
        df_grupo = df[df["tpComando"] == grupo].reset_index(drop=True)
        feature_cols_grupo = [c for c in df_grupo.columns if c not in 
                             ["usuario", "tpComando", "letra", "trial", "label"]]
        df_grupo[feature_cols_grupo] = log_transform_power_columns(df_grupo[feature_cols_grupo], feature_cols_grupo)
        feature_sets = get_feature_sets(feature_cols_grupo)
        
        n_clases = df_grupo["label"].nunique()
        n_comp = max(1, min(n_clases - 1, 50))  # LDA components (máx n_clases - 1)
        
        best_f1_grupo = -1
        for set_name, cols in feature_sets.items():
            if verbose:
                print(f"\n{set_name} ({len(cols)} features):")
            for clf_name, clf in classifiers.items():
                acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true = evaluate_combination(
                    df_grupo, cols, grupo, clf_name, clf,
                    n_components=n_comp, k_best=250, cv_splits=5, verbose=verbose
                )
                if acc_m is not None:
                    results.append({
                        "nivel": "por_grupo",
                        "grupo": grupo,
                        "n_clases": n_clases,
                        "feature_set": set_name,
                        "n_features": len(cols),
                        "clasificador": clf_name,
                        "accuracy_mean": acc_m,
                        "accuracy_std": acc_s,
                        "f1_macro_mean": f1_m,
                        "f1_macro_std": f1_s,
                    })
                    if f1_m > best_f1_grupo:
                        best_f1_grupo = f1_m
                        best_per_group[grupo] = {
                            "feature_set": set_name,
                            "cols": cols,
                            "clasificador": clf_name,
                            "clf": clf,
                            "n_components": n_comp,
                            "f1_macro": f1_m,
                            "accuracy_mean": acc_m,
                            "accuracy_std": acc_s,
                            "label_encoder": le,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }
    
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
    """Reentrena los 4 modelos ganadores (super + 3 grupos) sobre TODO el dataset de cada nivel"""
    
    final_bundles = {}
    
    # Super-clase
    print("\n" + "="*100)
    print("ENTRENANDO MODELO FINAL: SUPER-CLASE")
    print("="*100)
    df_super = df.copy()
    df_super["label"] = df_super["tpComando"]
    super_info = best_per_group["super_clase"]
    
    le_super = LabelEncoder()
    y_super = le_super.fit_transform(df_super["label"])
    X_super = df_super[super_info["cols"]].to_numpy()
    X_super = np.nan_to_num(X_super, nan=0.0, posinf=0.0, neginf=0.0)
    
    pipe_super = build_pipeline_hierarchical(
        super_info["clf"], super_info["n_components"], 
        X_super.shape[1], k_best=250
    )
    pipe_super.fit(X_super, y_super)
    
    final_bundles["super_clase"] = {
        "pipeline": pipe_super,
        "label_encoder": le_super,
        "feature_columns": super_info["cols"],
        "feature_set": super_info["feature_set"],
        "clasificador": super_info["clasificador"],
        "f1_macro_cv": float(super_info["f1_macro"]),
    }
    print(f"✓ Super-clase entrenado ({len(le_super.classes_)} clases, f1_cv={super_info['f1_macro']:.3f})")
    
    # Grupos especializados
    for grupo in ["Letters", "Numbers", "Controls"]:
        print(f"\nENTRENANDO MODELO FINAL: {grupo}")
        print("-" * 100)
        
        df_grupo = df[df["tpComando"] == grupo].reset_index(drop=True)
        grupo_info = best_per_group[grupo]
        
        le_grupo = LabelEncoder()
        y_grupo = le_grupo.fit_transform(df_grupo["label"])
        X_grupo = df_grupo[grupo_info["cols"]].to_numpy()
        X_grupo = np.nan_to_num(X_grupo, nan=0.0, posinf=0.0, neginf=0.0)
        
        pipe_grupo = build_pipeline_hierarchical(
            grupo_info["clf"], grupo_info["n_components"],
            X_grupo.shape[1], k_best=250
        )
        pipe_grupo.fit(X_grupo, y_grupo)
        
        final_bundles[grupo] = {
            "pipeline": pipe_grupo,
            "label_encoder": le_grupo,
            "feature_columns": grupo_info["cols"],
            "feature_set": grupo_info["feature_set"],
            "clasificador": grupo_info["clasificador"],
            "f1_macro_cv": float(grupo_info["f1_macro"]),
        }
        print(f"✓ {grupo} entrenado ({len(le_grupo.classes_)} clases, f1_cv={grupo_info['f1_macro']:.3f})")
    
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
    meta_cols = ["usuario", "tpComando", "letra", "trial", "label"]
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
        results_df, best_per_group = evaluate_all_hierarchical(df_usuario, verbose=False)
        
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
            print(f"   {grupo:20s} | feature_set={info['feature_set']:18s} | "
                  f"clf={info['clasificador']:22s} | f1_cv={info['f1_macro']:.3f}")
        
        # Paso 5: Entrenar modelos finales para este usuario
        print(f"\n3) Entrenando 4 modelos finales para {usuario}...")
        final_bundles = train_final_models(df_usuario, best_per_group)
        
        # Guardar bundle final POR USUARIO
        usuario_models_dir = os.path.join(config.MODELS_DIR, usuario)
        os.makedirs(usuario_models_dir, exist_ok=True)
        
        hierarchy_bundle = {
            "usuario": usuario,
            "super_clase": final_bundles["super_clase"],
            "por_grupo": {
                "Letters": final_bundles["Letters"],
                "Numbers": final_bundles["Numbers"],
                "Controls": final_bundles["Controls"],
            },
            "channel_names": config.CHANNEL_NAMES,
            "fs": config.FS,
            "use_p300_window_only": config.USE_P300_WINDOW_ONLY,
            "p300_window_s": config.P300_WINDOW_S,
            "apply_baseline_correction": config.APPLY_BASELINE_CORRECTION,
            "baseline_window_s": config.BASELINE_WINDOW_S,
            "log_transform_power": True,
            "use_flash_segmentation": True,
            "n_flashes": 2,  # 2 flashes por grupo en protocolo real
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