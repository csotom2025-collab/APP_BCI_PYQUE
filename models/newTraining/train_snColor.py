"""
ENTRENAMIENTO JERÁRQUICO DIRECTO (2 ETAPAS: SUPER-GRUPO → COMANDO FINO, SIN COLOR)
CON SELECCIÓN DE REDUCCIÓN DIMENSIONAL: 'LDA', 'PCA' O 'NADA'

Flujo:
1. Evalúa combinaciones de clasificadores y features con validación cruzada (Super-grupo y Fino)
   aplicando StandardScaler → mRMR → [SafeLDA | PCA | NADA] → Clasificador
2. Guarda tabla completa de resultados (evaluacion_jerarquica_completa.csv)
3. Identifica modelos ganadores para super-grupo y comando fino por categoría
4. Entrena la cascada: super-grupo (Letters/Numbers/Controls) → comando fino del grupo
5. Guarda bundle final listo para tiempo real (hierarchical_bundle_optimizado.joblib)
6. Guarda top 5 cascadas completas y top 5 modelos especializados por categoría
7. Guarda modelos top por ranking y sus pasos entrenados en archivos separados

Protocolo asumido:
- 2 flashes por grupo en 2 segundos de grabación
- 14 canales EEG (F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4)
- fs=128 Hz
"""

from datetime import datetime
from collections import Counter
from typing import Callable, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
from models.newTraining.lda_utils import SafeLDA
import models.newTraining.config as config
from models.newTraining.eeg_features import get_feature_sets, log_transform_power_columns


CV_N_JOBS = max(1, (os.cpu_count() or 4) - 1)
META_COLUMNS = ["usuario", "tpComando", "letra", "trial", "label", "color"]
VALID_REDUCTION_METHODS = {"LDA", "PCA", "NADA"}


def normalize_reduction_method(reduction_method):
    """Normaliza y valida el método de reducción ('LDA', 'PCA' o 'NADA')."""
    if reduction_method is None:
        return "NADA"
    method = str(reduction_method).strip().upper()
    if method in ("NONE", "SIN", ""):
        method = "NADA"
    if method not in VALID_REDUCTION_METHODS:
        raise ValueError(
            f"reduction_method inválido: {reduction_method!r}. "
            f"Usa uno de: {sorted(VALID_REDUCTION_METHODS)}."
        )
    return method


def report_progress(progress_callback, message):
    if progress_callback is not None:
        progress_callback(message)


def normalize_string_columns(df, columns):
    """Convierte columnas de metadatos a texto para evitar mezclas int/str en LabelEncoder."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else str(x))
    return df


def _resolve_n_components(
    reduction_method, n_classes, pca_n_components=0.95
):
    """Determina n_components según el método elegido ('LDA', 'PCA' o 'NADA')."""
    method = normalize_reduction_method(reduction_method)
    if method == "LDA":
        return max(1, min(int(n_classes) - 1, 50))
    if method == "PCA":
        return pca_n_components
    return None


def _resolve_pca_components(n_components, n_samples, n_features):
    """Calcula un valor válido de n_components para PCA (entero o varianza flotante)."""
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        return n_components
    max_valid = max(1, min(int(n_samples), int(n_features), 200))
    return max(1, min(int(n_components), max_valid))


# ===========================================================================================
# PASO 1: DEFINIR CLASIFICADORES
# ===========================================================================================
def build_classifiers(n_classes, seed=42):
    """Clasificadores candidatos sobre las características procesadas."""
    randomState = seed

    if n_classes == 2:
        xgb_params = {"objective": "binary:logistic", "random_state": randomState}
    else:
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "random_state": randomState,
            "eval_metric": "mlogloss",
        }

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-2,
        max_iter=2000,
        random_state=randomState,
    )
    svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=randomState)
    svm_rbf = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=randomState)
    logreg = LogisticRegression(C=1.0, max_iter=2000, random_state=randomState)
    rff = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=randomState)
    xgc = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, **xgb_params)

    gnb = GaussianNB()
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    lda_clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

    # Ensambles
    ensemble_mlp_svmrbf = VotingClassifier(
        estimators=[("mlp", mlp), ("svm_rbf", svm_rbf)],
        voting="soft",
    )
    ensemble_svmlin_logreg = VotingClassifier(
        estimators=[("svm_lin", svm_lin), ("logreg", logreg)],
        voting="soft",
    )
    ensemble_lda_gnb_logreg = VotingClassifier(
        estimators=[("lda", lda_clf), ("gnb", gnb), ("logreg", logreg)],
        voting="soft",
    )

    return {
        "MLP": mlp,
        "SVM_Lineal": svm_lin,
        "SVM_RBF": svm_rbf,
        "LogReg": logreg,
        "RF": rff,
        "XGBC": xgc,
        "GaussianNB": gnb,
        "QDA": qda,
        "KNN_7": knn,
        "LDA_Clf": lda_clf,
        "Ensamble_MLP_SVMRBF": ensemble_mlp_svmrbf,
        "Ensamble_SVMLin_LogReg": ensemble_svmlin_logreg,
        "Ensamble_LDA_GNB_LogReg": ensemble_lda_gnb_logreg,
    }


def classifiers_for_group(classifiers, n_classes, reduction_method="NADA"):
    """Adapta hiperparámetros según el nivel jerárquico y la reducción usada."""
    method = normalize_reduction_method(reduction_method)
    adapted = {k: clone(v) for k, v in classifiers.items()}

    if "KNN_7" in adapted:
        if n_classes <= 3:
            k_optimo = 15
        else:
            k_optimo = n_classes if n_classes % 2 != 0 else n_classes + 1
            k_optimo = max(7, k_optimo)

        adapted["KNN_7"].set_params(n_neighbors=k_optimo, weights="distance")

    # Solo si se usa LDA en espacios de muy baja dimensión (<= 3 clases -> <= 2D)
    if method == "LDA" and n_classes <= 3:
        adapted.pop("XGBC", None)
        adapted.pop("RF", None)

    return adapted


# ===========================================================================================
# PASO 2: PIPELINE CONFIGURABLE (StandardScaler → mRMR → [LDA | PCA | NADA] → Clasificador)
# ===========================================================================================
class MRMRSelector(BaseEstimator, TransformerMixin):
    """Selecciona variables con mRMR usando MI clase-variable y redundancia gaussiana."""

    def __init__(self, k, feature_names=None):
        self.k = k
        self.feature_names = feature_names

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError("mRMR requiere una matriz con al menos una variable.")

        n_features = X.shape[1]
        if self.feature_names is not None and len(self.feature_names) != n_features:
            raise ValueError(
                "La cantidad de nombres de variables no coincide con las columnas de X."
            )

        relevance = mutual_info_classif(X, y, random_state=42)
        correlation = np.asarray(np.corrcoef(X, rowvar=False))
        if correlation.ndim == 0:
            correlation = correlation.reshape(1, 1)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.clip(correlation, -1.0 + 1e-12, 1.0 - 1e-12)
        redundancy = -0.5 * np.log1p(-(correlation ** 2))
        np.fill_diagonal(redundancy, 0.0)

        n_selected = min(max(1, int(self.k)), n_features)
        available = np.ones(n_features, dtype=bool)
        selected_indices = []
        selection_scores = []
        redundancy = np.asfortranarray(redundancy)
        redundancy_sum = np.zeros(n_features)
        for _ in range(n_selected):
            scores = relevance.copy()
            if selected_indices:
                scores -= redundancy_sum / len(selected_indices)
            scores[~available] = -np.inf
            selected = int(np.argmax(scores))
            selected_indices.append(selected)
            selection_scores.append(float(scores[selected]))
            available[selected] = False
            redundancy_sum += redundancy[:, selected]

        if self.feature_names is None:
            names = [str(index) for index in range(n_features)]
        else:
            names = list(self.feature_names)

        self.n_features_in_ = n_features
        self.relevance_scores_ = relevance
        self.selected_indices_ = np.asarray(selected_indices, dtype=int)
        self.selected_feature_names_ = [
            names[index] for index in self.selected_indices_
        ]
        self.selection_scores_ = np.asarray(selection_scores)
        return self

    def transform(self, X):
        if not hasattr(self, "selected_indices_"):
            raise ValueError("El selector mRMR debe ajustarse antes de transformar.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("La matriz de inferencia no coincide con las variables de mRMR.")
        return X[:, self.selected_indices_]


def build_pipeline_hierarchical(
    clf,
    n_features_in,
    k_best=None,
    feature_names=None,
    reduction_method="NADA",
    n_components=None,
    n_samples=None,
):
    """
    Pipeline para cada nodo de la arquitectura jerárquica (Super-grupo → Fino).

    Flujo: StandardScaler → mRMR → [SafeLDA | PCA | NADA] → Clasificador
    """
    method = normalize_reduction_method(reduction_method)
    steps = [("scaler", StandardScaler())]

    if k_best is not None:
        k_final = min(max(1, k_best), n_features_in)
    else:
        k_final = max(1, int(n_features_in * 0.8))

    steps.append((
        "mrmr",
        MRMRSelector(k=k_final, feature_names=feature_names),
    ))

    if method == "LDA":
        req_comp = 1 if n_components is None else int(n_components)
        n_comp_safe = max(1, min(req_comp, k_final - 1, 200))
        steps.append((
            "lda",
            SafeLDA(solver="eigen", shrinkage="auto", n_components=n_comp_safe),
        ))
    elif method == "PCA":
        req_comp = 0.95 if n_components is None else n_components
        max_samples = n_samples if n_samples is not None else k_final
        n_comp_safe = _resolve_pca_components(req_comp, max_samples, k_final)
        steps.append((
            "pca",
            PCA(n_components=n_comp_safe, random_state=42),
        ))
    # Si method == "NADA", no se añade paso de proyección

    steps.append(("clf", clone(clf)))

    return Pipeline(steps)


def get_k_best_candidates(n_features_in):
    """Genera cantidades de variables candidatas según el tamaño de entrada."""
    if n_features_in < 1:
        raise ValueError("Se requiere al menos una característica para evaluar k_best.")

    ratios = (
        (0.02, 0.05, 0.10, 0.25, 0.50, 0.75)
        if n_features_in > 300
        else (0.10, 0.25, 0.50, 0.75)
    )
    return sorted({
        max(1, int(np.ceil(n_features_in * ratio)))
        for ratio in ratios
    })


# ===========================================================================================
# PASO 3: EVALUACIÓN DE 1 COMBINACIÓN (clf × feature_set × grupo)
# ===========================================================================================
def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    """Guarda una matriz de confusión normalizada en una imagen."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig_size = max(6, len(labels) * 0.5)

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="GnBu",
        vmin=0,
        vmax=1,
        cbar=True,
        square=True,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


_CHEAP_CLASSIFIERS = (
    GaussianNB,
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    KNeighborsClassifier,
)


def _fast_eval_clone(clf):
    """Clon del clasificador usado solo para la validación cruzada."""
    fast = clone(clf)
    if isinstance(fast, SVC):
        fast.set_params(probability=False)
    elif isinstance(fast, XGBClassifier):
        fast.set_params(n_jobs=1)
    return fast


def _fold_preprocessing(
    X,
    y,
    train_idx,
    test_idx,
    k_list,
    feature_names,
    reduction_method="NADA",
    n_components=None,
):
    """Scaler → mRMR → [LDA | PCA | NADA] de un fold para todos los k candidatos."""
    method = normalize_reduction_method(reduction_method)
    n_features = X.shape[1]
    n_train_samples = len(train_idx)
    k_finals = {k: min(max(1, int(k)), n_features) for k in k_list}
    y_train = y[train_idx]

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X[train_idx])
    Xs_test = scaler.transform(X[test_idx])

    selector = MRMRSelector(
        k=max(k_finals.values()), feature_names=feature_names
    ).fit(Xs_train, y_train)
    order = selector.selected_indices_

    out = {}
    for k, k_final in k_finals.items():
        idx = order[:k_final]
        Xk_train = Xs_train[:, idx]
        Xk_test = Xs_test[:, idx]

        if method == "LDA":
            req_comp = 1 if n_components is None else int(n_components)
            lda = SafeLDA(
                solver="eigen",
                shrinkage="auto",
                n_components=max(1, min(req_comp, k_final - 1, 200)),
            )
            z_fit = lda.fit_transform(Xk_train, y_train)
            out[k] = (z_fit, lda.transform(Xk_train), lda.transform(Xk_test))
        elif method == "PCA":
            req_comp = 0.95 if n_components is None else n_components
            n_comp_safe = _resolve_pca_components(
                req_comp, n_train_samples, k_final
            )
            pca = PCA(n_components=n_comp_safe, random_state=42)
            z_fit = pca.fit_transform(Xk_train)
            out[k] = (z_fit, pca.transform(Xk_train), pca.transform(Xk_test))
        else:
            out[k] = (Xk_train, Xk_train, Xk_test)

    return out


def _fit_predict_fold(clf, y_train, z_fit, z_train_pred, z_test):
    """Ajusta el clasificador final de un fold y devuelve (pred test, pred train)."""
    model = clone(clf)
    model.fit(z_fit, y_train)
    return model.predict(z_test), model.predict(z_train_pred)


def evaluate_combination(
    df_subset,
    feature_cols,
    group_name,
    clf_name,
    clf,
    reduction_method="NADA",
    n_components=None,
    k_best=None,
    cv_splits=5,
    verbose=True,
    cv_cache=None,
):
    """Evalúa una combinación y selecciona k_best por F1 macro de CV agrupada."""
    method = normalize_reduction_method(reduction_method)

    if cv_cache is None:
        cv_cache = {}
    if cv_cache.get("df") is not df_subset or cv_cache.get("cols") is not feature_cols:
        cv_cache.clear()
        cv_cache["df"] = df_subset
        cv_cache["cols"] = feature_cols

    base = cv_cache.get("base")
    if base is None:
        le = LabelEncoder()
        y = le.fit_transform(df_subset["label"].astype(str))
        n_classes = len(le.classes_)

        X = df_subset[feature_cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if X.shape[0] < 2 or n_classes < 2:
            return None, None, None, None, None, None, None, None, None, None

        groups = df_subset.groupby(["usuario", "label"]).cumcount() // 4

        min_groups = (
            pd.DataFrame({"y": y, "g": groups})
            .groupby("y")["g"]
            .nunique()
            .min()
        )
        n_splits = max(2, min(cv_splits, int(min_groups)))
        base = {
            "X": X,
            "y": y,
            "le": le,
            "n_classes": n_classes,
            "groups": groups,
            "n_splits": n_splits,
        }
        cv_cache["base"] = base

    X, y, le = base["X"], base["y"], base["le"]
    n_classes = base["n_classes"]

    try:
        if "fold_splits" not in base:
            sgkf = StratifiedGroupKFold(n_splits=base["n_splits"])
            base["fold_splits"] = list(sgkf.split(X, y, groups=base["groups"]))

        fold_splits = base["fold_splits"]
        k_candidates = (
            [k_best]
            if k_best is not None
            else get_k_best_candidates(X.shape[1])
        )
        fold_sample_counts = [
            f"F{i}={len(train_idx)}/{len(test_idx)}"
            for i, (train_idx, test_idx) in enumerate(fold_splits, 1)
        ]

        pre_key = (method, n_components, tuple(k_candidates))
        pre_store = cv_cache.setdefault("pre", {})
        pre = pre_store.get(pre_key)
        if pre is None:
            pre = joblib.Parallel(n_jobs=min(CV_N_JOBS, len(fold_splits)))(
                joblib.delayed(_fold_preprocessing)(
                    X,
                    y,
                    train_idx,
                    test_idx,
                    k_candidates,
                    feature_cols,
                    reduction_method=method,
                    n_components=n_components,
                )
                for train_idx, test_idx in fold_splits
            )
            pre_store[pre_key] = pre

        fast_clf = _fast_eval_clone(clf)
        task_keys = [
            (candidate_k, fold)
            for candidate_k in k_candidates
            for fold in range(len(fold_splits))
        ]
        args = [
            (fast_clf, y[fold_splits[fold][0]], *pre[fold][candidate_k])
            for candidate_k, fold in task_keys
        ]
        if isinstance(clf, _CHEAP_CLASSIFIERS):
            task_results = [_fit_predict_fold(*a) for a in args]
        else:
            task_results = joblib.Parallel(
                n_jobs=min(CV_N_JOBS, len(args))
            )(joblib.delayed(_fit_predict_fold)(*a) for a in args)
        preds = dict(zip(task_keys, task_results))

        best_result = None
        for candidate_k in k_candidates:
            y_pred_oof = np.empty_like(y)
            train_prediction_votes = np.zeros(
                (len(y), n_classes), dtype=np.int32
            )
            fold_acc = []
            fold_f1 = []
            for fold, (train_idx, test_idx) in enumerate(fold_splits):
                y_pred_test, y_pred_train = preds[(candidate_k, fold)]
                y_pred_oof[test_idx] = y_pred_test
                np.add.at(
                    train_prediction_votes,
                    (train_idx, y_pred_train),
                    1,
                )
                fold_acc.append(accuracy_score(y[test_idx], y_pred_test))
                fold_f1.append(f1_score(y[test_idx], y_pred_test, average="macro"))

            acc = np.asarray(fold_acc)
            f1 = np.asarray(fold_f1)
            y_pred_train = train_prediction_votes.argmax(axis=1)
            kappa = cohen_kappa_score(y, y_pred_oof)
            candidate_result = (
                float(acc.mean()),
                float(acc.std()),
                float(f1.mean()),
                float(f1.std()),
                y_pred_oof,
                le,
                y,
                y_pred_train,
                candidate_k,
                float(kappa),
            )
            if best_result is None or candidate_result[2] > best_result[2]:
                best_result = candidate_result

        if verbose:
            print(
                f"  {group_name:12s} × {clf_name:22s} [{method:4s}] | "
                f"muestras train/test por fold: "
                f"{', '.join(fold_sample_counts)} | "
                f"k_best={best_result[8]} | "
                f"acc={best_result[0]:.3f}±{best_result[1]:.3f} | "
                f"f1={best_result[2]:.3f}±{best_result[3]:.3f} | "
                f"kappa={best_result[9]:.3f}"
            )

        return best_result
    except Exception as e:
        print(f"  [ERROR] {group_name} × {clf_name}: {type(e).__name__}: {e}")
        return None, None, None, None, None, None, None, None, None, None


# ===========================================================================================
# PASO 4: EVALUACIÓN EXHAUSTIVA (Super-grupo → Comando fino)
# ===========================================================================================
def evaluate_all_hierarchical(
    df,
    verbose=True,
    seed=42,
    reduction_method="NADA",
    pca_n_components=0.95,
    progress_callback=None,
):
    """Evalúa todas las combinaciones en Super-grupo y Comando fino."""
    method = normalize_reduction_method(reduction_method)
    classifiers = build_classifiers(40, seed=seed)
    results = []
    best_per_group = {}

    feature_cols_all = [c for c in df.columns if c not in META_COLUMNS]

    # ============ ETAPA 1: SUPER-GRUPO (Letters / Numbers / Controls) ============
    if verbose:
        print("\n" + "=" * 100)
        print(f"EVALUANDO SUPER-GRUPO (Letters / Numbers / Controls) | REDUCCIÓN: {method}")
        print("=" * 100)

    df_super = df.copy()
    df_super["label"] = df_super["tpComando"]
    n_super_classes = df_super["label"].nunique()
    super_n_components = _resolve_n_components(
        method, n_super_classes, pca_n_components=pca_n_components
    )
    super_classifiers = classifiers_for_group(
        classifiers, n_super_classes, reduction_method=method
    )
    df_super[feature_cols_all] = log_transform_power_columns(
        df_super[feature_cols_all], feature_cols_all
    )
    feature_sets = get_feature_sets(feature_cols_all)
    best_f1_super = -1

    for set_name, cols in feature_sets.items():
        if verbose:
            print(f"\n{set_name} ({len(cols)} features):")
        cv_cache = {}
        for clf_name, clf in super_classifiers.items():
            report_progress(
                progress_callback,
                f"Evaluando supergrupo ({method}): {set_name} / {clf_name}...",
            )
            (
                acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true,
                y_train_pred, selected_k_best, selected_kappa,
            ) = evaluate_combination(
                df_super,
                cols,
                "super_clase",
                clf_name,
                clf,
                reduction_method=method,
                n_components=super_n_components,
                cv_splits=5,
                verbose=verbose,
                cv_cache=cv_cache,
            )
            if acc_m is None:
                continue
            results.append({
                "nivel": "super_clase",
                "grupo": "Letters/Numbers/Controls",
                "n_clases": len(le.classes_),
                "reduccion": method,
                "n_components": super_n_components,
                "feature_set": set_name,
                "n_features": len(cols),
                "k_best": selected_k_best,
                "mrmr_ratio": selected_k_best / len(cols),
                "cohen_kappa": selected_kappa,
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
                    "reduction_method": method,
                    "n_components": super_n_components,
                    "k_best": selected_k_best,
                    "cohen_kappa": selected_kappa,
                    "f1_macro": f1_m,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "label_encoder": le,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_train_pred": y_train_pred,
                }

    if "super_clase" not in best_per_group:
        raise RuntimeError(
            "No se pudo evaluar el nivel de super_clase; verifica que existan "
            "muestras suficientes para cada grupo."
        )

    # ============ ETAPA 2: COMANDO FINO DENTRO DE CADA SUPER-GRUPO ============
    for grupo in ("Letters", "Numbers", "Controls"):
        df_fine = df[df["tpComando"] == grupo].reset_index(drop=True)
        if df_fine.empty:
            raise ValueError(f"No hay muestras para el grupo {grupo}.")

        fine_classes = sorted(df_fine["label"].astype(str).unique())
        if len(fine_classes) == 1:
            fixed_label = fine_classes[0]
            print(
                f"  {grupo}: solo existe la clase {ascii(fixed_label)}; "
                "se usará una salida fija, sin evaluación multiclase."
            )
            best_per_group[grupo] = {
                "fixed_label": fixed_label,
                "feature_set": "N/A",
                "cols": [],
                "clasificador": "Salida fija",
                "reduction_method": method,
                "f1_macro": None,
                "y_true": None,
                "y_pred": None,
            }
            continue

        if verbose:
            print("\n" + "=" * 100)
            print(f"EVALUANDO COMANDOS FINOS: {grupo} | REDUCCIÓN: {method}")
            print("=" * 100)

        df_fine[feature_cols_all] = log_transform_power_columns(
            df_fine[feature_cols_all], feature_cols_all
        )
        feature_sets = get_feature_sets(feature_cols_all)
        fine_n_components = _resolve_n_components(
            method, len(fine_classes), pca_n_components=pca_n_components
        )
        fine_classifiers = classifiers_for_group(
            classifiers, len(fine_classes), reduction_method=method
        )
        best_f1 = -1

        for set_name, cols in feature_sets.items():
            if verbose:
                print(f"\n{set_name} ({len(cols)} features):")
            cv_cache = {}
            for clf_name, clf in fine_classifiers.items():
                report_progress(
                    progress_callback,
                    f"Evaluando {grupo} ({method}): {set_name} / {clf_name}...",
                )
                (
                    acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true,
                    y_train_pred, selected_k_best, selected_kappa,
                ) = evaluate_combination(
                    df_fine,
                    cols,
                    grupo,
                    clf_name,
                    clf,
                    reduction_method=method,
                    n_components=fine_n_components,
                    cv_splits=5,
                    verbose=verbose,
                    cv_cache=cv_cache,
                )
                if acc_m is None:
                    continue
                results.append({
                    "nivel": "comando_fino",
                    "grupo": grupo,
                    "n_clases": len(le.classes_),
                    "reduccion": method,
                    "n_components": fine_n_components,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "k_best": selected_k_best,
                    "mrmr_ratio": selected_k_best / len(cols),
                    "cohen_kappa": selected_kappa,
                    "clasificador": clf_name,
                    "accuracy_mean": acc_m,
                    "accuracy_std": acc_s,
                    "f1_macro_mean": f1_m,
                    "f1_macro_std": f1_s,
                })
                if f1_m > best_f1:
                    best_f1 = f1_m
                    best_per_group[grupo] = {
                        "feature_set": set_name,
                        "cols": cols,
                        "clasificador": clf_name,
                        "clf": clf,
                        "reduction_method": method,
                        "n_components": fine_n_components,
                        "k_best": selected_k_best,
                        "cohen_kappa": selected_kappa,
                        "f1_macro": f1_m,
                        "accuracy_mean": acc_m,
                        "accuracy_std": acc_s,
                        "label_encoder": le,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "y_train_pred": y_train_pred,
                    }

        if grupo not in best_per_group:
            raise RuntimeError(
                f"No se pudo evaluar el modelo fino para {grupo}; "
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
# PASO 5: ENTRENAR MODELOS GANADORES Y TOP 5 POR GRUPO
# ===========================================================================================
def train_final_models(df, best_per_group, progress_callback=None):
    """Entrena los modelos finales de super-grupo y comando fino por grupo."""

    def fit_bundle(df_subset, model_info, model_name):
        if "fixed_label" in model_info:
            return {
                "fixed_label": model_info["fixed_label"],
                "feature_columns": [],
                "feature_set": "N/A",
                "clasificador": "Salida fija",
                "f1_macro_cv": None,
            }

        method = normalize_reduction_method(
            model_info.get("reduction_method", "NADA")
        )
        le = LabelEncoder()
        y = le.fit_transform(df_subset["label"].astype(str))
        X = df_subset[model_info["cols"]].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        report_progress(progress_callback, f"Ajustando modelo final: {model_name}...")
        pipeline = build_pipeline_hierarchical(
            model_info["clf"],
            X.shape[1],
            k_best=model_info["k_best"],
            feature_names=model_info["cols"],
            reduction_method=method,
            n_components=model_info.get("n_components"),
            n_samples=X.shape[0],
        )
        pipeline.fit(X, y)
        selected_features = list(
            pipeline.named_steps["mrmr"].selected_feature_names_
        )
        bundle = {
            "pipeline": pipeline,
            "label_encoder": le,
            "feature_columns": model_info["cols"],
            "selected_feature_columns": selected_features,
            "feature_set": model_info["feature_set"],
            "clasificador": model_info["clasificador"],
            "reduction_method": method,
            "k_best": model_info["k_best"],
            "mrmr_ratio": len(selected_features) / len(model_info["cols"]),
            "cohen_kappa": model_info["cohen_kappa"],
            "accuracy_mean": float(model_info["accuracy_mean"]),
            "f1_macro_cv": (
                float(model_info["f1_macro"])
                if model_info.get("f1_macro") is not None
                else None
            ),
        }
        if "pca" in pipeline.named_steps:
            bundle["pca_components"] = int(
                pipeline.named_steps["pca"].n_components_
            )
        elif "lda" in pipeline.named_steps:
            bundle["lda_components"] = int(
                pipeline.named_steps["lda"].n_components
            )
        return bundle

    print("\n" + "=" * 100)
    print("ENTRENANDO MODELO FINAL: SUPER-GRUPO")
    print("=" * 100)
    df_super = df.copy()
    df_super["label"] = df_super["tpComando"]

    final_bundles = {
        "super_clase": fit_bundle(
            df_super, best_per_group["super_clase"], "supergrupo"
        ),
        "por_grupo": {},
    }

    print("\nENTRENANDO MODELOS FINOS POR GRUPO:")
    for grupo in ("Letters", "Numbers", "Controls"):
        df_fine = df[df["tpComando"] == grupo].reset_index(drop=True)
        fine_bundle = fit_bundle(df_fine, best_per_group[grupo], grupo)
        final_bundles["por_grupo"][grupo] = fine_bundle
        if "fixed_label" in fine_bundle:
            print(f"  OK {grupo}: salida fija {ascii(fine_bundle['fixed_label'])}")
        else:
            print(
                f"  OK {grupo}: {len(fine_bundle['label_encoder'].classes_)} "
                f"comandos, f1_cv={fine_bundle['f1_macro_cv']:.3f}"
            )

    return final_bundles


def write_mrmr_report(usuario, final_bundles, report_path):
    """Escribe las variables elegidas por nodo y un resumen de canales."""
    model_records = [("Supergrupo", final_bundles["super_clase"])]
    model_records.extend(
        (group, model) for group, model in final_bundles["por_grupo"].items()
    )

    channel_counts = Counter()
    channel_models = {}
    channel_relevance = {}
    lines = [
        f"Reporte mRMR - Usuario: {usuario}",
        "=" * 72,
        "Los nombres listados son las variables de entrada que el pipeline",
        "selecciona con mRMR. La inferencia reordena las columnas con",
        "'feature_columns' y aplica el selector mRMR guardado en cada pipeline.",
        "",
        "Modelos finales por nodo",
        "-" * 72,
    ]

    for model_name, model in model_records:
        lines.append(f"\n{model_name}")
        if "fixed_label" in model:
            lines.append(f"  Salida fija: {model['fixed_label']}")
            continue

        selector = model["pipeline"].named_steps["mrmr"]
        selected_features = model.get(
            "selected_feature_columns",
            selector.selected_feature_names_,
        )
        input_features = model["feature_columns"]
        ratio = len(selected_features) / len(input_features)
        method = model.get("reduction_method", "NADA")
        lines.extend([
            f"  Clasificador: {model['clasificador']} (Reducción: {method})",
            f"  Conjunto de entrada: {model['feature_set']} "
            f"({len(input_features)} variables)",
            f"  Variables mRMR: {len(selected_features)} "
            f"({ratio:.1%} de entrada)",
            f"  Métricas CV: F1 macro={model['f1_macro_cv']:.4f}, "
            f"accuracy={model['accuracy_mean']:.4f}, "
            f"kappa={model['cohen_kappa']:.4f}",
            "  Variables seleccionadas (orden mRMR):",
        ])
        relevance_by_feature = dict(zip(
            input_features, selector.relevance_scores_
        ))
        for feature in selected_features:
            channel = str(feature).split("_", 1)[0]
            relevance = float(relevance_by_feature[feature])
            lines.append(
                f"    - {feature} (relevancia MI={relevance:.5f})"
            )
            channel_counts[channel] += 1
            channel_models.setdefault(channel, set()).add(model_name)
            channel_relevance.setdefault(channel, []).append(relevance)

    lines.extend([
        "",
        "Canales con mayor presencia en la selección mRMR",
        "-" * 72,
        "El canal se identifica por el prefijo de cada variable hasta '_'.",
        "La frecuencia cuenta variables seleccionadas en todos los nodos;",
        "la relevancia MI es el promedio de esas variables seleccionadas.",
    ])
    for channel, count in sorted(
        channel_counts.items(),
        key=lambda item: (
            -item[1],
            -float(np.mean(channel_relevance[item[0]])),
            item[0],
        ),
    ):
        mean_relevance = float(np.mean(channel_relevance[channel]))
        lines.append(
            f"  {channel}: {count} variables seleccionadas, "
            f"{len(channel_models[channel])} nodos, "
            f"relevancia MI media={mean_relevance:.5f}"
        )

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def train_top_models_by_category(
    df,
    results_df,
    seed=42,
    reduction_method="NADA",
    pca_n_components=0.95,
    progress_callback=None,
):
    """Entrena los candidatos solicitados para cada ranking y categoría."""
    default_method = normalize_reduction_method(reduction_method)
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    classifiers = build_classifiers(40, seed=seed)
    prepared_groups = {}
    trained_models = {}
    evaluated_models = {}

    ranked_results = results_df.sort_values(
        "f1_macro_mean", ascending=False, kind="stable"
    )
    category_specs = {
        "general": (ranked_results, 5),
        "super": (
            ranked_results[ranked_results["nivel"] == "super_clase"],
            10,
        ),
        "letras": (
            ranked_results[
                (ranked_results["nivel"] == "comando_fino")
                & (ranked_results["grupo"] == "Letters")
            ],
            10,
        ),
        "numeros": (
            ranked_results[
                (ranked_results["nivel"] == "comando_fino")
                & (ranked_results["grupo"] == "Numbers")
            ],
            10,
        ),
        "controles": (
            ranked_results[
                (ranked_results["nivel"] == "comando_fino")
                & (ranked_results["grupo"] == "Controls")
            ],
            5,
        ),
    }
    cascade_specs = {
        "_cascada_super": ranked_results[
            ranked_results["nivel"] == "super_clase"
        ],
    }
    for group in ("Letters", "Numbers", "Controls"):
        cascade_specs[f"_cascada_fino_{group}"] = ranked_results[
            (ranked_results["nivel"] == "comando_fino")
            & (ranked_results["grupo"] == group)
        ]

    category_specs.update(
        {key: (value, 5) for key, value in cascade_specs.items()}
    )
    top_models = {}

    for category, (category_results, top_n) in category_specs.items():
        if category.startswith("_cascada_fino_"):
            group = category.split("_")[-1]
            labels = df.loc[df["tpComando"] == group, "label"].astype(str).unique()
            if len(labels) == 1:
                top_models[category] = [
                    {
                        "rank": rank,
                        "categoria": category,
                        "nivel": "comando_fino",
                        "grupo": group,
                        "fixed_label": labels[0],
                        "feature_set": "N/A",
                        "clasificador": "Salida fija",
                        "reduction_method": default_method,
                        "f1_macro_cv": None,
                    }
                    for rank in range(1, top_n + 1)
                ]
                continue
        if len(category_results) < top_n:
            raise RuntimeError(
                f"No hay suficientes modelos evaluados para '{category}': "
                f"se requieren {top_n} y solo hay {len(category_results)}."
            )

        top_models[category] = []
        for rank, (_, row) in enumerate(category_results.head(top_n).iterrows(), 1):
            report_progress(
                progress_callback,
                f"Entrenando candidatos top: {category}, modelo {rank}/{top_n}...",
            )
            nivel = row["nivel"]
            grupo = row["grupo"]
            row_method = normalize_reduction_method(
                row["reduccion"] if "reduccion" in row and not pd.isna(row["reduccion"])
                else default_method
            )

            if nivel == "super_clase":
                group_key = "super_clase"
                group_df = df
                label_column = "tpComando"
            else:
                group_key = grupo
                group_df = df[df["tpComando"] == grupo]
                label_column = "label"

            model_key = (
                group_key,
                row_method,
                row["feature_set"],
                row["clasificador"],
            )
            if model_key not in trained_models:
                if group_key not in prepared_groups:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(
                        group_df[label_column].astype(str)
                    )
                    feature_sets = get_feature_sets(feature_cols)
                    prepared_groups[group_key] = (
                        group_df,
                        label_encoder,
                        y,
                        feature_sets,
                    )

                group_df, label_encoder, y, feature_sets = prepared_groups[group_key]
                n_components = _resolve_n_components(
                    row_method,
                    len(label_encoder.classes_),
                    pca_n_components=pca_n_components,
                )
                cols = feature_sets[row["feature_set"]]
                X = group_df[cols].to_numpy()
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                group_classifiers = classifiers_for_group(
                    classifiers,
                    len(label_encoder.classes_),
                    reduction_method=row_method,
                )
                pipeline = build_pipeline_hierarchical(
                    group_classifiers[row["clasificador"]],
                    X.shape[1],
                    k_best=int(row["k_best"]),
                    feature_names=cols,
                    reduction_method=row_method,
                    n_components=n_components,
                    n_samples=X.shape[0],
                )
                pipeline.fit(X, y)
                selected_features = list(
                    pipeline.named_steps["mrmr"].selected_feature_names_
                )
                model_entry = {
                    "nivel": nivel,
                    "grupo": grupo,
                    "pipeline": pipeline,
                    "label_encoder": label_encoder,
                    "feature_columns": cols,
                    "selected_feature_columns": selected_features,
                    "feature_set": row["feature_set"],
                    "clasificador": row["clasificador"],
                    "reduction_method": row_method,
                    "k_best": int(row["k_best"]),
                    "mrmr_ratio": len(selected_features) / len(cols),
                    "cohen_kappa": float(row["cohen_kappa"]),
                    "f1_macro_cv": float(row["f1_macro_mean"]),
                    "accuracy_mean": float(row["accuracy_mean"]),
                }
                if "pca" in pipeline.named_steps:
                    model_entry["pca_components"] = int(
                        pipeline.named_steps["pca"].n_components_
                    )
                elif "lda" in pipeline.named_steps:
                    model_entry["lda_components"] = int(
                        pipeline.named_steps["lda"].n_components
                    )
                trained_models[model_key] = model_entry

            if model_key not in evaluated_models:
                eval_df = group_df.copy()
                eval_df["label"] = eval_df[
                    "tpComando" if nivel == "super_clase" else "label"
                ]
                eval_df[feature_cols] = log_transform_power_columns(
                    eval_df[feature_cols], feature_cols
                )
                n_components = _resolve_n_components(
                    row_method,
                    len(label_encoder.classes_),
                    pca_n_components=pca_n_components,
                )
                report_progress(
                    progress_callback,
                    f"Validando candidato top: {category}, modelo {rank}/{top_n}...",
                )
                cv_result = evaluate_combination(
                    eval_df,
                    cols,
                    f"{nivel}-{grupo}",
                    row["clasificador"],
                    group_classifiers[row["clasificador"]],
                    reduction_method=row_method,
                    n_components=n_components,
                    k_best=int(row["k_best"]),
                    cv_splits=5,
                    verbose=False,
                )
                if cv_result[4] is None:
                    raise RuntimeError(
                        "No se pudo generar la matriz de confusión para "
                        f"{category} #{rank} ({nivel}/{grupo})."
                    )
                evaluated_models[model_key] = {
                    "y_pred": cv_result[4],
                    "label_encoder_cv": cv_result[5],
                    "y_true": cv_result[6],
                    "y_train_pred": cv_result[7],
                    "k_best": cv_result[8],
                    "cohen_kappa": cv_result[9],
                }

            top_models[category].append({
                **trained_models[model_key],
                **evaluated_models[model_key],
                "rank": rank,
                "categoria": category,
            })

    return top_models


def build_top_hierarchical_bundles(top_models, bundle_metadata):
    """Construye top 5 cascadas completas (super_clase → por_grupo) y especializadas."""
    bundles = {
        "completas": [],
        "letras": [],
        "numeros": [],
        "comandos": [],
    }
    category_groups = {
        "letras": "Letters",
        "numeros": "Numbers",
        "comandos": "Controls",
    }

    for rank in range(1, 6):
        super_model = top_models["_cascada_super"][rank - 1]
        per_group = {
            group: _to_inference_bundle(
                top_models[f"_cascada_fino_{group}"][rank - 1]
            )
            for group in ("Letters", "Numbers", "Controls")
        }

        bundles["completas"].append({
            **bundle_metadata,
            "cascade_type": "completa",
            "modelo_top_rank": rank,
            "super_clase": _to_inference_bundle(super_model),
            "por_grupo": per_group,
        })

        for category, group in category_groups.items():
            bundles[category].append({
                **bundle_metadata,
                "cascade_type": category,
                "cascade_group": group,
                "modelo_top_rank": rank,
                "por_grupo": {
                    group: _to_inference_bundle(
                        top_models[f"_cascada_fino_{group}"][rank - 1]
                    )
                },
            })

    return bundles


def _to_inference_bundle(model):
    """Reduce un modelo top a la estructura esperada por el predictor."""
    if "fixed_label" in model:
        return {
            "fixed_label": model["fixed_label"],
            "feature_columns": [],
            "feature_set": "N/A",
            "clasificador": "Salida fija",
            "f1_macro_cv": None,
        }
    return {
        "pipeline": model["pipeline"],
        "label_encoder": model["label_encoder"],
        "feature_columns": model["feature_columns"],
        "selected_feature_columns": model.get(
            "selected_feature_columns", []
        ),
        "feature_set": model["feature_set"],
        "clasificador": model["clasificador"],
        "reduction_method": model.get("reduction_method", "NADA"),
        "k_best": model["k_best"],
        "mrmr_ratio": model.get("mrmr_ratio"),
        "cohen_kappa": model["cohen_kappa"],
        "f1_macro_cv": model["f1_macro_cv"],
    }


def save_top_model_artifacts(
    usuario_models_dir, top_models_by_category, progress_callback=None
):
    """Guarda cada modelo completo y cada paso ajustado del pipeline por separado."""
    artifacts_dir = os.path.join(usuario_models_dir, "top_modelos")
    os.makedirs(artifacts_dir, exist_ok=True)

    for category, models in top_models_by_category.items():
        if category.startswith("_cascada_"):
            continue
        category_dir = os.path.join(artifacts_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for model in models:
            report_progress(
                progress_callback,
                f"Guardando artefactos: {category}, modelo "
                f"{model['rank']}/{len(models)}...",
            )
            method = model.get("reduction_method", "NADA")
            parts = (
                f"top_{model['rank']:02d}",
                model["nivel"],
                model["grupo"],
                method,
                model["feature_set"],
                model["clasificador"],
            )
            filename = "_".join(
                "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in str(part)
                )
                for part in parts
            )
            model_path = os.path.join(category_dir, f"{filename}.joblib")
            layers_dir = os.path.join(category_dir, f"{filename}_capas")
            os.makedirs(layers_dir, exist_ok=True)

            artifact = dict(model)
            artifact["archivo_capas"] = {}
            for step_name, fitted_step in model["pipeline"].named_steps.items():
                layer_path = os.path.join(layers_dir, f"{step_name}.joblib")
                joblib.dump(fitted_step, layer_path)
                artifact["archivo_capas"][step_name] = os.path.relpath(
                    layer_path, category_dir
                )

            joblib.dump(artifact, model_path)
            test_confusion_path = os.path.join(
                category_dir, f"{filename}_matriz_test.png"
            )
            train_confusion_path = os.path.join(
                category_dir, f"{filename}_matriz_train.png"
            )
            labels = [
                str(label) for label in model["label_encoder_cv"].classes_
            ]
            plot_confusion_matrix(
                model["y_true"],
                model["y_pred"],
                labels,
                (
                    f"{category} top {model['rank']} | "
                    f"{model['grupo']} ({method}) | TEST OOF | "
                    f"kappa={model['cohen_kappa']:.3f}"
                ),
                test_confusion_path,
            )
            plot_confusion_matrix(
                model["y_true"],
                model["y_train_pred"],
                labels,
                (
                    f"{category} top {model['rank']} | "
                    f"{model['grupo']} ({method}) | TRAIN CV | "
                    f"kappa={cohen_kappa_score(model['y_true'], model['y_train_pred']):.3f}"
                ),
                train_confusion_path,
            )
            print(
                f"   OK {category} #{model['rank']}: {model_path} "
                f"({len(artifact['archivo_capas'])} capas, matrices: "
                f"train={train_confusion_path}, test={test_confusion_path})"
            )

    return artifacts_dir


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
        test_path = os.path.join(figures_dir, f"cm_{nivel}_test.png")
        train_path = os.path.join(figures_dir, f"cm_{nivel}_train.png")
        test_metrics = (
            f"acc={accuracy_score(info['y_true'], info['y_pred']):.3f}, "
            f"f1_macro={f1_score(info['y_true'], info['y_pred'], average='macro'):.3f}, "
            f"kappa={cohen_kappa_score(info['y_true'], info['y_pred']):.3f}"
        )
        train_metrics = (
            f"acc={accuracy_score(info['y_true'], info['y_train_pred']):.3f}, "
            "f1_macro="
            f"{f1_score(info['y_true'], info['y_train_pred'], average='macro'):.3f}, "
            f"kappa={cohen_kappa_score(info['y_true'], info['y_train_pred']):.3f}"
        )
        plot_confusion_matrix(
            info["y_true"],
            info["y_pred"],
            labels,
            f"{usuario} | {nivel} | TEST OOF\n{test_metrics}",
            test_path,
        )
        plot_confusion_matrix(
            info["y_true"],
            info["y_train_pred"],
            labels,
            f"{usuario} | {nivel} | TRAIN CV\n{train_metrics}",
            train_path,
        )
        print(
            f"   ✓ Matrices guardadas: train={train_path}, test={test_path}"
        )


def train_optimized_hierarchical_TP(
    output_dir=None,
    channels=None,
    fs=None,
    use_p300window=None,
    apply_baseline_correction=None,
    use_win_size=None,
    window_size=None,
    window_overlap=None,
    max_recordings_per_class=None,
    reduction_method="NADA",  # Opciones: "LDA", "PCA", "NADA"
    pca_n_components=0.95,
    verbose=False,
    semilla=None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    method = normalize_reduction_method(reduction_method)
    output_dir = output_dir or config.OUTPUT_DIR
    FEATURES_CSV = os.path.join(output_dir, "features_dataset.csv")
    FIGURES_DIR = os.path.join(output_dir, "figures")
    MODELS_DIR = os.path.join(output_dir, "models")
    fs = fs if fs is not None else config.FS
    channels = channels if channels is not None else config.CHANNEL_NAMES
    use_p300window = (
        config.USE_P300_WINDOW_ONLY
        if use_p300window is None
        else use_p300window
    )
    apply_baseline_correction = (
        apply_baseline_correction
        if apply_baseline_correction is not None
        else config.APPLY_BASELINE_CORRECTION
    )
    use_win_size = (
        config.USE_WIN_SIZE if use_win_size is None else use_win_size
    )
    window_size = (
        config.WINDOW_SIZE if window_size is None else window_size
    )
    window_overlap = (
        config.WINDOW_OVERLAP
        if window_overlap is None
        else window_overlap
    )
    semilla = semilla if semilla is not None else config.RANDOM_STATE
    print("\n" + "=" * 100)
    print(f"ENTRENAMIENTO JERÁRQUICO DIRECTO (2 ETAPAS) | REDUCCIÓN: {method}")
    print("SUPER-GRUPO → COMANDO FINO (POR USUARIO)")
    print("=" * 100)

    # Cargar y preparar datos
    print("\n1) Cargando dataset...")
    report_progress(progress_callback, "Cargando el dataset de características...")
    df = pd.read_csv(FEATURES_CSV)
    df = normalize_string_columns(df, ["usuario", "tpComando", "letra", "label"])
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    df[feature_cols] = log_transform_power_columns(df[feature_cols], feature_cols)
    print(f"   ✓ Dataset cargado: {df.shape[0]} trials, {len(feature_cols)} features")

    usuarios = sorted(df["usuario"].unique())
    print(f"   ✓ Usuarios detectados: {usuarios}\n")

    resumen_global = []

    # Entrenar un modelo POR USUARIO
    for usuario in usuarios:
        print("\n" + "=" * 100)
        print(f"USUARIO: {usuario}")
        print("=" * 100)

        df_usuario = df[df["usuario"] == usuario].reset_index(drop=True)
        print(f"Trials para {usuario}: {len(df_usuario)}")

        # Paso 1-4: Evaluación exhaustiva (solo para este usuario)
        print(f"\n2) Evaluando TODAS las combinaciones para {usuario} (reducción={method})...")
        report_progress(
            progress_callback,
            f"Evaluando combinaciones de modelos para {usuario}...",
        )
        ini = datetime.now()
        print("Inicio de evaluacion exhaustiva", ini)
        results_df, best_per_group = evaluate_all_hierarchical(
            df_usuario,
            verbose=verbose,
            seed=semilla,
            reduction_method=method,
            pca_n_components=pca_n_components,
            progress_callback=progress_callback,
        )
        fin = datetime.now()
        print("Fin de evaluacion exhaustiva", fin)
        print("Tiempo de evaluacion exhaustiva", fin - ini)

        usuario_dir = os.path.join(output_dir, usuario)
        os.makedirs(usuario_dir, exist_ok=True)

        out_csv = os.path.join(usuario_dir, "evaluacion_jerarquica_completa.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"   ✓ Tabla guardada en: {out_csv}")
        report_progress(
            progress_callback,
            "Evaluación completa. Guardando matrices de confusión...",
        )

        usuario_figures_dir = os.path.join(FIGURES_DIR, usuario)
        save_confusion_matrices_per_user(usuario, best_per_group, usuario_figures_dir)

        print(f"\n   GANADOR POR CADA GRUPO ({usuario} | {method}):")
        print("   " + "-" * 140)
        print(
            f"   {'grupo':20s} | {'feature_set':24s} | {'clf':22s} | "
            f"{'n_features':10s} | {'k_best':7s} | {'Acc':7s} | "
            f"{'f1_cv':7s} | {'Kappa':7s}"
        )
        print("   " + "-" * 140)
        for grupo, info in best_per_group.items():
            if info.get("fixed_label") is not None:
                print(f"   {grupo:20s} | salida fija={ascii(info['fixed_label'])}")
                continue
            print(
                f"   {grupo:20s} | {info['feature_set']:24s} | "
                f"{info['clasificador']:22s} | {len(info['cols']):10d} | "
                f"{info['k_best']:7d} | "
                f"{info['accuracy_mean']:.3f} | {info['f1_macro']:.3f} | "
                f"{info['cohen_kappa']:.3f}"
            )

        # Paso 5: Entrenar modelos finales para este usuario
        print(f"\n3) Entrenando modelos finales (Super-grupo + Fino) para {usuario}...")
        report_progress(
            progress_callback,
            f"Ajustando modelos finales para {usuario}...",
        )
        final_bundles = train_final_models(
            df_usuario, best_per_group, progress_callback=progress_callback
        )
        mrmr_report_path = os.path.join(usuario_dir, "reporte_mrmr.txt")
        write_mrmr_report(usuario, final_bundles, mrmr_report_path)
        print(f"   ✓ Reporte de variables mRMR guardado en: {mrmr_report_path}")

        top_models_by_category = train_top_models_by_category(
            df_usuario,
            results_df,
            seed=semilla,
            reduction_method=method,
            pca_n_components=pca_n_components,
            progress_callback=progress_callback,
        )

        usuario_models_dir = os.path.join(MODELS_DIR, usuario)
        os.makedirs(usuario_models_dir, exist_ok=True)

        hierarchy_bundle = {
            "usuario": usuario,
            "reduction_method": method,
            "super_clase": final_bundles["super_clase"],
            "por_grupo": final_bundles["por_grupo"],
            "channel_names": channels,
            "fs": fs,
            "use_p300_window_only": use_p300window,
            "p300_window_s": config.P300_WINDOW_S,
            "apply_baseline_correction": apply_baseline_correction,
            "baseline_window_s": config.BASELINE_WINDOW_S,
            "use_win_size": use_win_size,
            "window_size": window_size,
            "window_overlap": window_overlap,
            "max_recordings_per_class": max_recordings_per_class,
            "log_transform_power": True,
            "use_flash_segmentation": True,
            "n_flashes": 2,
            "min_group_confidence": 0.0,
        }

        out_bundle = os.path.join(usuario_models_dir, "hierarchical_bundle_optimizado.joblib")
        report_progress(progress_callback, "Guardando el modelo jerárquico...")
        joblib.dump(hierarchy_bundle, out_bundle)
        print(f"   ✓ Bundle guardado en: {out_bundle}")

        cascade_metadata = {
            key: value
            for key, value in hierarchy_bundle.items()
            if key not in ("super_clase", "por_grupo")
        }
        top_cascades_by_category = build_top_hierarchical_bundles(
            top_models_by_category, cascade_metadata
        )
        cascades_dir = os.path.join(usuario_models_dir, "top_cascadas")
        os.makedirs(cascades_dir, exist_ok=True)
        for category, category_bundles in top_cascades_by_category.items():
            category_dir = (
                cascades_dir
                if category == "completas"
                else os.path.join(cascades_dir, category)
            )
            os.makedirs(category_dir, exist_ok=True)
            for rank, cascade_bundle in enumerate(category_bundles, 1):
                filename = (
                    f"hierarchical_bundle_optimizado_top_{rank}.joblib"
                    if category == "completas"
                    else f"hierarchical_bundle_{category}_top_{rank}.joblib"
                )
                cascade_path = os.path.join(category_dir, filename)
                joblib.dump(cascade_bundle, cascade_path)
                print(
                    f"   ✓ Cascada {category} top {rank} guardada: "
                    f"{cascade_path}"
                )

        top_models_dir = save_top_model_artifacts(
            usuario_models_dir,
            top_models_by_category,
            progress_callback=progress_callback,
        )
        print(f"   ✓ Modelos individuales y sus capas guardados en: {top_models_dir}")
        for category, models in top_models_by_category.items():
            if not category.startswith("_cascada_"):
                print(f"      {category}: {len(models)} modelos")

        best_row = results_df.iloc[0]
        resumen_global.append({
            "usuario": usuario,
            "reduccion": method,
            "n_trials": len(df_usuario),
            "mejor_nivel": best_row["nivel"],
            "mejor_grupo": best_row["grupo"],
            "mejor_feature_set": best_row["feature_set"],
            "mejor_clasificador": best_row["clasificador"],
            "accuracy_mean": float(best_row["accuracy_mean"]),
            "f1_macro_mean": float(best_row["f1_macro_mean"]),
            "cohen_kappa": float(best_row["cohen_kappa"]),
        })

    print("\n" + "=" * 100)
    print("RESUMEN FINAL: MEJOR COMBINACIÓN POR USUARIO")
    print("=" * 100)
    resumen_df = pd.DataFrame(resumen_global)
    print(resumen_df.to_string(index=False))

    resumen_path = os.path.join(config.OUTPUT_DIR, "resumen_entrenamiento_por_usuario.csv")
    resumen_df.to_csv(resumen_path, index=False)
    print(f"\n✓ Resumen guardado en: {resumen_path}")

    print("\n" + "=" * 100)
    print("ENTRENAMIENTO COMPLETADO PARA TODOS LOS USUARIOS")
    print("=" * 100)
    report_progress(progress_callback, "Entrenamiento completado.")


if __name__ == "__main__":
    # Cambia reduction_method a "LDA", "PCA" o "NADA"
    train_optimized_hierarchical_TP(reduction_method="NADA")