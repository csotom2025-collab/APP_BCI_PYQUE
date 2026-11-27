"""
ENTRENAMIENTO DIRECTO MULTICLASE (SIN COLOR NI SUPER-GRUPOS: TODOS LOS COMANDOS JUNTOS)
CON SELECCIÓN DE REDUCCIÓN DIMENSIONAL: 'LDA', 'PCA' O 'NADA'

Flujo:
1. Detecta todos los comandos únicos en la columna 'label' del dataset
2. Evalúa combinaciones de clasificadores y feature_sets con validación cruzada
   aplicando StandardScaler → mRMR → [SafeLDA | PCA | NADA] → Clasificador
3. Guarda tabla completa de resultados (evaluacion_directa_completa.csv)
4. Entrena el modelo ganador con todos los comandos y guarda el bundle final
   listo para tiempo real (bundle_directo_optimizado.joblib)
5. Guarda los top 5 modelos y cada paso ajustado del pipeline por separado

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
    """Adapta hiperparámetros según el número total de comandos y la reducción usada."""
    method = normalize_reduction_method(reduction_method)
    adapted = {k: clone(v) for k, v in classifiers.items()}

    if "KNN_7" in adapted:
        if n_classes <= 3:
            k_optimo = 15
        else:
            k_optimo = n_classes if n_classes % 2 != 0 else n_classes + 1
            k_optimo = max(7, k_optimo)

        adapted["KNN_7"].set_params(n_neighbors=k_optimo, weights="distance")

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


def build_pipeline_direct(
    clf,
    n_features_in,
    k_best=None,
    feature_names=None,
    reduction_method="NADA",
    n_components=None,
    n_samples=None,
):
    """
    Pipeline directo para clasificación de todos los comandos.

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
# PASO 3: EVALUACIÓN DE 1 COMBINACIÓN (clf × feature_set)
# ===========================================================================================
def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    """Guarda una matriz de confusión normalizada en una imagen."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig_size = max(6, len(labels) * 0.45)

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
    task_name,
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
                f"  {task_name:16s} × {clf_name:22s} [{method:4s}] | "
                f"muestras train/test por fold: "
                f"{', '.join(fold_sample_counts)} | "
                f"k_best={best_result[8]} | "
                f"acc={best_result[0]:.3f}±{best_result[1]:.3f} | "
                f"f1={best_result[2]:.3f}±{best_result[3]:.3f} | "
                f"kappa={best_result[9]:.3f}"
            )

        return best_result
    except Exception as e:
        print(f"  [ERROR] {task_name} × {clf_name}: {type(e).__name__}: {e}")
        return None, None, None, None, None, None, None, None, None, None


# ===========================================================================================
# PASO 4: EVALUACIÓN EXHAUSTIVA DIRECTA (TODOS LOS COMANDOS JUNTOS)
# ===========================================================================================
def evaluate_all_direct(
    df,
    verbose=True,
    seed=42,
    reduction_method="NADA",
    pca_n_components=0.95,
    progress_callback=None,
):
    """Evalúa todas las combinaciones directamente sobre todos los comandos en 'label'."""
    method = normalize_reduction_method(reduction_method)
    feature_cols_all = [c for c in df.columns if c not in META_COLUMNS]

    all_commands = sorted(df["label"].astype(str).unique())
    n_classes = len(all_commands)
    if n_classes < 2:
        raise ValueError(
            f"Se requieren al menos 2 comandos distintos en 'label'; encontrados: {all_commands}"
        )

    if verbose:
        print("\n" + "=" * 100)
        print(
            f"EVALUANDO CLASIFICACIÓN DIRECTA ({n_classes} comandos) | REDUCCIÓN: {method}"
        )
        print(f"Comandos detectados: {all_commands}")
        print("=" * 100)

    df_eval = df.copy()
    df_eval[feature_cols_all] = log_transform_power_columns(
        df_eval[feature_cols_all], feature_cols_all
    )
    feature_sets = get_feature_sets(feature_cols_all)
    n_components = _resolve_n_components(
        method, n_classes, pca_n_components=pca_n_components
    )
    base_classifiers = build_classifiers(n_classes, seed=seed)
    classifiers = classifiers_for_group(
        base_classifiers, n_classes, reduction_method=method
    )

    results = []
    best_model = None
    best_f1 = -1.0

    for set_name, cols in feature_sets.items():
        if verbose:
            print(f"\n{set_name} ({len(cols)} features):")
        cv_cache = {}
        for clf_name, clf in classifiers.items():
            report_progress(
                progress_callback,
                f"Evaluando comandos directos ({method}): {set_name} / {clf_name}...",
            )
            (
                acc_m, acc_s, f1_m, f1_s, y_pred, le, y_true,
                y_train_pred, selected_k_best, selected_kappa,
            ) = evaluate_combination(
                df_eval,
                cols,
                "todos_comandos",
                clf_name,
                clf,
                reduction_method=method,
                n_components=n_components,
                cv_splits=5,
                verbose=verbose,
                cv_cache=cv_cache,
            )
            if acc_m is None:
                continue

            results.append({
                "nivel": "comando_directo",
                "grupo": "Todos",
                "n_clases": len(le.classes_),
                "reduccion": method,
                "n_components": n_components,
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
                best_model = {
                    "feature_set": set_name,
                    "cols": cols,
                    "clasificador": clf_name,
                    "clf": clf,
                    "reduction_method": method,
                    "n_components": n_components,
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

    if not results or best_model is None:
        raise RuntimeError(
            "Ninguna combinación produjo resultados válidos. "
            "Revisa los errores impresos arriba para ver qué falló."
        )

    results_df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    return results_df, best_model


# ===========================================================================================
# PASO 5: ENTRENAR MODELO GANADOR Y TOP 5
# ===========================================================================================
def train_final_direct_model(df, best_model, progress_callback=None):
    """Entrena el modelo ganador con todos los trials del usuario."""
    method = normalize_reduction_method(
        best_model.get("reduction_method", "NADA")
    )
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))
    X = df[best_model["cols"]].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    report_progress(progress_callback, "Ajustando modelo final directo...")
    pipeline = build_pipeline_direct(
        best_model["clf"],
        X.shape[1],
        k_best=best_model["k_best"],
        feature_names=best_model["cols"],
        reduction_method=method,
        n_components=best_model.get("n_components"),
        n_samples=X.shape[0],
    )
    pipeline.fit(X, y)
    selected_features = list(
        pipeline.named_steps["mrmr"].selected_feature_names_
    )

    bundle = {
        "pipeline": pipeline,
        "label_encoder": le,
        "clases": [str(c) for c in le.classes_],
        "feature_columns": best_model["cols"],
        "selected_feature_columns": selected_features,
        "feature_set": best_model["feature_set"],
        "clasificador": best_model["clasificador"],
        "reduction_method": method,
        "k_best": best_model["k_best"],
        "mrmr_ratio": len(selected_features) / len(best_model["cols"]),
        "cohen_kappa": best_model["cohen_kappa"],
        "accuracy_mean": float(best_model["accuracy_mean"]),
        "f1_macro_cv": float(best_model["f1_macro"]),
    }
    if "pca" in pipeline.named_steps:
        bundle["pca_components"] = int(pipeline.named_steps["pca"].n_components_)
    elif "lda" in pipeline.named_steps:
        bundle["lda_components"] = int(pipeline.named_steps["lda"].n_components_)

    return bundle


def write_mrmr_report(usuario, final_model, report_path):
    """Escribe las variables elegidas por mRMR y un resumen de canales."""
    channel_counts = Counter()
    channel_relevance = {}

    selector = final_model["pipeline"].named_steps["mrmr"]
    selected_features = final_model.get(
        "selected_feature_columns",
        selector.selected_feature_names_,
    )
    input_features = final_model["feature_columns"]
    ratio = len(selected_features) / len(input_features)
    method = final_model.get("reduction_method", "NADA")

    lines = [
        f"Reporte mRMR (Clasificación Directa) - Usuario: {usuario}",
        "=" * 72,
        f"Comandos ({len(final_model['clases'])}): {', '.join(final_model['clases'])}",
        f"Clasificador: {final_model['clasificador']} (Reducción: {method})",
        f"Conjunto de entrada: {final_model['feature_set']} ({len(input_features)} variables)",
        f"Variables mRMR: {len(selected_features)} ({ratio:.1%} de entrada)",
        f"Métricas CV: F1 macro={final_model['f1_macro_cv']:.4f}, "
        f"accuracy={final_model['accuracy_mean']:.4f}, "
        f"kappa={final_model['cohen_kappa']:.4f}",
        "",
        "Variables seleccionadas (orden mRMR):",
        "-" * 72,
    ]

    relevance_by_feature = dict(zip(input_features, selector.relevance_scores_))
    for feature in selected_features:
        channel = str(feature).split("_", 1)[0]
        relevance = float(relevance_by_feature[feature])
        lines.append(f"  - {feature} (relevancia MI={relevance:.5f})")
        channel_counts[channel] += 1
        channel_relevance.setdefault(channel, []).append(relevance)

    lines.extend([
        "",
        "Canales con mayor presencia en la selección mRMR",
        "-" * 72,
    ])
    for channel, count in sorted(
        channel_counts.items(),
        key=lambda item: (-item[1], -float(np.mean(channel_relevance[item[0]])), item[0]),
    ):
        mean_relevance = float(np.mean(channel_relevance[channel]))
        lines.append(
            f"  {channel}: {count} variables seleccionadas, "
            f"relevancia MI media={mean_relevance:.5f}"
        )

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def train_top_direct_models(
    df,
    results_df,
    top_n=5,
    seed=42,
    reduction_method="NADA",
    pca_n_components=0.95,
    progress_callback=None,
):
    """Entrena los top N modelos directos y calcula sus predicciones CV."""
    default_method = normalize_reduction_method(reduction_method)
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    feature_sets = get_feature_sets(feature_cols)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"].astype(str))
    n_classes = len(label_encoder.classes_)
    base_classifiers = build_classifiers(n_classes, seed=seed)

    ranked = results_df.sort_values("f1_macro_mean", ascending=False, kind="stable")
    limit = min(top_n, len(ranked))
    top_models = []

    for rank, (_, row) in enumerate(ranked.head(limit).iterrows(), 1):
        report_progress(
            progress_callback,
            f"Entrenando modelo directo top {rank}/{limit}...",
        )
        row_method = normalize_reduction_method(
            row["reduccion"]
            if "reduccion" in row and not pd.isna(row["reduccion"])
            else default_method
        )
        n_components = _resolve_n_components(
            row_method, n_classes, pca_n_components=pca_n_components
        )
        group_classifiers = classifiers_for_group(
            base_classifiers, n_classes, reduction_method=row_method
        )
        cols = feature_sets[row["feature_set"]]
        X = df[cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pipeline = build_pipeline_direct(
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

        cv_result = evaluate_combination(
            df,
            cols,
            "todos_comandos",
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
                f"No se pudo generar la matriz de confusión para el modelo top #{rank}."
            )

        entry = {
            "rank": rank,
            "nivel": "comando_directo",
            "grupo": "Todos",
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "clases": [str(c) for c in label_encoder.classes_],
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
            "y_pred": cv_result[4],
            "label_encoder_cv": cv_result[5],
            "y_true": cv_result[6],
            "y_train_pred": cv_result[7],
        }
        if "pca" in pipeline.named_steps:
            entry["pca_components"] = int(pipeline.named_steps["pca"].n_components_)
        elif "lda" in pipeline.named_steps:
            entry["lda_components"] = int(pipeline.named_steps["lda"].n_components_)

        top_models.append(entry)

    return top_models


def save_top_model_artifacts(
    usuario_models_dir, top_models, bundle_metadata, progress_callback=None
):
    """Guarda cada modelo top completo, sus capas por separado y sus matrices."""
    artifacts_dir = os.path.join(usuario_models_dir, "top_modelos_directos")
    os.makedirs(artifacts_dir, exist_ok=True)

    for model in top_models:
        report_progress(
            progress_callback,
            f"Guardando artefactos top {model['rank']}/{len(top_models)}...",
        )
        method = model.get("reduction_method", "NADA")
        parts = (
            f"top_{model['rank']:02d}",
            "directo",
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
        model_path = os.path.join(artifacts_dir, f"{filename}.joblib")
        layers_dir = os.path.join(artifacts_dir, f"{filename}_capas")
        os.makedirs(layers_dir, exist_ok=True)

        artifact = {
            **bundle_metadata,
            **model,
            "archivo_capas": {},
        }
        for step_name, fitted_step in model["pipeline"].named_steps.items():
            layer_path = os.path.join(layers_dir, f"{step_name}.joblib")
            joblib.dump(fitted_step, layer_path)
            artifact["archivo_capas"][step_name] = os.path.relpath(
                layer_path, artifacts_dir
            )

        joblib.dump(artifact, model_path)
        test_confusion_path = os.path.join(
            artifacts_dir, f"{filename}_matriz_test.png"
        )
        train_confusion_path = os.path.join(
            artifacts_dir, f"{filename}_matriz_train.png"
        )
        labels = [str(label) for label in model["label_encoder_cv"].classes_]
        plot_confusion_matrix(
            model["y_true"],
            model["y_pred"],
            labels,
            (
                f"Directo top {model['rank']} ({method}) | TEST OOF | "
                f"kappa={model['cohen_kappa']:.3f}"
            ),
            test_confusion_path,
        )
        plot_confusion_matrix(
            model["y_true"],
            model["y_train_pred"],
            labels,
            (
                f"Directo top {model['rank']} ({method}) | TRAIN CV | "
                f"kappa={cohen_kappa_score(model['y_true'], model['y_train_pred']):.3f}"
            ),
            train_confusion_path,
        )
        print(
            f"   OK top #{model['rank']}: {model_path} "
            f"({len(artifact['archivo_capas'])} capas)"
        )

    return artifacts_dir


# ===========================================================================================
# PASO 6: MAIN
# ===========================================================================================
def save_confusion_matrices_per_user(usuario, best_model, figures_dir):
    """Guarda matrices de confusión train/test del modelo ganador del usuario."""
    os.makedirs(figures_dir, exist_ok=True)
    labels = [str(lbl) for lbl in best_model["label_encoder"].classes_]
    test_path = os.path.join(figures_dir, "cm_directo_todos_test.png")
    train_path = os.path.join(figures_dir, "cm_directo_todos_train.png")
    test_metrics = (
        f"acc={accuracy_score(best_model['y_true'], best_model['y_pred']):.3f}, "
        f"f1_macro={f1_score(best_model['y_true'], best_model['y_pred'], average='macro'):.3f}, "
        f"kappa={cohen_kappa_score(best_model['y_true'], best_model['y_pred']):.3f}"
    )
    train_metrics = (
        f"acc={accuracy_score(best_model['y_true'], best_model['y_train_pred']):.3f}, "
        "f1_macro="
        f"{f1_score(best_model['y_true'], best_model['y_train_pred'], average='macro'):.3f}, "
        f"kappa={cohen_kappa_score(best_model['y_true'], best_model['y_train_pred']):.3f}"
    )
    plot_confusion_matrix(
        best_model["y_true"],
        best_model["y_pred"],
        labels,
        f"{usuario} | Directo ({len(labels)} comandos) | TEST OOF\n{test_metrics}",
        test_path,
    )
    plot_confusion_matrix(
        best_model["y_true"],
        best_model["y_train_pred"],
        labels,
        f"{usuario} | Directo ({len(labels)} comandos) | TRAIN CV\n{train_metrics}",
        train_path,
    )
    print(f"   ✓ Matrices guardadas: train={train_path}, test={test_path}")


def train_optimized_direct(
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
    top_n=5,
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
    print(f"ENTRENAMIENTO DIRECTO MULTICLASE (TODOS LOS COMANDOS) | REDUCCIÓN: {method}")
    print("POR USUARIO")
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

    for usuario in usuarios:
        print("\n" + "=" * 100)
        print(f"USUARIO: {usuario}")
        print("=" * 100)

        df_usuario = df[df["usuario"] == usuario].reset_index(drop=True)
        comandos_usuario = sorted(df_usuario["label"].astype(str).unique())
        print(
            f"Trials para {usuario}: {len(df_usuario)} | "
            f"Comandos ({len(comandos_usuario)}): {comandos_usuario}"
        )

        # Evaluación exhaustiva directa
        print(f"\n2) Evaluando TODAS las combinaciones para {usuario} (reducción={method})...")
        report_progress(
            progress_callback,
            f"Evaluando combinaciones directas para {usuario}...",
        )
        ini = datetime.now()
        results_df, best_model = evaluate_all_direct(
            df_usuario,
            verbose=verbose,
            seed=semilla,
            reduction_method=method,
            pca_n_components=pca_n_components,
            progress_callback=progress_callback,
        )
        fin = datetime.now()
        print("Tiempo de evaluacion exhaustiva:", fin - ini)

        usuario_dir = os.path.join(output_dir, usuario)
        os.makedirs(usuario_dir, exist_ok=True)

        out_csv = os.path.join(usuario_dir, "evaluacion_directa_completa.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"   ✓ Tabla guardada en: {out_csv}")

        usuario_figures_dir = os.path.join(FIGURES_DIR, usuario)
        save_confusion_matrices_per_user(usuario, best_model, usuario_figures_dir)

        print(f"\n   GANADOR DIRECTO ({usuario} | {method}):")
        print("   " + "-" * 120)
        print(
            f"   feature_set={best_model['feature_set']} | "
            f"clf={best_model['clasificador']} | "
            f"n_features={len(best_model['cols'])} | "
            f"k_best={best_model['k_best']} | "
            f"Acc={best_model['accuracy_mean']:.3f} | "
            f"f1_cv={best_model['f1_macro']:.3f} | "
            f"Kappa={best_model['cohen_kappa']:.3f}"
        )

        # Entrenar modelo ganador final
        print(f"\n3) Entrenando modelo final directo para {usuario}...")
        final_model_bundle = train_final_direct_model(
            df_usuario, best_model, progress_callback=progress_callback
        )

        mrmr_report_path = os.path.join(usuario_dir, "reporte_mrmr_directo.txt")
        write_mrmr_report(usuario, final_model_bundle, mrmr_report_path)
        print(f"   ✓ Reporte mRMR guardado en: {mrmr_report_path}")

        top_models = train_top_direct_models(
            df_usuario,
            results_df,
            top_n=top_n,
            seed=semilla,
            reduction_method=method,
            pca_n_components=pca_n_components,
            progress_callback=progress_callback,
        )

        usuario_models_dir = os.path.join(MODELS_DIR, usuario)
        os.makedirs(usuario_models_dir, exist_ok=True)

        bundle_metadata = {
            "usuario": usuario,
            "cascade_type": "directa",
            "reduction_method": method,
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
        }

        direct_bundle = {
            **bundle_metadata,
            "modelo": final_model_bundle,
        }

        out_bundle = os.path.join(usuario_models_dir, "bundle_directo_optimizado.joblib")
        report_progress(progress_callback, "Guardando el modelo directo...")
        joblib.dump(direct_bundle, out_bundle)
        print(f"   ✓ Bundle guardado en: {out_bundle}")

        top_models_dir = save_top_model_artifacts(
            usuario_models_dir,
            top_models,
            bundle_metadata,
            progress_callback=progress_callback,
        )
        print(f"   ✓ Top {len(top_models)} modelos directos guardados en: {top_models_dir}")

        best_row = results_df.iloc[0]
        resumen_global.append({
            "usuario": usuario,
            "reduccion": method,
            "n_trials": len(df_usuario),
            "n_comandos": len(comandos_usuario),
            "mejor_feature_set": best_row["feature_set"],
            "mejor_clasificador": best_row["clasificador"],
            "k_best": int(best_row["k_best"]),
            "accuracy_mean": float(best_row["accuracy_mean"]),
            "f1_macro_mean": float(best_row["f1_macro_mean"]),
            "cohen_kappa": float(best_row["cohen_kappa"]),
        })

    print("\n" + "=" * 100)
    print("RESUMEN FINAL: MEJOR COMBINACIÓN DIRECTA POR USUARIO")
    print("=" * 100)
    resumen_df = pd.DataFrame(resumen_global)
    print(resumen_df.to_string(index=False))

    resumen_path = os.path.join(config.OUTPUT_DIR, "resumen_entrenamiento_directo_por_usuario.csv")
    resumen_df.to_csv(resumen_path, index=False)
    print(f"\n✓ Resumen guardado en: {resumen_path}")

    print("\n" + "=" * 100)
    print("ENTRENAMIENTO DIRECTO COMPLETADO PARA TODOS LOS USUARIOS")
    print("=" * 100)
    report_progress(progress_callback, "Entrenamiento directo completado.")


if __name__ == "__main__":
    # Cambia reduction_method a "LDA", "PCA" o "NADA"
    train_optimized_direct(reduction_method="NADA")