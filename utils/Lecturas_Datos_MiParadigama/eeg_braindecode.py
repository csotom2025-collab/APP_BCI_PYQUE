"""
EEG Deep Learning Pipeline — braindecode
==========================================
Carga archivos CSV con formato: {Usuario}_{Clase}_{Trial}.csv
Cada CSV = una época → (n_timesamples × n_channels)

Modelos evaluados (braindecode):
  EEGNetv4 · ShallowFBCSPNet · Deep4Net · EEGConformer
  EEGTCNet  · FBCNet · TSception · ATCNet

Métricas: Accuracy · F1-Score (macro) · AUC-ROC (macro OvR)
Salidas:  comparacion_modelos.png · confusion_matrix_mejor.png
          radar_metricas.png · resumen_metricas.csv

Uso:
    python eeg_braindecode.py --folder ./datos_eeg
    python eeg_braindecode.py --folder ./datos_eeg --output ./resultados --epochs 50 --lr 1e-3
"""

import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from braindecode.models import (
    EEGNetv4, ShallowFBCSPNet, Deep4Net,
    EEGConformer, EEGTCNet, FBCNet, TSception, ATCNet
)

# ─────────────────────────────────────────────────────────────
#  Colores
# ─────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52",
    "#8172B2","#937860","#DA8BC3","#8C8C8C"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
#  1. Carga de datos
# ─────────────────────────────────────────────────────────────
def load_eeg_folder(folder_path):
    """
    Lee todos los CSV de la carpeta.
    Nombre: {Usuario}_{Clase}_{Trial}.csv
    Retorna:
        X : np.ndarray  (n_epochs, n_channels, n_times)   float32
        y : np.ndarray  (n_epochs,)                       int64
    """
    pattern = re.compile(r"^(.+?)_(\d+)_(\d+)\.csv$", re.IGNORECASE)
    records, skipped = [], []

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No hay archivos CSV en: {folder_path}")

    print(f"\n{'='*58}")
    print(f"  EEG DEEP LEARNING PIPELINE  —  braindecode")
    print(f"{'='*58}")
    print(f"\n📂 Carpeta : {folder_path}")
    print(f"📄 Archivos: {len(csv_files)}")
    print(f"💻 Device  : {DEVICE}\n")

    n_times_ref = None
    n_channels_ref = None

    for fname in csv_files:
        m = pattern.match(fname)
        if not m:
            skipped.append(fname)
            continue

        usuario, clase, trial = m.group(1), int(m.group(2)), int(m.group(3))
        try:
            df = pd.read_csv(os.path.join(folder_path, fname))
            arr = df.values.astype(np.float32)   # (n_times, n_channels)

            if n_times_ref is None:
                n_times_ref    = arr.shape[0]
                n_channels_ref = arr.shape[1]
            else:
                # Recortar o rellenar para tener la misma longitud temporal
                if arr.shape[0] < n_times_ref:
                    pad = np.zeros((n_times_ref - arr.shape[0], arr.shape[1]), dtype=np.float32)
                    arr = np.vstack([arr, pad])
                elif arr.shape[0] > n_times_ref:
                    arr = arr[:n_times_ref, :]

            epoch = arr.T  # (n_channels, n_times)
            records.append({"usuario": usuario, "clase": clase, "trial": trial, "epoch": epoch})
        except Exception as e:
            skipped.append(f"{fname} ({e})")

    if skipped:
        print(f"⚠️  Omitidos: {', '.join(skipped[:5])}")
    if not records:
        raise ValueError("No se pudo cargar ningún archivo.")

    X = np.stack([r["epoch"] for r in records], axis=0)  # (N, C, T)
    y = np.array([r["clase"] for r in records], dtype=np.int64)

    le = LabelEncoder()
    y  = le.fit_transform(y).astype(np.int64)

    print(f"✅ Épocas cargadas : {X.shape[0]}")
    print(f"🧠 Forma           : {X.shape}  (épocas × canales × tiempo)")
    print(f"📊 Clases          : {le.classes_.tolist()}")
    classes_count = {int(c): int(np.sum(y == i)) for i, c in enumerate(le.classes_)}
    print(f"📈 Distribución    : {classes_count}\n")

    return X, y, le


# ─────────────────────────────────────────────────────────────
#  2. Definición de modelos
# ─────────────────────────────────────────────────────────────
def build_models(n_channels, n_classes, n_times, sfreq=250):
    """
    Instancia todos los modelos braindecode.
    Retorna dict: nombre → modelo (nn.Module)
    """
    models = {}

    # ── Compactos / Clásicos ──────────────────────────────────
    try:
        models["EEGNetv4"] = EEGNetv4(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=n_times,
            kernel_length=max(8, sfreq // 4),
            drop_prob=0.5,
        )
    except Exception as e:
        print(f"  ⚠️  EEGNetv4 no disponible: {e}")

    try:
        models["ShallowFBCSP"] = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=n_times,
            final_conv_length="auto",
        )
    except Exception as e:
        print(f"  ⚠️  ShallowFBCSP no disponible: {e}")

    try:
        models["Deep4Net"] = Deep4Net(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=n_times,
            final_conv_length="auto",
        )
    except Exception as e:
        print(f"  ⚠️  Deep4Net no disponible: {e}")

    # ── Híbridos / Temporales ─────────────────────────────────
    try:
        models["EEGConformer"] = EEGConformer(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
        )
    except Exception as e:
        print(f"  ⚠️  EEGConformer no disponible: {e}")

    try:
        models["EEGTCNet"] = EEGTCNet(
            in_chans=n_channels,
            n_classes=n_classes,
            layers_dilation=[1, 2, 4, 8],
            n_filters=32,
        )
    except Exception as e:
        print(f"  ⚠️  EEGTCNet no disponible: {e}")

    # ── Especializados ────────────────────────────────────────
    try:
        models["FBCNet"] = FBCNet(
            in_chans=n_channels,
            n_classes=n_classes,
            n_windows=9,
        )
    except Exception as e:
        print(f"  ⚠️  FBCNet no disponible: {e}")

    try:
        models["TSception"] = TSception(
            in_channels=n_channels,
            n_classes=n_classes,
            sampling_rate=sfreq,
            embed_dim=64,
        )
    except Exception as e:
        print(f"  ⚠️  TSception no disponible: {e}")

    try:
        models["ATCNet"] = ATCNet(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=n_times,
        )
    except Exception as e:
        print(f"  ⚠️  ATCNet no disponible: {e}")

    if not models:
        raise RuntimeError("Ningún modelo pudo instanciarse. Verifica la instalación de braindecode.")

    print(f"🤖 Modelos listos  : {list(models.keys())}\n")
    return models


# ─────────────────────────────────────────────────────────────
#  3. Entrenamiento & evaluación de un modelo
# ─────────────────────────────────────────────────────────────
def normalize_X(X):
    """Z-score por canal sobre toda la muestra."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std  = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, n_classes):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        out   = model(Xb)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(yb.numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_true))


def compute_metrics(y_true, y_pred, y_prob, n_classes):
    avg = "macro"
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=avg)
    except Exception:
        auc = float("nan")
    return acc, f1, auc


def evaluate_model(
    model_fn,           # callable que devuelve un modelo nuevo
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    n_classes, n_epochs, lr, batch_size
):
    """
    Entrena un modelo y evalúa en val y test.
    model_fn() se llama aquí para tener pesos frescos.
    """
    model = model_fn().to(DEVICE)

    # Tensores
    def make_loader(Xd, yd, shuffle=False):
        ds = TensorDataset(torch.tensor(Xd, dtype=torch.float32),
                           torch.tensor(yd, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    train_losses = []
    val_accs     = []

    for ep in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        train_losses.append(loss)

        # Early stopping suave: guardar mejor según val
        _, preds_v, true_v = predict(model, val_loader, n_classes)
        v_acc = accuracy_score(true_v, preds_v)
        val_accs.append(v_acc)
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Cargar mejor estado
    model.load_state_dict(best_state)

    # Métricas finales
    prob_v, pred_v, true_v = predict(model, val_loader,  n_classes)
    prob_t, pred_t, true_t = predict(model, test_loader, n_classes)

    val_metrics  = compute_metrics(true_v, pred_v, prob_v, n_classes)
    test_metrics = compute_metrics(true_t, pred_t, prob_t, n_classes)

    return {
        "val_acc":  val_metrics[0],  "val_f1":  val_metrics[1],  "val_auc":  val_metrics[2],
        "test_acc": test_metrics[0], "test_f1": test_metrics[1], "test_auc": test_metrics[2],
        "y_val":    true_v, "y_pred_val":  pred_v,
        "y_test":   true_t, "y_pred_test": pred_t,
        "train_losses": train_losses, "val_accs": val_accs,
    }


# ─────────────────────────────────────────────────────────────
#  4. Pipeline principal
# ─────────────────────────────────────────────────────────────
def run_pipeline(X, y, model_defs, n_classes,
                 n_epochs=30, lr=1e-3, batch_size=8,
                 test_size=0.20, val_size=0.15):
    """
    Para cada modelo: split → entrenar → evaluar.
    """
    X = normalize_X(X)

    # Splits estratificados
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=42
    )

    print(f"{'─'*58}")
    print(f"  ENTRENANDO {len(model_defs)} MODELOS")
    print(f"{'─'*58}")
    print(f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"  Epochs={n_epochs}  LR={lr}  Batch={batch_size}")
    print(f"{'─'*58}\n")

    hdr = f"  {'Modelo':<14} {'Val Acc':>8} {'Val F1':>8} {'Val AUC':>8}  {'Test Acc':>9} {'Test F1':>8} {'Test AUC':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    results = {}
    for name, model_fn in model_defs.items():
        try:
            r = evaluate_model(
                model_fn,
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                n_classes=n_classes,
                n_epochs=n_epochs,
                lr=lr,
                batch_size=batch_size,
            )
            results[name] = r
            print(f"  {name:<14}"
                  f"  {r['val_acc']:>7.4f}  {r['val_f1']:>7.4f}  {r['val_auc']:>7.4f}"
                  f"   {r['test_acc']:>8.4f}  {r['test_f1']:>7.4f}  {r['test_auc']:>8.4f}")
        except Exception as e:
            print(f"  {name:<14}  ❌ Error: {e}")

    if not results:
        raise RuntimeError("Todos los modelos fallaron durante el entrenamiento.")

    best = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\n🏆 Mejor modelo: {best}  (Test Acc={results[best]['test_acc']:.4f})\n")
    return results, best


# ─────────────────────────────────────────────────────────────
#  5. Visualizaciones
# ─────────────────────────────────────────────────────────────
def plot_comparison(results, output_path):
    names   = list(results.keys())
    metrics = [("val_acc","Val Accuracy"), ("val_f1","Val F1-Score"), ("val_auc","Val AUC-ROC"),
               ("test_acc","Test Accuracy"),("test_f1","Test F1-Score"),("test_auc","Test AUC-ROC")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparación de Modelos Deep Learning EEG\n(braindecode)",
                 fontsize=15, fontweight="bold")

    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [results[n][key] for n in names]
        bars = ax.barh(names, vals, color=PALETTE[:len(names)], edgecolor="white", height=0.55)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(label, fontsize=11)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(output_path, "comparacion_modelos.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 comparacion_modelos.png")


def plot_confusion_matrices(results, best_name, class_names, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Matrices de Confusión — {best_name}", fontsize=14, fontweight="bold")

    for ax, (split, yt, yp) in zip(axes, [
        ("Test",       results[best_name]["y_test"],  results[best_name]["y_pred_test"]),
        ("Validación", results[best_name]["y_val"],   results[best_name]["y_pred_val"]),
    ]):
        cm      = confusion_matrix(yt, yp)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Proporción"})
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.72, f"(n={cm[i,j]})",
                        ha="center", va="center", fontsize=8, color="gray")

        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        ax.set_title(f"{split}  |  Acc={acc:.3f}  F1={f1:.3f}", fontsize=11)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta Real")

    plt.tight_layout()
    p = os.path.join(output_path, "confusion_matrix_mejor.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 confusion_matrix_mejor.png")


def plot_radar(results, output_path):
    names   = list(results.keys())
    metrics = ["test_acc","test_f1","test_auc"]
    labels  = ["Test Accuracy","Test F1-Score","Test AUC-ROC"]
    N = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#f9f9f9")

    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics] + [results[name][metrics[0]]]
        ax.plot(angles, vals, "o-", lw=1.8, color=PALETTE[i % len(PALETTE)], label=name)
        ax.fill(angles, vals, alpha=0.06, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color="gray")
    ax.set_title("Métricas en Test — Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    p = os.path.join(output_path, "radar_metricas.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 radar_metricas.png")


def plot_training_curves(results, output_path):
    """Curvas de loss y val_acc durante entrenamiento."""
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Curvas de Entrenamiento por Modelo", fontsize=14, fontweight="bold")

    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        epochs = range(1, len(r["train_losses"]) + 1)
        color  = PALETTE[i % len(PALETTE)]

        ax2 = ax.twinx()
        ax.plot(epochs,  r["train_losses"], color=color, lw=2, label="Train Loss")
        ax2.plot(epochs, r["val_accs"],     color=color, lw=2, linestyle="--",
                 alpha=0.7, label="Val Acc")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss", color=color)
        ax2.set_ylabel("Val Acc", color=color)
        ax2.set_ylim(0, 1)
        ax.spines[["top"]].set_visible(False)

        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    p = os.path.join(output_path, "curvas_entrenamiento.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 curvas_entrenamiento.png")


def save_summary(results, best_name, output_path):
    rows = []
    for name, r in results.items():
        rows.append({
            "Modelo":    name,
            "Val Acc":   round(r["val_acc"],  4),
            "Val F1":    round(r["val_f1"],   4),
            "Val AUC":   round(r["val_auc"],  4),
            "Test Acc":  round(r["test_acc"], 4),
            "Test F1":   round(r["test_f1"],  4),
            "Test AUC":  round(r["test_auc"], 4),
            "Mejor":     "✓" if name == best_name else "",
        })

    df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False)
    p  = os.path.join(output_path, "resumen_metricas.csv")
    df.to_csv(p, index=False)
    print(f"  💾 resumen_metricas.csv")
    return df


def print_report(results, best_name, class_names):
    print(f"\n{'='*58}")
    print(f"  REPORTE DETALLADO — {best_name}")
    print(f"{'='*58}")
    r = results[best_name]
    for split, yt, yp in [
        ("TEST",       r["y_test"],  r["y_pred_test"]),
        ("VALIDACIÓN", r["y_val"],   r["y_pred_val"]),
    ]:
        print(f"\n  📋 {split}:")
        print(classification_report(yt, yp,
              target_names=[str(c) for c in class_names], zero_division=0))


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline EEG deep learning con modelos braindecode."
    )
    parser.add_argument("--folder", type=str,
                    default=r"D:/EEG_Python/results/Usermar/Digit/Separados",
                    help="Carpeta con CSV (formato: Usuario_Clase_Trial.csv)")
    parser.add_argument("--output",     type=str,
                    default=r"D:/EEG_Python/results/Usermar/Digit/Resultados_Clasificadores",
                        help="Carpeta de salida (default: misma que --folder)")
    parser.add_argument("--sfreq",      type=int, default=250,
                        help="Frecuencia de muestreo en Hz (default: 250)")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Épocas de entrenamiento por modelo (default: 30)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--test_size",  type=float, default=0.20,
                        help="Proporción test holdout (default: 0.20)")
    parser.add_argument("--val_size",   type=float, default=0.15,
                        help="Proporción validación holdout (default: 0.15)")
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Seleccionar modelos específicos, ej: --models EEGNetv4 Deep4Net")
    args = parser.parse_args()

    out_dir = args.output or args.folder
    os.makedirs(out_dir, exist_ok=True)

    # 1. Cargar datos
    X, y, le = load_eeg_folder(args.folder)
    n_channels, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    class_names = le.classes_

    # 2. Construir modelos
    all_models = build_models(n_channels, n_classes, n_times, sfreq=args.sfreq)

    # Filtrar si se especificaron modelos concretos
    if args.models:
        sel = {k: v for k, v in all_models.items() if k in args.models}
        if not sel:
            raise ValueError(f"Modelos solicitados no encontrados. Disponibles: {list(all_models.keys())}")
        all_models = sel

    # Convertir cada modelo a un callable (factory) con pesos frescos
    import copy
    model_fns = {name: (lambda m=m: copy.deepcopy(m)) for name, m in all_models.items()}

    # 3. Entrenar y evaluar
    results, best_name = run_pipeline(
        X, y, model_fns, n_classes,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    # 4. Reporte en consola
    print_report(results, best_name, class_names)

    # 5. Gráficas
    print(f"\n{'─'*58}")
    print(f"  GENERANDO VISUALIZACIONES")
    print(f"{'─'*58}")
    plot_comparison(results, out_dir)
    plot_confusion_matrices(results, best_name, class_names, out_dir)
    plot_radar(results, out_dir)
    plot_training_curves(results, out_dir)
    save_summary(results, best_name, out_dir)

    print(f"\n✅ Resultados guardados en: {out_dir}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
