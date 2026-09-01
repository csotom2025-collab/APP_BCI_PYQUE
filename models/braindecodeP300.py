"""
P300 Detection Pipeline with Braindecode + PyTorch

Tus datos tienen forma:

signals.shape =
    (n_trials, n_channels, n_samples)

Ejemplo:
    (300, 8, 1000)

labels.shape =
    (n_trials,)

y:
- 1 = hay P300
- 0 = no hay P300

"""

import os
from turtle import pd

from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from braindecode.models import EEGNetv4, EEGInceptionERP,InterpolatedSignalJEPA,EEGSimpleConv	

import matplotlib.pyplot as plt
import pandas as pd
import mne



CONFIG = {

    # ==========================
    # DATA
    # ==========================
    "sampling_rate": 500,

    # ventana temporal en segundos
    "tmin": 0.0,
    "tmax": 0.7,
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,

    # ==========================
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 1e-3,
    # ==========================
    "dropout": 0.5,
    # ==========================

    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# =========================================================
# UTILITIES
# =========================================================

def crop_time_window(
    signals,
    fs,
    tmin,
    tmax
):
    """
    signals:
        (N, C, T)

    returns:
        cropped signals
    """

    start_sample = int(tmin * fs)
    end_sample = int(tmax * fs)

    return signals[:, :, start_sample:end_sample]


def normalize_per_channel(signals):
    """
    Z-score por canal
    """

    mean = signals.mean(axis=-1, keepdims=True)
    std = signals.std(axis=-1, keepdims=True)

    return (signals - mean) / (std + 1e-8)


# =========================================================
# PYTORCH DATASET
# =========================================================

class P300Dataset(Dataset):

    def __init__(self, signals, labels):

        self.signals = torch.tensor(
            signals,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):

        x = self.signals[idx]
        y = self.labels[idx]

        return x, y


# =========================================================
# SPLIT DATA
# =========================================================

def create_datasets(signals,labels,config):

    X_train, X_temp, y_train, y_temp = train_test_split(
        signals,
        labels,
        train_size=config["train_size"],
        stratify=labels,
        random_state=42
    )

    relative_test_size = (
        config["test_size"] /
        (config["test_size"] + config["val_size"])
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=42
    )

    train_dataset = P300Dataset(X_train, y_train)
    val_dataset = P300Dataset(X_val, y_val)
    test_dataset = P300Dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


# =========================================================
# DATALOADERS
# =========================================================

def create_dataloaders(train_dataset,val_dataset,test_dataset,config):

    train_loader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True    )

    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False)

    test_loader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False)

    return train_loader, val_loader, test_loader


# =========================================================
# MODEL
# =========================================================

def create_model(n_channels,n_samples,config,model_type="eegnet"):

    if model_type == "eegnet":
        model = EEGNetv4(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_samples,
        drop_prob=config["dropout"]
    )
    elif model_type == "EEGInceptionERP":
        model = EEGInceptionERP(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_samples,
            drop_prob=config["dropout"]
        )
    elif model_type == "InterpolatedSignalJEPA":
        model = InterpolatedSignalJEPA(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_samples,
            drop_prob=config["dropout"]
        )
    elif model_type == "EEGSimpleConv":
        model = EEGSimpleConv(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=2,
            hidden_size=64,
            num_layers=2,
            dropout=config["dropout"]
        )

    return model.to(config["device"])


# =========================================================
# TRAIN LOOP
# =========================================================

def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()

    total_loss = 0
    preds_all = []
    labels_all = []

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)

    return total_loss / len(loader), acc


# =========================================================
# VALIDATION LOOP
# =========================================================

def validate(model,loader,criterion,device):

    model.eval()

    total_loss = 0

    preds_all = []
    labels_all = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)

    return total_loss / len(loader), acc


# =========================================================
# TEST LOOP
# =========================================================

def test_model(model,loader,device):

    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            outputs = model(x)

            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.numpy())

    acc = accuracy_score(labels_all, preds_all)

    cm = confusion_matrix(labels_all, preds_all)

    report = classification_report(
        labels_all,
        preds_all
    )

    return acc, cm, report


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_pipeline(signals, labels, config ,model_type="eegnet"):

    # =========================================
    # Normalize
    # =========================================

    signals = normalize_per_channel(signals)

    # =========================================
    # Create datasets
    # =========================================

    train_dataset, val_dataset, test_dataset = create_datasets(
        signals,
        labels,
        config
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config
    )

    # =========================================
    # Model
    # =========================================

    n_channels = signals.shape[1]
    n_samples = signals.shape[2]

    print(f"\nChannels: {n_channels}")
    print(f"Samples per trial: {n_samples}")

    model = create_model(
        n_channels,
        n_samples,
        config,
        model_type=model_type
    )

    # =========================================
    # Loss + optimizer
    # =========================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    # =========================================
    # TRAIN
    # =========================================

    for epoch in range(config["epochs"]):

        train_loss, train_acc = train_one_epoch(model,train_loader,optimizer,criterion,config["device"])

        val_loss, val_acc = validate(model,val_loader,criterion,config["device"])

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # =========================================
    # TEST
    # =========================================

    test_acc, cm, report = test_model(
        model,
        test_loader,
        config["device"]
    )

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")

    print(f"Accuracy: {test_acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # =========================================
    # Plot confusion matrix
    # =========================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No P300", "P300"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.show()

    return model


# =========================================================
# INFERENCE
# =========================================================

def predict_single_trial(model, signal, config):

    model.eval()

    # =========================================
    # Normalize
    # =========================================

    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)

    signal = (signal - mean) / (std + 1e-8)

    # =========================================
    # Tensor
    # =========================================

    x = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0)

    x = x.to(config["device"])

    # =========================================
    # Predict
    # =========================================

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1)

    return {
        "prediction": pred.item(),
        "probabilities": probs.cpu().numpy()
    }


class P300DataGenerator:

    def __init__(self, sampling_rate=250):

        self.fs = sampling_rate

    # =====================================================
    # Convertir tiempo -> samples
    # =====================================================

    def time_to_sample(self, t):

        return int(t * self.fs)

    # =====================================================
    # Extraer ventana temporal
    # =====================================================

    def extract_window(self,data,start_time,end_time):
        """
        data shape:
            (channels, samples)
        """

        start_sample = self.time_to_sample(start_time)
        end_sample = self.time_to_sample(end_time)

        return data[:, start_sample:end_sample]

    # =====================================================
    # Leer archivo CSV

    def read_csv_eeg(self, filepath):

        df = pd.read_csv(filepath)

        columns = df.columns

        # ignoramos columna tiempo
        eeg_channels = []

        for channel in columns[1:]:

            eeg_channels.append(
                np.array(df[channel])
            )

        eeg_channels = np.array(eeg_channels)

        return eeg_channels

    # =====================================================
    # Crear dataset desde archivos
    # =====================================================

    def generate_dataset_by_files(self,path,files,positive_window=(0.5, 1.2),negative_window=(1.2, 1.9)):
        menor =0
        X = []
        y = []

        for file_name in files:

            filepath = os.path.join(path, file_name)

            #print(f"Leyendo: {filepath}")

            eeg_data = self.read_csv_eeg(filepath)
            #print("EEG data shape:", eeg_data.shape)
            if eeg_data.shape[1] < 480:
                print(f"Advertencia: {file_name} tiene menos de 480 muestras. Se omite., tiene {eeg_data.shape[1]}")
                
                continue
            # =========================================
            # POSITIVE SAMPLE (P300)
            # =========================================

            positive_sample = self.extract_window(
                eeg_data,
                positive_window[0],
                positive_window[1]
            )

            X.append(positive_sample)
            y.append(1)

            # =========================================
            # NEGATIVE SAMPLE (NO P300)
            # =========================================

            negative_sample = self.extract_window(
                eeg_data,
                negative_window[0],
                negative_window[1]
            )

            X.append(negative_sample)
            y.append(0)
        print("Total samples generated:", len(X))
        print("Example sample shape:", X[0].shape)
        X = np.array(X)

        print("X shape:", X.shape)
        y = np.array(y)
        print("y shape:", y.shape)

        print("\nDataset generado:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Archivos menores al umbral",menor)
        return X, y


# =========================================================
# FUNCION AUXILIAR
# =========================================================

def create_dataset_from_folder(path,sampling_rate=250):

    files = os.listdir(path)

    csv_files = [
        file for file in files
        if file.lower().endswith(".csv")
    ]

    generator = P300DataGenerator(
        sampling_rate=sampling_rate
    )

    X, y = generator.generate_dataset_by_files(
        path,
        csv_files,

        # CONFIGURABLES
        positive_window=(0.5, 1.2),
        negative_window=(1.2, 1.9)
    )

    return X, y



# =========================================================
# MAIN
# =========================================================

def main():

    """
    EJEMPLO

    Debes reemplazar esto con tu dataset real.
    """

    n_trials = 300
    n_channels = 8
    n_samples = 1000

    path = "./captures/User69/Letters"
    print(path)
    X, y = create_dataset_from_folder(path)
    return
    # EEG random de ejemplo
    signals = X

    # labels binarias
    labels = y

    models = ["eegnet", "EEGInceptionERP","EEGSimpleConv"]
    for model_type in models:
        print(f"\n\n==============================")
        print(f"Training model: {model_type}")
        
        model = train_pipeline(
            signals,
            labels,
            CONFIG,
            model_type=model_type
        )


main()