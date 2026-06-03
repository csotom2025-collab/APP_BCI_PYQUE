"""
PASO 3 — Pipeline Binario LDA con Decodificador de Comandos por Votación + Gráficas
===================================================================================
"""

import os
import re
import copy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
USUARIO    = 'UserPony'
RAND       = 42

CARPETA_ENTRADA = Path(f'results/{USUARIO}/Features_Avg')
CARPETA_SALIDA  = Path(f'results/{USUARIO}/Pipeline_P300')
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

def cargar_dataset_binario() -> pd.DataFrame:
    if not CARPETA_ENTRADA.exists():
        return pd.DataFrame()

    pattern = re.compile(r'^.+?_(.+?)_(\d+)_(p300|post)_features\.csv$', re.IGNORECASE)
    records = []

    for fname in sorted(os.listdir(CARPETA_ENTRADA)):
        m = pattern.match(fname)
        if not m:
            continue
        cmd, idx, tipo = m.group(1), int(m.group(2)), m.group(3)

        try:
            df = pd.read_csv(CARPETA_ENTRADA / fname)
            feat_cols = [c for c in df.columns if c != 'cmd_label']
            
            feats = df[feat_cols].iloc[0].to_dict()
            feats['cmd'] = cmd
            feats['epoch_idx'] = idx
            feats['is_target'] = 1 if tipo == 'p300' else 0
            
            records.append(feats)
        except Exception as e:
            pass

    return pd.DataFrame(records)

def build_pipelines() -> dict:
    return {
        'LDA (BCI Shrinkage Standard)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ]),
        'SVM Lineal Binario': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(kernel='linear', probability=True, random_state=RAND)),
        ]),
        'MLP Binario': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=RAND)),
        ]),
    }

if __name__ == '__main__':
    print('=' * 65)
    print('PASO 3 — Pipeline Binario + Decodificador + Gráficas')
    print('=' * 65 + '\n')

    df_all = cargar_dataset_binario()
    if df_all.empty:
        print('✗ Dataset vacío.')
        exit()

    feature_cols = [c for c in df_all.columns if c.startswith('sample_')]
    
    # ═════════════════════════════════════════════════════════════════════
    # LIMPIEZA DE NaNs
    # ═════════════════════════════════════════════════════════════════════
    df_all = df_all.dropna(subset=feature_cols, how='all')
    df_all[feature_cols] = df_all[feature_cols].fillna(0)
    # ═════════════════════════════════════════════════════════════════════

    all_epochs = sorted(df_all['epoch_idx'].unique())
    print('Épocas totales:', all_epochs)

    if len(all_epochs) < 2:
        print('Solo una época — train y test usan todo el dataset.')
        df_train = df_all
        df_test = df_all
    else:
        train_epochs = all_epochs[:-1]
        test_epoch = all_epochs[-1]
        print('Épocas train:', train_epochs)
        print('Época test:', test_epoch)
        print('Muestras train:', df_all[df_all['epoch_idx'].isin(train_epochs)].shape[0])
        print('Muestras test:', df_all[df_all['epoch_idx'] == test_epoch].shape[0])

        df_train = df_all[df_all['epoch_idx'].isin(train_epochs)]
        df_test = df_all[df_all['epoch_idx'] == test_epoch]

    # Conteo por comando para la evaluación del Speller (muestras usadas)
    comandos_test_global = sorted(df_test['cmd'].unique())
    conteos = []
    for true_cmd in comandos_test_global:
        row_target = df_test[(df_test['cmd'] == true_cmd) & (df_test['is_target'] == 1)]
        rows_nontarget = df_test[(df_test['cmd'] != true_cmd) & (df_test['is_target'] == 0)]
        conteos.append({
            'cmd': true_cmd,
            'n_target': len(row_target),
            'n_nontarget': len(rows_nontarget),
            'n_total': len(row_target) + len(rows_nontarget)
        })

    conteos_df = pd.DataFrame(conteos).sort_values('cmd') if len(conteos) > 0 else pd.DataFrame()
    ruta_conteos = CARPETA_SALIDA / 'conteo_muestras_por_comando.csv'
    if not conteos_df.empty:
        conteos_df.to_csv(ruta_conteos, index=False)
        print(f"Conteo muestras por comando guardado en: {ruta_conteos}")
        print(conteos_df.to_string(index=False))

    X_train = df_train[feature_cols].values
    y_train_binary = df_train['is_target'].values

    X_test = df_test[feature_cols].values
    y_test_binary = df_test['is_target'].values

    pipelines = build_pipelines()

    print(f'\n  Resultados de Clasificación')
    print(f'{"─"*85}')
    print(f'  {"Algoritmo":<30} | {"Acc Binaria":<12} | {"F1 Binario":<12} | {"ACC SELECCIÓN COMANDO (Speller)"}')
    print(f'{"─"*85}')

    resultados_tabla = []
    diccionario_predicciones = {}

    for nombre, pipe_tmpl in pipelines.items():
        pipe = copy.deepcopy(pipe_tmpl)
        pipe.fit(X_train, y_train_binary)
        
        y_pred_bin = pipe.predict(X_test)
        acc_bin = accuracy_score(y_test_binary, y_pred_bin)
        f1_bin = f1_score(y_test_binary, y_pred_bin, zero_division=0)

        comandos_test = df_test['cmd'].unique()
        
        y_true_speller = []
        y_pred_speller = []
        
        correctas_speller = 0
        total_intentos_speller = 0

        for true_cmd in comandos_test:
            row_target = df_test[(df_test['cmd'] == true_cmd) & (df_test['is_target'] == 1)]
            if row_target.empty: 
                continue
            
            rows_nontarget = df_test[(df_test['cmd'] != true_cmd) & (df_test['is_target'] == 0)]
            if rows_nontarget.empty: 
                continue

            df_bloque_eval = pd.concat([row_target, rows_nontarget], ignore_index=True)
            X_eval_speller = df_bloque_eval[feature_cols].values

            if hasattr(pipe.named_steps['clf'], 'decision_function'):
                scores = pipe.decision_function(X_eval_speller)
            else:
                scores = pipe.predict_proba(X_eval_speller)[:, 1]

            idx_ganador = np.argmax(scores)
            cmd_ganador = df_bloque_eval.loc[idx_ganador, 'cmd']

            # Guardamos la predicción para la matriz de confusión
            y_true_speller.append(true_cmd)
            y_pred_speller.append(cmd_ganador)

            if cmd_ganador == true_cmd:
                correctas_speller += 1
            total_intentos_speller += 1

        acc_speller = (correctas_speller / total_intentos_speller) if total_intentos_speller > 0 else 0.0

        # Guardamos en la tabla para el Heatmap de métricas
        resultados_tabla.append({
            'Algoritmo': nombre,
            'Acc Binaria': acc_bin,
            'F1 Binario': f1_bin,
            'Acc Speller': acc_speller
        })
        
        diccionario_predicciones[nombre] = {
            'y_true': y_true_speller,
            'y_pred': y_pred_speller,
            'y_test_binary': list(y_test_binary),
            'y_pred_binary': list(y_pred_bin)
        }

        print(f'  {nombre:<30} | {acc_bin:<12.2%} | {f1_bin:<12.2%} | {acc_speller:.2%}')

    print(f'{"─"*85}\n')

    # ═════════════════════════════════════════════════════════════════════
    # SECCIÓN DE GRÁFICAS
    # ═════════════════════════════════════════════════════════════════════
    
    # 1. Mapa de Calor (Heatmap) de las Métricas
    df_res = pd.DataFrame(resultados_tabla).set_index('Algoritmo')
    plt.figure(figsize=(9, 4))
    sns.heatmap(df_res, annot=True, cmap='viridis', fmt='.2%', cbar=True)
    plt.title('Comparativa de Modelos BCI (Mapa de Calor)', pad=15)
    plt.tight_layout()
    ruta_heatmap = CARPETA_SALIDA / 'heatmap_metricas.png'
    plt.savefig(ruta_heatmap)
    print(f"  ✓ Mapa de Calor guardado en: {ruta_heatmap}")
    
    # 2. Matriz de Confusión del Mejor Modelo de Speller
    mejor_modelo = df_res['Acc Speller'].idxmax()
    y_t = diccionario_predicciones[mejor_modelo]['y_true']
    y_p = diccionario_predicciones[mejor_modelo]['y_pred']
    
    if len(y_t) > 0:
        labels = sorted(list(set(y_t) | set(y_p)))
        cm = confusion_matrix(y_t, y_p, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels, cbar=False)
        plt.xlabel('Comando Predicho por el Speller', labelpad=15)
        plt.ylabel('Comando Real que el usuario miró', labelpad=15)
        plt.title(f'Matriz de Confusión (Speller) - {mejor_modelo}', pad=15)
        plt.tight_layout()
        ruta_cm = CARPETA_SALIDA / 'matriz_confusion_speller.png'
        plt.savefig(ruta_cm)
        print(f"  ✓ Matriz de Confusión guardada en: {ruta_cm}\n")
    
    # 3. Matriz de Confusión Binaria por Muestras (test samples) del Mejor Modelo
    y_test_bin = diccionario_predicciones[mejor_modelo].get('y_test_binary', [])
    y_pred_bin_all = diccionario_predicciones[mejor_modelo].get('y_pred_binary', [])
    if len(y_test_bin) > 0 and len(y_test_bin) == len(y_pred_bin_all):
        labels_bin = [0, 1]
        cm_bin = confusion_matrix(y_test_bin, y_pred_bin_all, labels=labels_bin)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_bin, annot=True, cmap='Oranges', fmt='g', xticklabels=['non-target', 'target'], yticklabels=['non-target', 'target'], cbar=False)
        plt.xlabel('Predicción Binaria', labelpad=10)
        plt.ylabel('Etiqueta Real', labelpad=10)
        plt.title(f'Matriz de Confusión Binaria (muestras) - {mejor_modelo}', pad=15)
        plt.tight_layout()
        ruta_cm_bin = CARPETA_SALIDA / 'matriz_confusion_binaria.png'
        plt.savefig(ruta_cm_bin)
        print(f"  ✓ Matriz de Confusión Binaria (muestras) guardada en: {ruta_cm_bin}\n")
    plt.show() # Mostrar las gráficas en pantalla