import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score, roc_auc_score

# Imports para los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# --- 1. CONFIGURACIÓN ---
# Carpeta raíz donde están las subcarpetas generadas por optimizacion_lda.py
# Estructura esperada:
#   LDA_BASE/Digit/Estadisticas/Usermar_0_0_features_lda.csv
#   LDA_BASE/Digit/Wavelets/Usermar_0_0_features_lda.csv
#   ...  cada CSV tiene columnas: LD1, LD2, ..., LDn, label

LDA_BASE   = r'D:/EEG_Python/results/Usermar/Char/Resultados_LDA'
OUTPUT_DIR = r'D:/EEG_Python/results/Usermar/Char/Resultados_Clasificadores'

os.makedirs(OUTPUT_DIR, exist_ok=True)

tipClases = {'Char': None}   # las etiquetas se detectan automáticamente desde los CSV

# Diccionario global para almacenar resultados de todos los tipos
resultados_globales = {}

# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL: PROCESAR CADA TIPO DE CLASE POR SEPARADO
# ══════════════════════════════════════════════════════════════════════════════
for tipo_clase in tipClases.keys():

    print(f"\n{'#'*80}")
    print(f"# PROCESANDO TIPO DE CLASE: {tipo_clase}")
    print(f"{'#'*80}\n")

    # --- 2. CARGA DE DATOS LDA ---
    # Cada subcarpeta dentro de LDA_BASE/tipo_clase/ es un grupo de features
    # (Estadisticas, Wavelets, Frecuencias_Abs, etc.)
    base_tipo = os.path.join(LDA_BASE, tipo_clase)

    if not os.path.isdir(base_tipo):
        print(f"Carpeta no encontrada: {base_tipo}. Omitiendo...")
        continue

    # Detectar subcarpetas (grupos LDA)
    grupos_disponibles = sorted([
        d for d in os.listdir(base_tipo)
        if os.path.isdir(os.path.join(base_tipo, d))
    ])

    if not grupos_disponibles:
        print(f"No se encontraron subcarpetas en {base_tipo}. Omitiendo...")
        continue

    print(f"Grupos LDA detectados: {grupos_disponibles}\n")

    # Cargar todos los grupos en un dict: { nombre_grupo -> DataFrame }
    feature_sets = {}
    for grupo in grupos_disponibles:
        grupo_path = os.path.join(base_tipo, grupo)
        csv_files  = sorted([f for f in os.listdir(grupo_path)
                              if f.lower().endswith('.csv')])
        if not csv_files:
            print(f"  Sin CSV en {grupo}. Omitiendo grupo...")
            continue

        dfs = []
        for fname in csv_files:
            fpath = os.path.join(grupo_path, fname)
            try:
                df = pd.read_csv(fpath)
                if 'label' not in df.columns:
                    print(f"  Archivo sin columna 'label': {fname}. Omitiendo...")
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"  Error leyendo {fname}: {e}")

        if dfs:
            data_grupo = pd.concat(dfs, ignore_index=True)
            feature_sets[grupo] = data_grupo
            ld_cols = [c for c in data_grupo.columns if c.startswith('LD')]
            print(f"  ✔ {grupo:<25} → {len(data_grupo):3} épocas | {len(ld_cols):2} componentes LDA")

    if not feature_sets:
        print(f"No se cargó ningún grupo LDA para {tipo_clase}. Omitiendo...")
        continue

    # Detectar etiquetas de clase desde los datos cargados
    todas_etiquetas = set()
    for df in feature_sets.values():
        todas_etiquetas.update(df['label'].astype(str).unique())
    etiquetas_clase = sorted(todas_etiquetas)
    N_CLASES = len(etiquetas_clase)
    print(f"\nDetectadas {N_CLASES} clases: {etiquetas_clase}")

    # --- 3. DEFINICIÓN DE MODELOS (sin K-Means) ---
    randNum = 42

    modelos = {
        'Regresión Logística': LogisticRegression(
            max_iter=1000, random_state=randNum,
            penalty='l2', solver='lbfgs', C=10.**10
        ),
        'SVM Lineal': SVC(
            kernel='linear', probability=True, random_state=randNum
        ),
        'SVM RBF': SVC(
            kernel='rbf', probability=True, random_state=randNum
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, eval_metric='mlogloss',
            verbosity=0, random_state=randNum
        ),
        'Red Neuronal MLP': MLPClassifier(
            max_iter=1000, hidden_layer_sizes=(1000,), random_state=randNum
        ),
    }

    # --- 4. EVALUACIÓN DE MODELOS (LOOP POR GRUPO LDA Y MODELO) ---
    print("\n" + "="*80)
    print(f"EVALUACIÓN DE MÚLTIPLES MODELOS - {tipo_clase}")
    print("="*80)

    all_results  = {}
    best_overall = {
        'accuracy': -1, 'f1': -1, 'auc': -1,
        'model': None, 'features': None,
        'y_pred': None, 'X_test': None, 'y_test': None
    }

    for feat_name, data_grupo in feature_sets.items():

        print(f"\n{'='*60}")
        print(f"GRUPO LDA: {feat_name}")
        print(f"{'='*60}")

        # Separar X (componentes LD*) e y (label)
        ld_cols = [c for c in data_grupo.columns if c.startswith('LD')]
        X       = data_grupo[ld_cols].values.astype(float)
        y_raw   = data_grupo['label'].astype(str).values

        # Manejar NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # División train-test (80-20) con escalado sin data leakage
        TRAIN_SIZE = 0.8
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=randNum,
                train_size=TRAIN_SIZE, stratify=y_raw
            )
        except ValueError:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=randNum, train_size=TRAIN_SIZE
            )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        # Codificar etiquetas (necesario para XGBoost)
        le = LabelEncoder()
        le.fit(etiquetas_clase)             # consistencia entre grupos
        y_train_enc = le.transform(y_train)
        y_test_enc  = le.transform(y_test)

        # Evaluar cada modelo
        results_feat = {}

        for model_name, model_template in modelos.items():
            model = copy.deepcopy(model_template)   # pesos frescos por grupo
            try:
                # Entrenamiento
                model.fit(X_train, y_train_enc)
                y_test_pred = model.predict(X_test)

                # Decodificar predicciones a etiquetas originales
                if isinstance(y_test_pred.flat[0], (int, np.integer)):
                    try:
                        y_test_pred = le.inverse_transform(y_test_pred)
                    except Exception:
                        pass

                # Métricas
                acc = accuracy_score(y_test, y_test_pred)
                cm  = confusion_matrix(y_test, y_test_pred, labels=etiquetas_clase)
                f1  = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                # AUC-ROC (one-vs-rest para multiclase)
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba    = model.predict_proba(X_test)
                        y_test_bin = label_binarize(y_test_enc, classes=np.arange(N_CLASES))
                        if N_CLASES == 2:
                            auc = roc_auc_score(y_test_bin, y_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
                    else:
                        auc = -1
                except Exception:
                    auc = -1

                results_feat[model_name] = {
                    'accuracy': acc,
                    'f1':       f1,
                    'auc':      auc,
                    'cm':       cm,
                    'y_pred':   y_test_pred,
                    'X_test':   X_test,
                    'y_test':   y_test
                }

                # Actualizar mejor modelo global
                if acc > best_overall['accuracy']:
                    best_overall.update({
                        'accuracy': acc,
                        'model':    model_name,
                        'features': feat_name,
                        'y_pred':   y_test_pred,
                        'X_test':   X_test,
                        'y_test':   y_test
                    })

                auc_str = f"{auc:.2%}" if auc > 0 else "N/A"
                print(f"  {model_name:25} -> Accuracy: {acc:.2%} | F1: {f1:.2%} | AUC: {auc_str}")

                # Curva de aprendizaje del MLP
                if hasattr(model, 'loss_curve_'):
                    fig_lc, ax_lc = plt.subplots(figsize=(6, 4))
                    ax_lc.plot(model.loss_curve_, color='#4C72B0', linewidth=1.8)
                    ax_lc.set_title(f'Curva de Aprendizaje - Red Neuronal MLP\nGrupo: {feat_name}',
                                    fontsize=11, fontweight='bold')
                    ax_lc.set_xlabel('Épocas')
                    ax_lc.set_ylabel('Pérdida')
                    ax_lc.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR,
                                f'learning_curve_MLP_{feat_name}_{tipo_clase}.png'),
                                dpi=150, bbox_inches='tight')
                    plt.show()

            except Exception as e:
                print(f"  {model_name:25} -> ERROR: {str(e)}")
                results_feat[model_name] = None

        all_results[feat_name] = results_feat

    # --- 5. VISUALIZACIÓN DE RESULTADOS ---

    # 5.1 Tabla comparativa
    print("\n" + "="*80)
    print(f"RESUMEN DE RESULTADOS POR MODELO - {tipo_clase}")
    print("="*80)

    summary_data = []
    for feat_name, results in all_results.items():
        for model_name, model_results in results.items():
            if model_results is not None:
                summary_data.append({
                    'Grupo LDA':  feat_name,
                    'Modelo':     model_name,
                    'Accuracy':   model_results['accuracy'],
                    'F1-Score':   model_results['f1'],
                    'AUC-ROC':    model_results['auc']
                })

    df_summary = pd.DataFrame(summary_data)

    # Tablas pivote
    df_pivot_acc = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='Accuracy')
    df_pivot_f1  = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='F1-Score')
    df_pivot_auc = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='AUC-ROC')

    df_pivot_acc['Promedio'] = df_pivot_acc.mean(axis=1)
    df_pivot_acc = df_pivot_acc.sort_values('Promedio', ascending=False)

    df_pivot_f1['Promedio'] = df_pivot_f1.mean(axis=1)
    df_pivot_f1 = df_pivot_f1.sort_values('Promedio', ascending=False)

    df_pivot_auc['Promedio'] = df_pivot_auc.mean(axis=1)
    df_pivot_auc = df_pivot_auc.sort_values('Promedio', ascending=False)

    print("\nAccuracy por modelo y grupo LDA:")
    print("-" * 80)
    print(df_pivot_acc.round(4).to_string())

    print("\n" + "="*80)
    print("F1-Score por modelo y grupo LDA:")
    print("-" * 80)
    print(df_pivot_f1.round(4).to_string())

    print("\n" + "="*80)
    print("AUC-ROC por modelo y grupo LDA:")
    print("-" * 80)
    print(df_pivot_auc.replace(-1, np.nan).round(4).to_string())

    # 5.2 Mejores modelos por métrica
    print("\n" + "="*80)
    print(f"MEJORES MODELOS POR MÉTRICA - {tipo_clase}:")
    print("="*80)

    best_acc_model = df_pivot_acc['Promedio'].idxmax()
    best_acc_value = df_pivot_acc['Promedio'].max()
    print(f"\n1. MEJOR POR ACCURACY: {best_acc_model} ({best_acc_value:.2%})")

    best_f1_model = df_pivot_f1['Promedio'].idxmax()
    best_f1_value = df_pivot_f1['Promedio'].max()
    print(f"2. MEJOR POR F1-SCORE: {best_f1_model} ({best_f1_value:.2%})")

    auc_promedio = df_pivot_auc.replace(-1, np.nan)['Promedio']
    if auc_promedio.notna().any():
        best_auc_model = auc_promedio.idxmax()
        best_auc_value = auc_promedio.max()
        print(f"3. MEJOR POR AUC-ROC: {best_auc_model} ({best_auc_value:.2%})")
    else:
        best_auc_model = None
        best_auc_value = None
        print(f"3. MEJOR POR AUC-ROC: No disponible")

    # Guardar resultado global
    resultados_globales[tipo_clase] = {
        'best_accuracy': {'model': best_acc_model, 'value': best_acc_value},
        'best_f1':       {'model': best_f1_model,  'value': best_f1_value},
        'best_auc':      {'model': best_auc_model,  'value': best_auc_value}
    }

    # Guardar CSV resumen
    df_summary.to_csv(
        os.path.join(OUTPUT_DIR, f'resumen_clasificadores_{tipo_clase}.csv'),
        index=False
    )
    print(f"\n  💾 resumen_clasificadores_{tipo_clase}.csv guardado en {OUTPUT_DIR}")

    # --- 5.3 MAPAS DE CALOR ---
    grupos_keys = list(feature_sets.keys())
    model_names = df_pivot_acc.index.tolist()

    # Mapa de calor — Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(max(10, len(grupos_keys) * 1.5), 6))
    heatmap_acc_data = []
    for model_name in model_names:
        row = []
        for gname in grupos_keys:
            r = all_results.get(gname, {}).get(model_name)
            row.append(r['accuracy'] if r else 0)
        heatmap_acc_data.append(row)

    sns.heatmap(heatmap_acc_data,
                xticklabels=grupos_keys,
                yticklabels=model_names,
                annot=True, fmt='.2%', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'},
                ax=ax_acc, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    ax_acc.set_title(f'Mapa de Calor - Accuracy por Modelo y Grupo LDA - {tipo_clase}',
                     fontsize=14, fontweight='bold', pad=20)
    ax_acc.set_xlabel('Grupo LDA', fontsize=12)
    ax_acc.set_ylabel('Modelo',    fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_accuracy_{tipo_clase}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Mapa de calor — F1-Score
    fig_f1, ax_f1 = plt.subplots(figsize=(max(10, len(grupos_keys) * 1.5), 6))
    heatmap_f1_data = []
    for model_name in model_names:
        row = []
        for gname in grupos_keys:
            r = all_results.get(gname, {}).get(model_name)
            row.append(r['f1'] if r else 0)
        heatmap_f1_data.append(row)

    sns.heatmap(heatmap_f1_data,
                xticklabels=grupos_keys,
                yticklabels=model_names,
                annot=True, fmt='.2%', cmap='RdYlGn',
                cbar_kws={'label': 'F1-Score'},
                ax=ax_f1, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    ax_f1.set_title(f'Mapa de Calor - F1-Score por Modelo y Grupo LDA - {tipo_clase}',
                    fontsize=14, fontweight='bold', pad=20)
    ax_f1.set_xlabel('Grupo LDA', fontsize=12)
    ax_f1.set_ylabel('Modelo',    fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_f1_{tipo_clase}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Mapa de calor — AUC-ROC
    fig_auc, ax_auc = plt.subplots(figsize=(max(10, len(grupos_keys) * 1.5), 6))
    heatmap_auc_data = []
    for model_name in model_names:
        row = []
        for gname in grupos_keys:
            r = all_results.get(gname, {}).get(model_name)
            auc_val = r['auc'] if r else 0
            row.append(auc_val if auc_val > 0 else 0)
        heatmap_auc_data.append(row)

    sns.heatmap(heatmap_auc_data,
                xticklabels=grupos_keys,
                yticklabels=model_names,
                annot=True, fmt='.2%', cmap='RdYlGn',
                cbar_kws={'label': 'AUC-ROC'},
                ax=ax_auc, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    ax_auc.set_title(f'Mapa de Calor - AUC-ROC por Modelo y Grupo LDA - {tipo_clase}',
                     fontsize=14, fontweight='bold', pad=20)
    ax_auc.set_xlabel('Grupo LDA', fontsize=12)
    ax_auc.set_ylabel('Modelo',    fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_auc_{tipo_clase}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # --- 5.4 MATRICES DE CONFUSIÓN DEL MEJOR MODELO POR MÉTRICA ---
    print("\n" + "="*80)
    print(f"MATRICES DE CONFUSIÓN DEL MEJOR MODELO POR MÉTRICA - {tipo_clase}")
    print("="*80)

    best_models = [
        ('Accuracy', best_acc_model, 'accuracy'),
        ('F1-Score', best_f1_model,  'f1'),
        ('AUC-ROC',  best_auc_model, 'auc'),
    ]

    for metric_name, best_model, metric_col in best_models:
        if best_model is None:
            continue

        # Encontrar la mejor combinación de grupo para este modelo
        best_feat = None
        best_cm   = None
        best_val  = -1

        for feat_name, results in all_results.items():
            if best_model in results and results[best_model] is not None:
                val = results[best_model][metric_col]
                if val > best_val:
                    best_val  = val
                    best_feat = feat_name
                    best_cm   = results[best_model]['cm']

        if best_cm is not None:
            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=best_cm,
                display_labels=etiquetas_clase
            )
            disp.plot(ax=ax_cm, cmap='Blues')
            ax_cm.set_title(
                f'Matriz de Confusión - {best_model} ({metric_name}: {best_val:.2%})\n'
                f'Grupo LDA: {best_feat} | {tipo_clase}',
                fontsize=13, fontweight='bold', pad=20
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR,
                             f'confusion_{metric_name.replace("-","_")}_{tipo_clase}.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GLOBAL FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#"*80)
print("# RESUMEN GLOBAL DE TODOS LOS TIPOS DE CLASE")
print("#"*80)

for tipo, res in resultados_globales.items():
    print(f"\n  {tipo}:")
    print(f"    Mejor Accuracy : {res['best_accuracy']['model']}  ({res['best_accuracy']['value']:.2%})")
    print(f"    Mejor F1-Score : {res['best_f1']['model']}  ({res['best_f1']['value']:.2%})")
    if res['best_auc']['model']:
        print(f"    Mejor AUC-ROC  : {res['best_auc']['model']}  ({res['best_auc']['value']:.2%})")