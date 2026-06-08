import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ══════════════════════════════════════════════════════════════════════════════
# --- 1. CONFIGURACIÓN ---
# ══════════════════════════════════════════════════════════════════════════════
# Carpeta raíz del usuario — el script detecta automáticamente:
#   - Tipos con LDA:   {USUARIO_BASE}/{Tipo}/Resultados_LDA/
#   - LDA General:     {USUARIO_BASE}/LDA_General/
#
# Ejemplo de estructura reconocida:
#   Usermar/
#   ├── Digit/Resultados_LDA/Estadisticas/   ← tipo Digit
#   ├── Char/Resultados_LDA/Wavelets/         ← tipo Char
#   ├── Comando/Resultados_LDA/TODAS/         ← tipo Comando
#   └── LDA_General/Estadisticas/             ← todas las clases juntas

USUARIO_BASE    = r'./results/UserArcane_LDA'
OUTPUT_BASE     = r'./results/UserArcane_LDA/Resultados_Clasificadores'
CARPETA_LDA     = 'Resultados_LDA'
CARPETA_GENERAL = 'LDA_General'
TIPOS_VALIDOS   = ['Digit', 'Char', 'Comando']

randNum    = 42
TRAIN_SIZE = 0.8

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# --- 2. DESCUBRIMIENTO AUTOMÁTICO DE FUENTES LDA ---
# Construye dict: { nombre_fuente -> ruta_con_grupos_lda }
# ══════════════════════════════════════════════════════════════════════════════
fuentes_lda = {}

# 2a. Tipos individuales (Digit, Char, Comando, ...)
for tipo in TIPOS_VALIDOS:
    ruta = os.path.join(USUARIO_BASE, tipo, CARPETA_LDA)
    if os.path.isdir(ruta):
        fuentes_lda[tipo] = ruta
    else:
        print(f"  ⚠️  Sin {CARPETA_LDA} para {tipo}: {ruta}")

# 2b. LDA_General (grupos van directamente dentro, sin subcarpeta de tipo)
ruta_general = os.path.join(USUARIO_BASE, CARPETA_GENERAL)
if os.path.isdir(ruta_general):
    fuentes_lda['LDA_General'] = ruta_general
else:
    print(f"  ⚠️  Sin carpeta {CARPETA_GENERAL} en: {ruta_general}")

if not fuentes_lda:
    raise FileNotFoundError(
        f"No se encontró ninguna fuente LDA en {USUARIO_BASE}.\n"
        f"Verifica que existan las carpetas {CARPETA_LDA} o {CARPETA_GENERAL}."
    )

print(f"\n{'#'*80}")
print(f"# CLASIFICADORES SOBRE FEATURES LDA")
print(f"{'#'*80}")
print(f"\n  Usuario base : {USUARIO_BASE}")
print(f"  Salida       : {OUTPUT_BASE}")
print(f"  Fuentes LDA  : {list(fuentes_lda.keys())}\n")

# Diccionario global para almacenar resultados
resultados_globales = {}

# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL: PROCESAR CADA FUENTE LDA
# ══════════════════════════════════════════════════════════════════════════════
for tipo_clase, ruta_grupos in fuentes_lda.items():

    print(f"\n{'#'*80}")
    print(f"# PROCESANDO: {tipo_clase}  →  {ruta_grupos}")
    print(f"{'#'*80}\n")

    output_dir = os.path.join(OUTPUT_BASE, tipo_clase)
    os.makedirs(output_dir, exist_ok=True)

    # --- CARGA DE GRUPOS LDA ---
    grupos_disponibles = sorted([
        d for d in os.listdir(ruta_grupos)
        if os.path.isdir(os.path.join(ruta_grupos, d))
    ])

    if not grupos_disponibles:
        print(f"  Sin subcarpetas de grupos en {ruta_grupos}. Omitiendo...")
        continue

    print(f"  Grupos LDA detectados: {grupos_disponibles}\n")

    feature_sets = {}
    for grupo in grupos_disponibles:
        grupo_path = os.path.join(ruta_grupos, grupo)
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
                    print(f"  Sin columna 'label': {fname}. Omitiendo...")
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"  Error leyendo {fname}: {e}")

        if dfs:
            data_grupo = pd.concat(dfs, ignore_index=True)
            feature_sets[grupo] = data_grupo
            ld_cols = [c for c in data_grupo.columns if c.startswith('LD')]
            print(f"  ✔ {grupo:<25} → {len(data_grupo):4} épocas | {len(ld_cols):2} componentes LDA")

    if not feature_sets:
        print(f"  No se cargó ningún grupo LDA para {tipo_clase}. Omitiendo...")
        continue

    # Detectar etiquetas de clase automáticamente
    todas_etiquetas = set()
    for df in feature_sets.values():
        todas_etiquetas.update(df['label'].astype(str).unique())
    etiquetas_clase = sorted(todas_etiquetas)
    N_CLASES        = len(etiquetas_clase)
    fig_cm_size     = max(10, N_CLASES * 0.8)

    print(f"\n  Detectadas {N_CLASES} clases: {etiquetas_clase}")

    # --- DEFINICIÓN DE MODELOS ---
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

    # --- EVALUACIÓN DE MODELOS ---
    print("\n" + "="*80)
    print(f"  EVALUACIÓN DE MÚLTIPLES MODELOS - {tipo_clase}")
    print("="*80)

    all_results  = {}
    best_overall = {
        'accuracy': -1, 'f1': -1, 'auc': -1,
        'model': None, 'features': None,
        'y_pred': None, 'X_test': None, 'y_test': None
    }

    for feat_name, data_grupo in feature_sets.items():

        print(f"\n{'='*60}")
        print(f"  GRUPO LDA: {feat_name}")
        print(f"{'='*60}")

        ld_cols = [c for c in data_grupo.columns if c.startswith('LD')]
        X       = data_grupo[ld_cols].values.astype(float)
        y_raw   = data_grupo['label'].astype(str).values
        X       = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=randNum,
                train_size=TRAIN_SIZE, stratify=y_raw
            )
        except ValueError:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=randNum, train_size=TRAIN_SIZE
            )

        scaler      = StandardScaler()
        X_train     = scaler.fit_transform(X_train_raw)
        X_test      = scaler.transform(X_test_raw)

        le          = LabelEncoder()
        le.fit(etiquetas_clase)
        y_train_enc = le.transform(y_train)
        y_test_enc  = le.transform(y_test)

        results_feat = {}

        for model_name, model_template in modelos.items():
            model = copy.deepcopy(model_template)
            try:
                model.fit(X_train, y_train_enc)
                y_test_pred = model.predict(X_test)

                if isinstance(y_test_pred.flat[0], (int, np.integer)):
                    try:
                        y_test_pred = le.inverse_transform(y_test_pred)
                    except Exception:
                        pass

                acc = accuracy_score(y_test, y_test_pred)
                cm  = confusion_matrix(y_test, y_test_pred, labels=etiquetas_clase)
                f1  = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba    = model.predict_proba(X_test)
                        y_test_bin = label_binarize(y_test_enc, classes=np.arange(N_CLASES))
                        auc = (roc_auc_score(y_test_bin, y_proba[:, 1])
                               if N_CLASES == 2
                               else roc_auc_score(y_test_bin, y_proba, multi_class='ovr'))
                    else:
                        auc = -1
                except Exception:
                    auc = -1

                results_feat[model_name] = {
                    'accuracy': acc, 'f1': f1, 'auc': auc,
                    'cm': cm, 'y_pred': y_test_pred,
                    'X_test': X_test, 'y_test': y_test
                }

                if acc > best_overall['accuracy']:
                    best_overall.update({
                        'accuracy': acc, 'model': model_name,
                        'features': feat_name,
                        'y_pred': y_test_pred, 'X_test': X_test, 'y_test': y_test
                    })

                auc_str = f"{auc:.2%}" if auc > 0 else "N/A"
                print(f"  {model_name:25} -> Accuracy: {acc:.2%} | F1: {f1:.2%} | AUC: {auc_str}")

                # Curva de aprendizaje MLP
                if hasattr(model, 'loss_curve_'):
                    fig_lc, ax_lc = plt.subplots(figsize=(6, 4))
                    ax_lc.plot(model.loss_curve_, color='#4C72B0', linewidth=1.8)
                    ax_lc.set_title(f'Curva de Aprendizaje - Red Neuronal MLP\n'
                                    f'Grupo: {feat_name} | {tipo_clase}',
                                    fontsize=11, fontweight='bold')
                    ax_lc.set_xlabel('Épocas')
                    ax_lc.set_ylabel('Pérdida')
                    ax_lc.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir,
                                f'learning_curve_MLP_{feat_name}.png'),
                                dpi=150, bbox_inches='tight')
                    plt.show()

            except Exception as e:
                print(f"  {model_name:25} -> ERROR: {str(e)}")
                results_feat[model_name] = None

        all_results[feat_name] = results_feat

    # --- VISUALIZACIÓN ---

    # Cantidad de datos por clase
    print("\n" + "="*80)
    print("  Cantidad de datos usados para train y test por clase:")
    print("-" * 80)
    for feat_name, data_grupo in feature_sets.items():
        y_counts = data_grupo['label'].value_counts()
        print(f"\n  Grupo LDA: {feat_name}")
        for label in etiquetas_clase:
            count   = y_counts.get(label, 0)
            n_train = int(count * TRAIN_SIZE)
            n_test  = count - n_train
            print(f"    Clase '{label}': {count} muestras  "
                  f"(~{n_train} train / ~{n_test} test)")

    # Tabla comparativa
    print("\n" + "="*80)
    print(f"  RESUMEN DE RESULTADOS POR MODELO - {tipo_clase}")
    print("="*80)

    summary_data = []
    for feat_name, results in all_results.items():
        for model_name, model_results in results.items():
            if model_results is not None:
                summary_data.append({
                    'Grupo LDA': feat_name,
                    'Modelo':    model_name,
                    'Accuracy':  model_results['accuracy'],
                    'F1-Score':  model_results['f1'],
                    'AUC-ROC':   model_results['auc']
                })

    df_summary = pd.DataFrame(summary_data)

    df_pivot_acc = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='Accuracy')
    df_pivot_f1  = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='F1-Score')
    df_pivot_auc = df_summary.pivot(index='Modelo', columns='Grupo LDA', values='AUC-ROC')

    df_pivot_acc['Promedio'] = df_pivot_acc.mean(axis=1)
    df_pivot_acc = df_pivot_acc.sort_values('Promedio', ascending=False)
    df_pivot_f1['Promedio']  = df_pivot_f1.mean(axis=1)
    df_pivot_f1  = df_pivot_f1.sort_values('Promedio', ascending=False)
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

    # Mejores modelos
    print("\n" + "="*80)
    print(f"  MEJORES MODELOS POR MÉTRICA - {tipo_clase}:")
    print("="*80)

    best_acc_model = df_pivot_acc['Promedio'].idxmax()
    best_acc_value = df_pivot_acc['Promedio'].max()
    print(f"\n  1. MEJOR POR ACCURACY: {best_acc_model} ({best_acc_value:.2%})")

    best_f1_model = df_pivot_f1['Promedio'].idxmax()
    best_f1_value = df_pivot_f1['Promedio'].max()
    print(f"  2. MEJOR POR F1-SCORE: {best_f1_model} ({best_f1_value:.2%})")

    auc_promedio = df_pivot_auc.replace(-1, np.nan)['Promedio']
    if auc_promedio.notna().any():
        best_auc_model = auc_promedio.idxmax()
        best_auc_value = auc_promedio.max()
        print(f"  3. MEJOR POR AUC-ROC: {best_auc_model} ({best_auc_value:.2%})")
    else:
        best_auc_model = None
        best_auc_value = None
        print(f"  3. MEJOR POR AUC-ROC: No disponible")

    resultados_globales[tipo_clase] = {
        'best_accuracy': {'model': best_acc_model, 'value': best_acc_value},
        'best_f1':       {'model': best_f1_model,  'value': best_f1_value},
        'best_auc':      {'model': best_auc_model,  'value': best_auc_value}
    }

    df_summary.to_csv(
        os.path.join(output_dir, f'resumen_clasificadores_{tipo_clase}.csv'),
        index=False
    )
    print(f"\n  💾 resumen_clasificadores_{tipo_clase}.csv  →  {output_dir}")

    # Mapas de calor
    grupos_keys = list(feature_sets.keys())
    model_names = df_pivot_acc.index.tolist()
    hm_w        = max(10, len(grupos_keys) * 1.5)

    for metric_key, metric_label in [
        ('accuracy', 'Accuracy'),
        ('f1',       'F1-Score'),
        ('auc',      'AUC-ROC'),
    ]:
        heatmap_data = []
        for mname in model_names:
            row = []
            for gname in grupos_keys:
                r   = all_results.get(gname, {}).get(mname)
                val = r[metric_key] if r else 0
                row.append(val if val > 0 else 0)
            heatmap_data.append(row)

        fig_hm, ax_hm = plt.subplots(figsize=(hm_w, 6))
        sns.heatmap(heatmap_data,
                    xticklabels=grupos_keys,
                    yticklabels=model_names,
                    annot=True, fmt='.2%', cmap='RdYlGn',
                    cbar_kws={'label': metric_label},
                    ax=ax_hm, linewidths=0.5, linecolor='gray',
                    vmin=0, vmax=1)
        ax_hm.set_title(f'Mapa de Calor - {metric_label} por Modelo y Grupo LDA\n'
                        f'{tipo_clase}',
                        fontsize=14, fontweight='bold', pad=20)
        ax_hm.set_xlabel('Grupo LDA', fontsize=12)
        ax_hm.set_ylabel('Modelo',    fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f'heatmap_{metric_key}_{tipo_clase}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

    # Matrices de confusión del mejor modelo por métrica
    print("\n" + "="*80)
    print(f"  MATRICES DE CONFUSIÓN DEL MEJOR MODELO POR MÉTRICA - {tipo_clase}")
    print("="*80)

    for metric_name, best_model, metric_col in [
        ('Accuracy', best_acc_model, 'accuracy'),
        ('F1-Score', best_f1_model,  'f1'),
        ('AUC-ROC',  best_auc_model, 'auc'),
    ]:
        if best_model is None:
            continue

        best_feat, best_cm, best_val = None, None, -1
        for feat_name, results in all_results.items():
            if best_model in results and results[best_model] is not None:
                val = results[best_model][metric_col]
                if val > best_val:
                    best_val  = val
                    best_feat = feat_name
                    best_cm   = results[best_model]['cm']

        if best_cm is not None:
            fig_cm, ax_cm = plt.subplots(figsize=(fig_cm_size, fig_cm_size * 0.85))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=best_cm,
                display_labels=etiquetas_clase
            )
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=True)
            if N_CLASES > 15:
                ax_cm.tick_params(axis='both', labelsize=7)
                for text in ax_cm.texts:
                    text.set_fontsize(6)
            ax_cm.set_title(
                f'Matriz de Confusión - {best_model}\n'
                f'({metric_name}: {best_val:.2%}) | Grupo: {best_feat} | {tipo_clase}',
                fontsize=12, fontweight='bold', pad=15
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir,
                             f'confusion_{metric_name.replace("-","_")}_{tipo_clase}.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN GLOBAL FINAL — comparación entre todas las fuentes LDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "#"*80)
print("# RESUMEN GLOBAL DE TODAS LAS FUENTES LDA")
print("#"*80)

resumen_rows = []
for tipo, res in resultados_globales.items():
    print(f"\n  {tipo}:")
    print(f"    Mejor Accuracy : {res['best_accuracy']['model']:<25}  ({res['best_accuracy']['value']:.2%})")
    print(f"    Mejor F1-Score : {res['best_f1']['model']:<25}  ({res['best_f1']['value']:.2%})")
    if res['best_auc']['model']:
        print(f"    Mejor AUC-ROC  : {res['best_auc']['model']:<25}  ({res['best_auc']['value']:.2%})")

    resumen_rows.append({
        'Fuente':         tipo,
        'Mejor_Accuracy': res['best_accuracy']['value'],
        'Modelo_Acc':     res['best_accuracy']['model'],
        'Mejor_F1':       res['best_f1']['value'],
        'Modelo_F1':      res['best_f1']['model'],
        'Mejor_AUC':      res['best_auc']['value'] if res['best_auc']['model'] else np.nan,
        'Modelo_AUC':     res['best_auc']['model'] or 'N/A',
    })

# CSV resumen global
df_resumen_global = pd.DataFrame(resumen_rows)
out_global = os.path.join(OUTPUT_BASE, 'resumen_global_clasificadores.csv')
df_resumen_global.to_csv(out_global, index=False)
print(f"\n  💾 Resumen global: {out_global}")

# Gráfica comparativa entre fuentes (Accuracy y F1 del mejor modelo por fuente)
if resumen_rows:
    fig_comp, ax_comp = plt.subplots(figsize=(max(8, len(resumen_rows) * 2), 5))
    fuentes = [r['Fuente']         for r in resumen_rows]
    accs    = [r['Mejor_Accuracy'] for r in resumen_rows]
    f1s     = [r['Mejor_F1']       for r in resumen_rows]
    x       = np.arange(len(fuentes))
    w       = 0.35

    bars1 = ax_comp.bar(x - w/2, accs, w, label='Mejor Accuracy',
                        color='#4C72B0', edgecolor='white')
    bars2 = ax_comp.bar(x + w/2, f1s,  w, label='Mejor F1-Score',
                        color='#DD8452', edgecolor='white')

    for bar in list(bars1) + list(bars2):
        ax_comp.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f'{bar.get_height():.2%}',
                     ha='center', va='bottom', fontsize=9)

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(fuentes, fontsize=11)
    ax_comp.set_ylim(0, 1.12)
    ax_comp.set_ylabel('Métrica', fontsize=12)
    ax_comp.set_title('Comparación de Mejores Resultados por Fuente LDA\n'
                      '(Digit / Char / Comando / LDA_General)',
                      fontsize=13, fontweight='bold')
    ax_comp.legend(fontsize=10)
    ax_comp.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_comp.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out_comp = os.path.join(OUTPUT_BASE, 'comparacion_fuentes_lda.png')
    plt.savefig(out_comp, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  📊 comparacion_fuentes_lda.png  →  {OUTPUT_BASE}")