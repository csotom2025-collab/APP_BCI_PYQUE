import copy
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             ConfusionMatrixDisplay, f1_score, roc_auc_score)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

DEFAULT_TIPOS_VALIDOS = ['Digit', 'Char', 'Comando']
DEFAULT_CARPETA_LDA = 'Resultados_LDA'
DEFAULT_CARPETA_GENERAL = 'LDA_General'
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_RANDOM_STATE = 42


def get_default_models(random_state: int = DEFAULT_RANDOM_STATE):
    return {
        'Regresión Logística': LogisticRegression(
            max_iter=1000, random_state=random_state,
            penalty='l2', solver='lbfgs', C=10.0**10
        ),
        'SVM Lineal': SVC(
            kernel='linear', probability=True, random_state=random_state
        ),
        'SVM RBF': SVC(
            kernel='rbf', probability=True, random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, eval_metric='mlogloss',
            verbosity=0, random_state=random_state
        ),
        'Red Neuronal MLP': MLPClassifier(
            max_iter=1000, hidden_layer_sizes=(1000,), random_state=random_state
        ),
    }


class LDAClassifierTrainer:
    """
    Entrena y evalúa clasificadores sobre features LDA generadas en carpetas.

    El módulo busca automáticamente subcarpetas dentro de:
      {usuario_base}/{tipo}/{carpeta_lda}/
    y también admite una carpeta global:
      {usuario_base}/{carpeta_general}/
    """

    def __init__(
        self,
        usuario_base: str,
        output_base: str | None = None,
        tipos_validos: list[str] | None = None,
        carpeta_lda: str = DEFAULT_CARPETA_LDA,
        carpeta_general: str = DEFAULT_CARPETA_GENERAL,
        train_size: float = DEFAULT_TRAIN_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        models: dict | None = None,
        verbose: bool = True,
        save_plots: bool = True,
    ):
        self.usuario_base = Path(usuario_base)
        self.output_base = Path(output_base) if output_base else self.usuario_base / 'Resultados_Clasificadores'
        self.tipos_validos = tipos_validos or DEFAULT_TIPOS_VALIDOS
        self.carpeta_lda = carpeta_lda
        self.carpeta_general = carpeta_general
        self.train_size = train_size
        self.random_state = random_state
        self.models = models or get_default_models(random_state=random_state)
        self.verbose = verbose
        self.save_plots = save_plots
        self.output_base.mkdir(parents=True, exist_ok=True)

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def discover_sources(self) -> dict[str, Path]:
        sources: dict[str, Path] = {}

        for tipo in self.tipos_validos:
            ruta = self.usuario_base / tipo / self.carpeta_lda
            if ruta.is_dir():
                sources[tipo] = ruta
            else:
                self._log(f"  ⚠️  Sin {self.carpeta_lda} para {tipo}: {ruta}")

        ruta_general = self.usuario_base / self.carpeta_general
        if ruta_general.is_dir():
            sources['LDA_General'] = ruta_general
        else:
            self._log(f"  ⚠️  Sin carpeta {self.carpeta_general} en: {ruta_general}")

        if not sources:
            raise FileNotFoundError(
                f"No se encontró ninguna fuente LDA en {self.usuario_base}.\n"
                f"Verifica que existan las carpetas {self.carpeta_lda} o {self.carpeta_general}."
            )

        return sources

    def load_feature_sets(self, ruta_grupos: Path) -> dict[str, pd.DataFrame]:
        ruta_grupos = Path(ruta_grupos)
        feature_sets: dict[str, pd.DataFrame] = {}

        subdirs = sorted([d for d in ruta_grupos.iterdir() if d.is_dir()])
        if not subdirs:
            csv_files = sorted([f for f in ruta_grupos.iterdir() if f.is_file() and f.suffix.lower() == '.csv'])
            if csv_files:
                subdirs = [ruta_grupos]

        for grupo_path in subdirs:
            csv_files = sorted([f for f in grupo_path.iterdir() if f.is_file() and f.suffix.lower() == '.csv'])
            grupo_nombre = grupo_path.name if grupo_path != ruta_grupos else ruta_grupos.name

            if not csv_files:
                self._log(f"  Sin CSV en {grupo_nombre}. Omitiendo grupo...")
                continue

            dfs = []
            for fichero in csv_files:
                try:
                    df = pd.read_csv(fichero)
                    if 'label' not in df.columns:
                        self._log(f"  Sin columna 'label': {fichero.name}. Omitiendo...")
                        continue
                    dfs.append(df)
                except Exception as exc:
                    self._log(f"  Error leyendo {fichero.name}: {exc}")

            if dfs:
                data_grupo = pd.concat(dfs, ignore_index=True)
                feature_sets[grupo_nombre] = data_grupo
                ld_cols = [c for c in data_grupo.columns if c.startswith('LD')]
                #self._log(f"  ✔ {grupo_nombre:<25} → {len(data_grupo):4} épocas | {len(ld_cols):2} componentes LDA")

        return feature_sets

    def _detect_labels(self, feature_sets: dict[str, pd.DataFrame]) -> list[str]:
        todas_etiquetas = set()
        for df in feature_sets.values():
            todas_etiquetas.update(df['label'].astype(str).unique())
        return sorted(todas_etiquetas)

    def _prepare_train_test(self, df: pd.DataFrame, etiquetas_clase: list[str]):
        ld_cols = [c for c in df.columns if c.startswith('LD')]
        X = df[ld_cols].astype(float).values
        y_raw = df['label'].astype(str).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=self.random_state,
                train_size=self.train_size, stratify=y_raw
            )
        except ValueError:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y_raw, random_state=self.random_state,
                train_size=self.train_size
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        le = LabelEncoder()
        le.fit(etiquetas_clase)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        return X_train, X_test, y_train_enc, y_test_enc, y_test, le, scaler

    def _evaluate_model(
        self,
        model_name: str,
        model_template,
        X_train,
        X_test,
        y_train_enc,
        y_test_enc,
        y_test,
        le: LabelEncoder,
        etiquetas_clase: list[str],
    ) -> dict | None:
        model = copy.deepcopy(model_template)
        try:
            model.fit(X_train, y_train_enc)
            y_pred = model.predict(X_test)

            if isinstance(y_pred.flat[0], (int, np.integer)):
                try:
                    y_pred = le.inverse_transform(y_pred)
                except Exception:
                    pass

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=etiquetas_clase)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            auc = -1
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(etiquetas_clase)))
                    auc = (roc_auc_score(y_test_bin, y_proba[:, 1])
                           if len(etiquetas_clase) == 2
                           else roc_auc_score(y_test_bin, y_proba, multi_class='ovr'))
                except Exception:
                    auc = -1

            return {
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
                'cm': cm,
                'y_pred': y_pred,
                'X_test': X_test,
                'y_test': y_test,
                'model': model,
            }
        except Exception as exc:
            self._log(f"  {model_name:25} -> ERROR: {exc}")
            return None

    def _plot_and_save(self, fig, path: Path):
        fig.savefig(path, dpi=150, bbox_inches='tight')
        if self.save_plots:
            self._log(f"  💾 {path}")
        if not self.verbose:
            plt.close(fig)
        else:
            plt.show()

    def evaluate_source(
        self,
        tipo_clase: str,
        ruta_grupos: Path,
        save_plots: bool | None = None,
        show_plots: bool = False,
    ) -> dict:
        save_plots = self.save_plots if save_plots is None else save_plots
        self._log(f"\n{'#'*80}")
        self._log(f"# PROCESANDO: {tipo_clase}  →  {ruta_grupos}")
        self._log(f"{'#'*80}\n")

        output_dir = self.output_base / tipo_clase
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_sets = self.load_feature_sets(ruta_grupos)
        if not feature_sets:
            self._log(f"  No se cargó ningún grupo LDA para {tipo_clase}. Omitiendo...")
            return {}

        etiquetas_clase = self._detect_labels(feature_sets)
        if not etiquetas_clase:
            self._log(f"  No se detectaron etiquetas para {tipo_clase}. Omitiendo...")
            return {}

        self._log(f"\n  Detectadas {len(etiquetas_clase)} clases: {etiquetas_clase}")

        all_results: dict[str, dict[str, dict | None]] = {}
        summary_rows: list[dict] = []

        for feat_name, data_grupo in feature_sets.items():
            self._log(f"\n{'='*60}")
            self._log(f"  GRUPO LDA: {feat_name}")
            self._log(f"{'='*60}")

            X_train, X_test, y_train_enc, y_test_enc, y_test, le, _ = self._prepare_train_test(
                data_grupo, etiquetas_clase
            )

            results_feat: dict[str, dict | None] = {}
            for model_name, model_template in self.models.items():
                result = self._evaluate_model(
                    model_name,
                    model_template,
                    X_train,
                    X_test,
                    y_train_enc,
                    y_test_enc,
                    y_test,
                    le,
                    etiquetas_clase,
                )
                results_feat[model_name] = result
                if result is not None:
                    auc_str = f"{result['auc']:.2%}" if result['auc'] >= 0 else 'N/A'
                    summary_rows.append({
                        'Grupo LDA': feat_name,
                        'Modelo': model_name,
                        'Accuracy': result['accuracy'],
                        'F1-Score': result['f1'],
                        'AUC-ROC': result['auc'],
                    })
                    self._log(
                        f"  {model_name:25} -> Accuracy: {result['accuracy']:.2%} | "
                        f"F1: {result['f1']:.2%} | AUC: {auc_str}"
                    )
            all_results[feat_name] = results_feat

        if not summary_rows:
            self._log(f"  No se generaron resultados válidos para {tipo_clase}.")
            return {}

        df_summary = pd.DataFrame(summary_rows)
        self._save_summary_csv(df_summary, output_dir, tipo_clase)
        self._plot_heatmaps(df_summary, all_results, output_dir, tipo_clase, save_plots, show_plots)
        self._plot_confusion_matrices(all_results, etiquetas_clase, output_dir, tipo_clase, save_plots, show_plots)

        best_models = self._best_models_from_summary(df_summary)

        return {
            'tipo_clase': tipo_clase,
            'feature_sets': feature_sets,
            'etiquetas_clase': etiquetas_clase,
            'all_results': all_results,
            'summary': df_summary,
            'best_models': best_models,
            'output_dir': output_dir,
        }

    def evaluate_all(self, save_plots: bool | None = None, show_plots: bool = False) -> dict[str, dict]:
        save_plots = self.save_plots if save_plots is None else save_plots
        sources = self.discover_sources()
        results: dict[str, dict] = {}
        for tipo_clase, ruta_grupos in sources.items():
            source_results = self.evaluate_source(
                tipo_clase,
                ruta_grupos,
                save_plots=save_plots,
                show_plots=show_plots,
            )
            if source_results:
                results[tipo_clase] = source_results

        # if results:
        #     self._save_global_summary(results)

        return results

    def _save_summary_csv(self, df_summary: pd.DataFrame, output_dir: Path, tipo_clase: str):
        output_path = output_dir / f'resumen_clasificadores_{tipo_clase}.csv'
        df_summary.to_csv(output_path, index=False)
        self._log(f"  💾 {output_path}")

    def _best_models_from_summary(self, df_summary: pd.DataFrame) -> dict[str, dict]:
        best_models = {}
        for metric in ['Accuracy', 'F1-Score', 'AUC-ROC']:
            pivot = df_summary.pivot(index='Modelo', columns='Grupo LDA', values=metric)
            pivot['Promedio'] = pivot.mean(axis=1)
            pivot = pivot.sort_values('Promedio', ascending=False)
            best_models[metric] = {
                'model': pivot.index[0],
                'value': pivot['Promedio'].iloc[0],
            }
        return best_models

    def _plot_heatmaps(
        self,
        df_summary: pd.DataFrame,
        all_results: dict[str, dict[str, dict | None]],
        output_dir: Path,
        tipo_clase: str,
        save_plots: bool,
        show_plots: bool,
    ):
        groups = sorted(all_results.keys())
        model_names = sorted({name for results in all_results.values() for name in results.keys()})
        hm_w = max(10, len(groups) * 1.5)

        for metric_key, metric_label in [('accuracy', 'Accuracy'), ('f1', 'F1-Score'), ('auc', 'AUC-ROC')]:
            heatmap_data = []
            for model_name in model_names:
                row = []
                for group in groups:
                    result = all_results.get(group, {}).get(model_name)
                    val = result[metric_key] if result else 0
                    row.append(val if val > 0 else 0)
                heatmap_data.append(row)

            fig, ax = plt.subplots(figsize=(hm_w, 6))
            sns.heatmap(
                heatmap_data,
                xticklabels=groups,
                yticklabels=model_names,
                annot=True,
                fmt='.2%',
                cmap='RdYlGn',
                cbar_kws={'label': metric_label},
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                f'Mapa de Calor - {metric_label} por Modelo y Grupo LDA\n{tipo_clase}',
                fontsize=14,
                fontweight='bold',
                pad=20,
            )
            ax.set_xlabel('Grupo LDA', fontsize=12)
            ax.set_ylabel('Modelo', fontsize=12)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            path = output_dir / f'heatmap_{metric_key}_{tipo_clase}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            if not show_plots:
                plt.close(fig)
            if save_plots:
                self._log(f"  💾 {path}")
            elif show_plots:
                plt.show()

    def _plot_confusion_matrices(
        self,
        all_results: dict[str, dict[str, dict | None]],
        etiquetas_clase: list[str],
        output_dir: Path,
        tipo_clase: str,
        save_plots: bool,
        show_plots: bool,
    ):
        for group_name, results in all_results.items():
            for model_name, result in results.items():
                if not result:
                    continue
                cm = result['cm']
                fig, ax = plt.subplots(figsize=(max(10, len(etiquetas_clase) * 0.5), 7))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas_clase)
                disp.plot(ax=ax, cmap='Blues', colorbar=True)
                if len(etiquetas_clase) > 15:
                    ax.tick_params(axis='both', labelsize=7)
                    for text in ax.texts:
                        text.set_fontsize(6)
                ax.set_title(
                    f'Matriz de Confusión - {model_name}\nGrupo: {group_name} | {tipo_clase}',
                    fontsize=12,
                    fontweight='bold',
                    pad=15,
                )
                plt.tight_layout()
                safe_model_name = model_name.replace(' ', '_')
                safe_group_name = str(group_name).replace(' ', '_')
                path = output_dir / f'confusion_{safe_model_name}_{safe_group_name}.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                if not show_plots:
                    plt.close(fig)
                if save_plots:
                    self._log(f"  💾 {path}")
                elif show_plots:
                    plt.show()

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = 'Regresión Logística',
        train_size: float | None = None,
    ) -> dict:
        feature_sets = {'all': df}
        etiquetas_clase = self._detect_labels(feature_sets)
        if not etiquetas_clase:
            raise ValueError('No se detectaron etiquetas en el DataFrame provisto.')

        if train_size is not None:
            self.train_size = train_size

        X_train, X_test, y_train_enc, y_test_enc, y_test, le, scaler = self._prepare_train_test(
            df, etiquetas_clase
        )

        model_template = self.models.get(model_name)
        if model_template is None:
            raise ValueError(f'Modelo desconocido: {model_name}')

        result = self._evaluate_model(
            model_name,
            model_template,
            X_train,
            X_test,
            y_train_enc,
            y_test_enc,
            y_test,
            le,
            etiquetas_clase,
        )

        if result is None:
            raise RuntimeError(f'No se pudo entrenar el modelo {model_name}.')

        return {
            'model': result['model'],
            'scaler': scaler,
            'label_encoder': le,
            'metrics': {k: result[k] for k in ['accuracy', 'f1', 'auc']},
            'y_test': y_test,
            'y_pred': result['y_pred'],
        }

    def train_model_on_source(
        self,
        ruta_grupos: Path,
        group_name: str | None = None,
        model_name: str = 'Regresión Logística',
        train_size: float | None = None,
    ) -> dict:
        feature_sets = self.load_feature_sets(ruta_grupos)
        if not feature_sets:
            raise FileNotFoundError(f'No se encontraron grupos LDA en {ruta_grupos}.')

        if group_name is None:
            df = pd.concat(feature_sets.values(), ignore_index=True)
        else:
            if group_name not in feature_sets:
                raise ValueError(f'Grupo no encontrado: {group_name}')
            df = feature_sets[group_name]

        return self.train_model(df, model_name=model_name, train_size=train_size)


if __name__ == '__main__':
    trainer = LDAClassifierTrainer(
        usuario_base=r'D:/EEG_Python/results/User94',
        output_base=r'D:/EEG_Python/results/User94/Resultados_Clasificadores',
        verbose=True,
        save_plots=True,
    )

    modelos = trainer.evaluate_all(save_plots=True, show_plots=False)
    if modelos:
        print('\nEvaluación completa finalizada.')
        for tipo, resultado in modelos.items():
            mejor = resultado.get('best_models')
            if mejor:
                print(f"{tipo}: {mejor}")
