
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __package__:
    from .DivisorTiempos import SeparacionTiempos
    from .GetCaracteristiacas import FeatureExtractor
    from .clasificador_lda import LDAClassifierTrainer, get_default_models
    from .optimizacion_lda import (
        discover_structure,
        procesar_lda_general,
        procesar_lda_caracteristicas,
        procesar_trabajo,
    )
else:
    from models.DivisorTiempos import SeparacionTiempos
    from models.GetCaracteristiacas import FeatureExtractor
    from models.clasificador_lda import LDAClassifierTrainer, get_default_models
    from models.optimizacion_lda import (
        discover_structure,
        procesar_lda_general,
        procesar_lda_caracteristicas,
        procesar_trabajo,
    )

class PipelineCompletoLDA:
    def __init__(self,base_output_dir='results'):
        self.base_output_dir = base_output_dir
        self.separador = SeparacionTiempos(sampling_rate=128)
        self.featureExtractor = FeatureExtractor(fs=128)


    def separar_archivos_csv(self, usuario):
        self.separador = SeparacionTiempos(sampling_rate=128)
        
        # Procesar un archivo individual
        print("="*70)
        print("SEPARADOR DE TRIALS P300 - EEG")
        print("="*70 + "\n")
        letras=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        lista_carpetas  =os.listdir(f'captures/{usuario}/')


        for carpeta in lista_carpetas:
            tpComando= carpeta
            print(f"carpetea {tpComando}")
            archivos = os.listdir(f'captures/{usuario}/{tpComando}/')
            archivos_csv = [archivo for archivo in archivos if archivo.endswith('.csv')]
            #tpComandoOutput = 'Char' if tpComando == 'Letters' else 'Digit'
            for archivo in archivos_csv:
                resultado = self.separar_archivo_por_ruta(f'captures/{usuario}/{tpComando}/{archivo}',usuario)
            
    def separar_archivo_por_ruta(self, ruta_archivo,usuario):
        resultado = self.separador.procesar_archivo(
            ruta_archivo,
            carpeta_salida_base=self.base_output_dir + f'/{usuario}',
        )
        if resultado['exito']:
            #print(f"\n✓ Procesamiento exitoso:")
            return resultado['archivo_post']
        else:
            print(f"\n✗ Error: {resultado['error']}")
            return None
    def obtener_caracteristicas_usuario(self,usuario):
        self.featureExtractor = FeatureExtractor(fs=128)
        subcarpetas = os.listdir(self.base_output_dir+f'/{usuario}/')
        print(f"Subcarpetas encontradas para el usuario {usuario}: {subcarpetas}")
        for subcarpeta in subcarpetas:
            if subcarpeta == 'features' or subcarpeta=='Resultados_LDA' or subcarpeta=='LDA_General' or subcarpeta=='Resultados_Clasificadores':
                continue
            if not os.path.exists(f"{self.base_output_dir}/{usuario}/{subcarpeta}/features/"):
                os.makedirs(f"{self.base_output_dir}/{usuario}/{subcarpeta}/features")
            print(f"Procesando subcarpeta: {subcarpeta}")
            archivos = os.listdir(self.base_output_dir+f'/{usuario}/{subcarpeta}/Separados/')
            for archivo in archivos:
                if not archivo.endswith('post_estimulo.csv'):
                    continue
                letra = archivo.split("_")[1]
                trial = archivo.split("_")[2]
                ruta_archivo = os.path.join(self.base_output_dir+f'/{usuario}/{subcarpeta}/Separados/', archivo)
                self.extraer_caracteristicas_file(ruta_archivo,usuario,subcarpeta,letra,trial)
                # Aquí puedes almacenar o procesar las características extraídas según tus necesidades


        return {
            'edad': 30,
            'genero': 'masculino',
            'experiencia': 'intermedia'
        }
    def extraer_caracteristicas_file(self, ruta_archivo,usuario,subcarpeta="Unknown",letra='?',trial='?',save_output=True):
        try:
            print(f"Extrayendo características de: {ruta_archivo}")
            data = pd.read_csv(ruta_archivo)
            #channel_names = ["Oz", "Po7", "Po4", "Po3", "P4", "P3", "Po8", "Pz", "Fz", "F2", "F3", "F4", "AF3", "Cz", "AF4", "F1" ]
            #canales Epoc
            channel_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
            #channel_names = ["F3", "F4","AF3", "F1","P4", "P3","Po4", "Po3"]
            signals = data.values.T  # Transponer para tener shape (n_channels, n_samples)

            features_df = self.featureExtractor.extract_features(signals, channel_names=channel_names, available_channel_names=channel_names,window_size=50,overlap=0)
            # print(f"Caracteristicas extraidas para {archivo}:")
            # print(f"tamanio de archivo {data.shape} len featrues{len(features_df)}")
            #print(features_df.head())
            # Guardar las caracteristicas en un nuevo archivo CSV
            if save_output:
                output_path = f"{self.base_output_dir}/{usuario}/{subcarpeta}/features/features_{letra}_{trial}.csv"
                features_df.to_csv(output_path, index=False)
            return features_df
        except Exception as e:
            print(f"Error procesando {ruta_archivo}: {e}")
            return None

    def optimizacion_lda(self,user):
        args ={
            'base': f'{self.base_output_dir}',
            'usuarios': [user],
            'tipos': ['Digit','Char','Comando'],
            'grupos': ['Estadisticas' ,'Wavelets',"Frecuencias_Abs","Frecuencias_Rel","Frecuencias_Est","Frecuencias_Todas","TODAS"]
        }

        print(f"\n{'#'*62}")
        print(f"#  OPTIMIZACIÓN LDA — MULTI-USUARIO / MULTI-TIPO / N CLASES")
        print(f"{'#'*62}")
        print(f"\n  Base       : {args['base']}")
        print(f"  Usuarios   : {args['usuarios'] }")
        print(f"  Tipos      : {args['tipos']} ")
        print(f"  Grupos feat: {args['grupos']   }\n")
        # Descubrir estructura
        trabajos = discover_structure(
            args['base'],
            usuarios_filter=args['usuarios'],
            tipos_filter=args['tipos'] ,
        )
        
        if not trabajos:
            print("❌ No se encontró ninguna combinación usuario/tipo con datos.")
            return

        print(f"\n  Trabajos encontrados: {len(trabajos)}")
        for t in trabajos:
            print(f"    • {t['usuario']:<15} / {t['tipo']:<10}  ({t['csv_count']} CSV)")

        # Procesar cada trabajo (LDA por tipo)
        resumen_global = []
        for trabajo in trabajos:
            res = procesar_trabajo(trabajo, grupos_filter=args['grupos'])
            if res:
                for r in res:
                    r["usuario"] = trabajo["usuario"]
                    r["tipo"]    = trabajo["tipo"]
                    resumen_global.append(r)

        # ── LDA GENERAL por usuario (todas las clases de todos los tipos juntas) ──
        usuarios_unicos = sorted(set(t["usuario"] for t in trabajos))
        for usuario in usuarios_unicos:
            trabajos_usuario = [t for t in trabajos if t["usuario"] == usuario]
            res_gen = procesar_lda_general(
                usuario, trabajos_usuario, args['base'],
                grupos_filter=args['grupos']
            )
            if res_gen:
                for r in res_gen:
                    r["usuario"] = usuario
                    r["tipo"]    = "LDA_General"
                    resumen_global.append(r)

        # Resumen global
        if resumen_global:
            print(f"\n{'#'*62}")
            print(f"#  RESUMEN GLOBAL")
            print(f"{'#'*62}")
            df_global = pd.DataFrame(resumen_global)
            cols_show = ["usuario","tipo","grupo","n_clases","n_comp_lda",
                        "var_acum","cv_acc","cv_std"]
            print(df_global[[c for c in cols_show if c in df_global.columns]]
                .sort_values(["usuario","tipo","cv_acc"], ascending=[True,True,False])
                .to_string(index=False))

            out_global = os.path.join(args['base'], "resumen_lda_global.csv")
            df_global.to_csv(out_global, index=False)
            print(f"\n  💾 Resumen global: {out_global}")

        print(f"\n{'#'*62}")
        print(f"#  FINALIZADO")
        print(f"{'#'*62}\n")
    def optimizar_lda_caracteristicas(self, usuario, caracteristicas, label_col='label',
                                      grupos=['Estadisticas', 'Wavelets', 'Frecuencias_Abs',
                                              'Frecuencias_Rel', 'Frecuencias_Est',
                                              'Frecuencias_Todas', 'TODAS']):
        """Optimiza LDA sobre un DataFrame de características ya extraídas.

        El DataFrame debe contener una columna de etiquetas con el nombre especificado en label_col.
        Esta función no busca carpetas en disco; recibe directamente las características en memoria.
        """
        if not isinstance(caracteristicas, pd.DataFrame):
            raise ValueError("Se debe pasar un DataFrame de características.")
        if label_col not in caracteristicas.columns:
            raise ValueError(f"No se encontró la columna de etiquetas '{label_col}' en el DataFrame.")

        output_path = Path(self.base_output_dir) / usuario / 'LDA_Caracteristicas'
        resultado = procesar_lda_caracteristicas(
            df=caracteristicas,
            usuario=usuario,
            tipo_clase='Muestra',
            label_col=label_col,
            grupos_filter=grupos,
            output_path=str(output_path),
            save_output=True,
        )
        return resultado
    def entrenar_modelo(self, usuario, tipo='ALL', grupo=None, modelo='Regresión Logística'):
        usuario_base = Path(self.base_output_dir) / usuario
        trainer = LDAClassifierTrainer(
            usuario_base=str(usuario_base),
            output_base=str(usuario_base / 'Resultados_Clasificadores'),
            verbose=True,
            save_plots=True
        )

        if tipo.upper() == 'ALL':
            try:
                sources = trainer.discover_sources()
            except FileNotFoundError as exc:
                print(exc)
                return None

            tipo_sources = {k: v for k, v in sources.items() if k != 'LDA_General'}
            if not tipo_sources:
                print(f"No se encontraron tipos específicos para combinar en {usuario_base}.")
                return None

            dataframes = []
            for tipo_clase, ruta in tipo_sources.items():
                feature_sets = trainer.load_feature_sets(ruta)
                if not feature_sets:
                    continue
                if grupo is None:
                    dataframes.extend(feature_sets.values())
                else:
                    if grupo not in feature_sets:
                        print(f"No se encontró el grupo '{grupo}' en {tipo_clase}.")
                        continue
                    dataframes.append(feature_sets[grupo])

            if not dataframes:
                print(f"No se encontraron datos LDA válidos para {usuario}.")
                return None

            df_all = pd.concat(dataframes, ignore_index=True)
            resultado = trainer.train_model(df_all, model_name=modelo)

            print(f"\nEntrenamiento finalizado para {usuario} / ALL")
            print(f"  Modelo: {modelo}")
            print(f"  Accuracy: {resultado['metrics']['accuracy']:.2%}")
            print(f"  F1-Score: {resultado['metrics']['f1']:.2%}")
            print(f"  AUC-ROC: {resultado['metrics']['auc']:.2%}" if resultado['metrics']['auc'] >= 0 else "  AUC-ROC: N/A")
            return resultado

        ruta_lda = usuario_base / tipo / 'Resultados_LDA' if tipo != 'LDA_General' else usuario_base / 'LDA_General'
        if not ruta_lda.exists():
            print(f"No se encontró la carpeta LDA para {usuario} / {tipo}: {ruta_lda}")
            return None
        if modelo == 'ALL':
            print(f"Entrenando todos los modelos para {usuario} / {tipo} / {grupo or 'ALL GRUPOS'}")
            modelos = get_default_models().keys()
            resultados = {}
            for m in modelos:
                res = trainer.train_model_on_source(
                    ruta_grupos=ruta_lda,
                    group_name=grupo,
                    model_name=m,
                )
                resultados[m] = res
                print(f"  Modelo: {m} - Accuracy: {res['metrics']['accuracy']:.2%}, F1-Score: {res['metrics']['f1']:.2%}, AUC-ROC: {res['metrics']['auc']:.2%}" if res['metrics']['auc'] >= 0 else f"  Modelo: {m} - Accuracy: {res['metrics']['accuracy']:.2%}, F1-Score: {res['metrics']['f1']:.2%}, AUC-ROC: N/A")
            return resultados
        else:
            resultado = trainer.train_model_on_source(
                ruta_grupos=ruta_lda,
                group_name=grupo,
                model_name=modelo,
            )

            print(f"\nEntrenamiento finalizado para {usuario} / {tipo} / {grupo or 'ALL GRUPOS'}")
            print(f"  Modelo: {modelo}")
            print(f"  Accuracy: {resultado['metrics']['accuracy']:.2%}")
            print(f"  F1-Score: {resultado['metrics']['f1']:.2%}")
            print(f"  AUC-ROC: {resultado['metrics']['auc']:.2%}" if resultado['metrics']['auc'] >= 0 else "  AUC-ROC: N/A")
            return resultado

    def entrenar_todos_los_modelos(self, usuario, tipos_validos=None, carpeta_lda='Resultados_LDA', carpeta_general='LDA_General', save_plots=True, show_plots=True):
        usuario_base = Path(self.base_output_dir) / usuario

        trainer = LDAClassifierTrainer(
            usuario_base=str(usuario_base),
            output_base=str(usuario_base / 'Resultados_Clasificadores'),
            tipos_validos=tipos_validos,
            carpeta_lda=carpeta_lda,
            carpeta_general=carpeta_general,
            verbose=True,
            save_plots=save_plots,
        )

        resultados = trainer.evaluate_all(save_plots=save_plots, show_plots=show_plots)
        print(f"\nEntrenamiento completo terminado para {usuario}.")
        return resultados

def entrenar_pipeline_usuario(user):
    pipeline = PipelineCompletoLDA()
    print("Separando archivos")
    pipeline.separar_archivos_csv(usuario=user)
    print("SE SEPARARON LOS ARCHIVOS ")
    print("Obteniendo caracteristicas")
    pipeline.obtener_caracteristicas_usuario(usuario=user)
    print("EXTRAIDAS TODAS LAS CARACTERISTICAS")
    print("Optimizando LDA")
    pipeline.optimizacion_lda(user=user)
    print("Entrenado Modelo")
    #pipeline.entrenar_modelo(user, tipo='LDA_General', grupo='TODAS', modelo='ALL')
    pipeline.entrenar_todos_los_modelos(usuario=user, save_plots=False, show_plots=False)

def main_single_file():
    pipeline = PipelineCompletoLDA(base_output_dir='resultsOnlyOneFile')
    ruta = './captures/UserArcane/Letters/UserArcane_A_1.csv'
    resultado = pipeline.separar_archivo_por_ruta(ruta, usuario='UserArcane')
    print(f"Archivo separado guardado en: {resultado}")
    caracteristicas = pipeline.extraer_caracteristicas_file(resultado, usuario='UserArcane',save_output=False)

    print(f"Características extraídas para {ruta}:")
    print(caracteristicas.head())
    resultado_lda = pipeline.optimizar_lda_caracteristicas(usuario='UserArcane', caracteristicas=caracteristicas, label_col='label', grupos=['Estadisticas'])
    print(f"Resultado LDA optimizado para {ruta}:")
    print(resultado_lda)
#entrenar_pipeline_usuario(user='UserMartinEpoc')