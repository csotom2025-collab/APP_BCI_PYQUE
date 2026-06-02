
import os

from DivisorTiempos import SeparacionTiempos
from GetCaracteristiacas import FeatureExtractor
import pandas as pd

class PipelineCompleto:
    def __init__(self):
        self.base_output_dir = 'results'
        pass
    def separar_archivos_csv(self, usuario):
        separador = SeparacionTiempos(sampling_rate=250)
        
        # Procesar un archivo individual
        print("="*70)
        print("SEPARADOR DE TRIALS P300 - EEG")
        print("="*70 + "\n")
        
        usuario= 'Usermar'

        letras=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        lista_carpetas  =os.listdir(f'captures/{usuario}/')  
        print(lista_carpetas)
        for carpeta in lista_carpetas:
            tpComando= carpeta
            print(f"carpetea {tpComando}")
            archivos = os.listdir(f'captures/{usuario}/{tpComando}/')
            print(archivos)
            archivos_csv = [archivo for archivo in archivos if archivo.endswith('.csv')]
            for archivo in archivos_csv:
                resultado = separador.procesar_archivo(
                f'captures/{usuario}/{tpComando}/{archivo}',
                carpeta_salida_base=self.base_output_dir + f'/{usuario}/{tpComando}',
                )
            
            if resultado['exito']:
                print(f"\n✓ Procesamiento exitoso:")
                print(f"  Usuario: {resultado['user']}")
                print(f"  Letra: {resultado['letra']}")
                print(f"  Trial: {resultado['trial']}")
                print(f"  Tipo: {resultado['tipo']}")
                print(f"  Canales: {resultado['canales']}")
                print(f"  Muestras P300: {resultado['muestras_p300']}")
                print(f"  Muestras Trial: {resultado['muestras_trial']}")
                print(f"  Muestras Post: {resultado['muestras_post']}")
            else:
                print(f"\n✗ Error: {resultado['error']}")
    def obtener_caracteristicas_usuario(self,usuario):
        featureExtractor = FeatureExtractor(fs=250)
        subcarpetas = os.listdir(self.base_output_dir+f'/{usuario}/')
        for subcarpeta in subcarpetas:
            archivos = os.listdir(self.base_output_dir+f'/{usuario}/{subcarpeta}/')
            for archivo in archivos:
                letra = subcarpeta.split("_")[1]
                trial = subcarpeta.split("_")[2]

                ruta_archivo = os.path.join(self.base_output_dir+f'/{usuario}/{subcarpeta}/', archivo)
                print(ruta_archivo)
                try:
                    data = pd.read_csv(ruta_archivo)
                    channel_names = ["Oz", "Po7", "Po4", "Po3", "P4", "P3", "Po8", "Pz", "Fz", "F2", "F3", "F4", "AF3", "Cz", "AF4", "F1" ]
                    signals = data.values.T  # Transponer para tener shape (n_channels, n_samples)
                    features_df = featureExtractor.extract_features(signals, channel_names=channel_names, available_channel_names=channel_names,window_size=100,overlap=0)
                    print(f"Caracteristicas extraidas para {archivo}:")
                    print(f"tamanio de archivo {data.shape} len featrues{len(features_df)}")
                    #print(features_df.head())
                    # Guardar las caracteristicas en un nuevo archivo CSV
                    if not os.path.exists(f"self.base_output_dir/{usuario}/features"):
                        os.makedirs(f"self.base_output_dir{usuario}/features")
                    output_path = self.base_output_dir / f'features/{usuario}_{letra}_{trial}_features.csv'
                    features_df.to_csv(output_path, index=False)
                except Exception as e:
                    print(f"Error procesando {archivo}: {e}")
                print("-"*70)
                # Aquí puedes almacenar o procesar las características extraídas según tus necesidades


        return {
            'edad': 30,
            'genero': 'masculino',
            'experiencia': 'intermedia'
        }
if __name__ == "__main__":
    pipeline = PipelineCompleto()
    user = 'Usermar'
    pipeline.separar_archivos_csv(user)
    print("SODFJASODFJ")
    pipeline.obtener_caracteristicas_usuario(user)
    pipeline.optimizacion_lda()