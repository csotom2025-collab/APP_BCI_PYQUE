import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import sys

# Importar la clase de preprocesamiento EEG
sys.path.insert(0, str(Path(__file__).parent))
from DivisorTiempos import SeparacionTiempos


class BusquedaP300(SeparacionTiempos):
    def __init__(self, sampling_rate=250, ruta_base="d:/EEG_Python/results/User27/Char/Separados"):
        """
        Inicializa la búsqueda del P300
        
        Args:
            sampling_rate: Frecuencia de muestreo en Hz (por defecto 250 Hz)
            ruta_base: Ruta base donde están los archivos CSV
        """
        super().__init__(sampling_rate=sampling_rate)
        self.ruta_base = Path(ruta_base)
        self.datos = None
        self.datos_preprocesados = None
        self.canales_disponibles = []
        self.canales_seleccionados = []
        self.nombre_archivo_actual = None
        
    def cargar_archivo(self, nombre_archivo: str) -> bool:
        """
        Carga un archivo CSV de P300 y aplica preprocesamiento EEG
        
        Args:
            nombre_archivo: Nombre del archivo (ej: 'User27_A_0_P300.csv')
            
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        ruta_completa = self.ruta_base / nombre_archivo
        
        try:
            self.datos = pd.read_csv(ruta_completa)
            self.nombre_archivo_actual = nombre_archivo
            self.canales_disponibles = list(self.datos.columns)
            
            # Aplicar preprocesamiento EEG (normalización + filtros)
            canales_array = self.datos[self.canales_disponibles].to_numpy().T
            canales_normalizados = self.normalize_eeg(canales_array)
            canales_filtrados = self.preprocess_eeg(canales_normalizados, fs=self.sampling_rate)
            
            # Guardar datos preprocesados
            self.datos_preprocesados = pd.DataFrame(
                canales_filtrados.T,
                columns=self.canales_disponibles
            )
            
            print(f"✓ Archivo cargado: {nombre_archivo}")
            print(f"  Dimensiones: {self.datos.shape}")
            print(f"  Canales disponibles: {len(self.canales_disponibles)}")
            print(f"  ✓ Preprocesamiento EEG aplicado (normalización + filtrado)")
            return True
        except FileNotFoundError:
            print(f"✗ Error: Archivo no encontrado en {ruta_completa}")
            return False
        except Exception as e:
            print(f"✗ Error al cargar archivo: {e}")
            return False
    
    def listar_archivos_disponibles(self) -> List[str]:
        """
        Lista todos los archivos CSV disponibles en la ruta
        
        Returns:
            Lista de archivos CSV
        """
        if not self.ruta_base.exists():
            print(f"✗ La ruta no existe: {self.ruta_base}")
            return []
        
        archivos = list(self.ruta_base.glob("*P300.csv"))
        return [f.name for f in archivos]
    
    def seleccionar_canales(self, nombres_canales: Optional[List[str]] = None) -> bool:
        """
        Selecciona los canales a analizar
        
        Args:
            nombres_canales: Lista de nombres de canales a seleccionar.
                            Si es None, muestra menu interactivo
        
        Returns:
            True si se seleccionaron canales correctamente
        """
        if self.datos is None:
            print("✗ Primero debes cargar un archivo")
            return False
        
        if nombres_canales is None:
            # Menu interactivo
            print("\n=== CANALES DISPONIBLES ===")
            for i, canal in enumerate(self.canales_disponibles, 1):
                print(f"{i:2}. {canal}")
            
            print("\nIngresa los números de los canales separados por comas")
            print("(ej: 1,2,3 o presiona Enter para todos):")
            
            entrada = "1,2,4,8,9,14"
            
            if not entrada:
                self.canales_seleccionados = self.canales_disponibles
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in entrada.split(",")]
                    self.canales_seleccionados = [self.canales_disponibles[i] 
                                                   for i in indices 
                                                   if 0 <= i < len(self.canales_disponibles)]
                except ValueError:
                    print("✗ Entrada inválida")
                    return False
        else:
            # Validar que los canales existan
            canales_validos = [c for c in nombres_canales if c in self.canales_disponibles]
            if not canales_validos:
                print(f"✗ Ninguno de los canales especificados existe")
                return False
            self.canales_seleccionados = canales_validos
        
        print(f"\n✓ Canales seleccionados ({len(self.canales_seleccionados)}):")
        for canal in self.canales_seleccionados:
            print(f"  - {canal}")
        return True
    
    def buscar_p300(self, ventana_inicio_ms: int = 200, ventana_fin_ms: int = 600, usar_preprocesado: bool = True) -> Dict:
        """
        Busca características del P300 en los canales seleccionados
        
        Args:
            ventana_inicio_ms: Inicio de ventana de búsqueda en ms
            ventana_fin_ms: Fin de ventana de búsqueda en ms
            usar_preprocesado: Si True, usa datos preprocesados; si False, usa datos crudos
            
        Returns:
            Diccionario con resultados del P300
        """
        if self.datos is None:
            print("✗ Primero debes cargar un archivo")
            return {}
        
        if not self.canales_seleccionados:
            print("✗ Primero debes seleccionar canales")
            return {}
        
        # Usar datos preprocesados si están disponibles y se solicita
        datos_analisis = self.datos_preprocesados if (usar_preprocesado and self.datos_preprocesados is not None) else self.datos
        
        # Convertir ventanas de ms a muestras
        ventana_inicio_muestra = int((ventana_inicio_ms / 1000) * self.sampling_rate)
        ventana_fin_muestra = int((ventana_fin_ms / 1000) * self.sampling_rate)
        
        resultados = {
            'archivo': self.nombre_archivo_actual,
            'ventana_ms': (ventana_inicio_ms, ventana_fin_ms),
            'ventana_muestras': (ventana_inicio_muestra, ventana_fin_muestra),
            'datos_tipo': 'Preprocesados (normalizado + filtrado)' if usar_preprocesado and self.datos_preprocesados is not None else 'Crudos',
            'canales': {}
        }
        
        for canal in self.canales_seleccionados:
            datos_canal = datos_analisis[canal].values[ventana_inicio_muestra:ventana_fin_muestra]
            
            # Calcular características del P300
            amplitud_max = np.max(np.abs(datos_canal))
            amplitud_positiva = np.max(datos_canal)
            amplitud_negativa = np.min(datos_canal)
            idx_max = np.argmax(np.abs(datos_canal))
            latencia_max = (idx_max + ventana_inicio_muestra) / self.sampling_rate * 1000
            promedio = np.mean(datos_canal)
            desv_std = np.std(datos_canal)
            area_integral = np.sum(datos_canal)  # Área bajo la curva
            
            resultados['canales'][canal] = {
                'amplitud_max': round(amplitud_max, 6),
                'amplitud_positiva': round(amplitud_positiva, 6),
                'amplitud_negativa': round(amplitud_negativa, 6),
                'latencia_max_ms': round(latencia_max, 2),
                'promedio': round(promedio, 6),
                'desv_std': round(desv_std, 6),
                'area_integral': round(area_integral, 6),
                'datos': datos_canal
            }
        
        return resultados
    
    def mostrar_resultados(self, resultados: Dict) -> None:
        """
        Muestra los resultados de la búsqueda del P300 de forma formateada
        
        Args:
            resultados: Diccionario con los resultados de buscar_p300
        """
        if not resultados:
            print("✗ No hay resultados para mostrar")
            return
        
        print("\n" + "="*80)
        print("RESULTADOS DE BÚSQUEDA DEL P300")
        print("="*80)
        print(f"Archivo: {resultados.get('archivo', 'N/A')}")
        print(f"Tipo de datos: {resultados.get('datos_tipo', 'N/A')}")
        print(f"Ventana de búsqueda: {resultados['ventana_ms'][0]}-{resultados['ventana_ms'][1]} ms")
        print(f"Muestras: {resultados['ventana_muestras'][0]}-{resultados['ventana_muestras'][1]}")
        print("-"*80)
        
        for canal, datos_canal in resultados['canales'].items():
            print(f"\n📊 CANAL: {canal}")
            print(f"   Amplitud máxima (|V|): {datos_canal['amplitud_max']}")
            print(f"   Amplitud positiva (mV): {datos_canal['amplitud_positiva']}")
            print(f"   Amplitud negativa (mV): {datos_canal['amplitud_negativa']}")
            print(f"   Latencia del pico (ms): {datos_canal['latencia_max_ms']}")
            print(f"   Promedio: {datos_canal['promedio']}")
            print(f"   Desv. Estándar: {datos_canal['desv_std']}")
            print(f"   Área integral: {datos_canal['area_integral']}")
    
    def graficar_p300(self, resultados: Dict, guardar: bool = False, nombre_salida: str = "P300_analisis.png") -> None:
        """
        Grafica los datos del P300 para los canales seleccionados
        
        Args:
            resultados: Diccionario con los resultados de buscar_p300
            guardar: Si True, guarda la imagen
            nombre_salida: Nombre del archivo de salida
        """
        if not resultados or not resultados['canales']:
            print("✗ No hay resultados para graficar")
            return
        
        num_canales = len(resultados['canales'])
        fig, axes = plt.subplots(num_canales, 1, figsize=(12, 3*num_canales))
        
        if num_canales == 1:
            axes = [axes]
        
        ventana_inicio_ms = resultados['ventana_ms'][0]
        tiempo_ms = np.arange(len(resultados['canales'][list(resultados['canales'].keys())[0]]['datos'])) / self.sampling_rate * 1000 + ventana_inicio_ms
        
        for idx, (canal, datos_canal) in enumerate(resultados['canales'].items()):
            ax = axes[idx]
            ax.plot(tiempo_ms, datos_canal['datos'], 'b-', linewidth=1.5, label=canal)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Amplitud (mV)')
            ax.set_title(f'P300 - Canal {canal} | Amplitud Max: {datos_canal["amplitud_max"]:.4f}')
            ax.legend()
        
        plt.tight_layout()
        
        if guardar:
            plt.savefig(nombre_salida, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico guardado como: {nombre_salida}")
        
        plt.show()
    
    def comparar_canales(self, resultados: Dict) -> None:
        """
        Compara la amplitud del P300 entre canales
        
        Args:
            resultados: Diccionario con los resultados de buscar_p300
        """
        if not resultados or not resultados['canales']:
            print("✗ No hay resultados para comparar")
            return
        
        print("\n" + "="*80)
        print("COMPARACIÓN DE AMPLITUDES ENTRE CANALES")
        print("="*80)
        
        amplitudes = {canal: datos['amplitud_max'] 
                     for canal, datos in resultados['canales'].items()}
        
        amplitudes_ordenadas = sorted(amplitudes.items(), key=lambda x: x[1], reverse=True)
        
        for i, (canal, amplitud) in enumerate(amplitudes_ordenadas, 1):
            barra = "█" * int(amplitud * 100)
            print(f"{i:2}. {canal:6} | {barra} {amplitud:.6f}")
        
        canal_max = amplitudes_ordenadas[0][0]
        print(f"\n★ Canal con mayor amplitud: {canal_max} ({amplitudes_ordenadas[0][1]:.6f})")
    
    def exportar_resultados(self, resultados: Dict, nombre_salida: str = "resultados_P300.csv") -> bool:
        """
        Exporta los resultados del análisis del P300 a un archivo CSV
        
        Args:
            resultados: Diccionario con los resultados de buscar_p300
            nombre_salida: Nombre del archivo de salida
            
        Returns:
            True si se exportó correctamente
        """
        if not resultados or not resultados['canales']:
            print("✗ No hay resultados para exportar")
            return False
        
        try:
            datos_export = []
            for canal, datos_canal in resultados['canales'].items():
                datos_export.append({
                    'Archivo': resultados.get('archivo', 'N/A'),
                    'Canal': canal,
                    'Amplitud_Max': datos_canal['amplitud_max'],
                    'Amplitud_Positiva': datos_canal['amplitud_positiva'],
                    'Amplitud_Negativa': datos_canal['amplitud_negativa'],
                    'Latencia_ms': datos_canal['latencia_max_ms'],
                    'Promedio': datos_canal['promedio'],
                    'Desv_Std': datos_canal['desv_std'],
                    'Area_Integral': datos_canal['area_integral']
                })
            
            df_export = pd.DataFrame(datos_export)
            df_export.to_csv(nombre_salida, index=False)
            print(f"✓ Resultados exportados a: {nombre_salida}")
            return True
        except Exception as e:
            print(f"✗ Error al exportar resultados: {e}")
            return False


# Script de prueba
if __name__ == "__main__":
    # Crear instancia
    busqueda = BusquedaP300()
    
    # Listar archivos disponibles
    print("=== ARCHIVOS DISPONIBLES ===")
    archivos = busqueda.listar_archivos_disponibles()
    if archivos:
        for i, arch in enumerate(archivos, 1):
            print(f"{i}. {arch}")
    else:
        print("No se encontraron archivos")
        exit()
    
    # Cargar archivo
    print("\nIngresa el nombre del archivo (ej: User27_A_0_P300.csv):")
    archivo = input().strip()
    
    if not busqueda.cargar_archivo(archivo):
        exit()
    
    # Seleccionar canales
    busqueda.seleccionar_canales()
    
    # Buscar P300
    print("\n¿Ventana de búsqueda personalizada? (s/n, por defecto 200-600ms):")
    if input().strip().lower() == 's':
        print("Ingresa inicio (ms):")
        inicio = int(input())
        print("Ingresa fin (ms):")
        fin = int(input())
    else:
        inicio, fin = 200, 600
    
    # Elegir si usar datos preprocesados o crudos
    print("\n¿Usar datos preprocesados? (s/n, por defecto si):")
    usar_preprocesado = input().strip().lower() != 'n'
    
    resultados = busqueda.buscar_p300(inicio, fin, usar_preprocesado=usar_preprocesado)
    
    # Mostrar y graficar resultados
    busqueda.mostrar_resultados(resultados)
    busqueda.comparar_canales(resultados)
    
    # Exportar resultados
    print("\n¿Exportar resultados a CSV? (s/n):")
    if input().strip().lower() == 's':
        busqueda.exportar_resultados(resultados)
    
    print("\n¿Deseas ver gráficos? (s/n):")
    if input().strip().lower() == 's':
        busqueda.graficar_p300(resultados)