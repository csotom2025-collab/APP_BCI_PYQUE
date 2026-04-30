"""
Script Simple para Extraer y Visualizar P300
Uso rápido y directo para una grabación de 2 segundos
"""

from pathlib import Path
from P300_Window_Extractor import P300WindowExtractor


def main():
    # ==========================================================================
    # CONFIGURACIÓN - MODIFICAR SEGÚN TU CASO
    # ==========================================================================
    
    # Datos del ensayo a analizar
    USER = "User1"           # Usuario (ej: User1, User2, etc.)
    SYMBOL = "A"             # Símbolo/Letra (ej: A, H, etc.)
    TRIAL = 0                # Número de ensayo (0, 1, 2, etc.)
    CATEGORY = "Letters"     # Categoría: "Letters", "Numbers", "Controls"
    
    # Canales a visualizar (máx 8 por claridad)
    CHANNELS = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    
    # ==========================================================================
    # EJECUTAR ANÁLISIS
    # ==========================================================================
    
    file_path = f'captures/{USER}/{CATEGORY}/{USER}_{SYMBOL}_{TRIAL}.edf'
    
    print("\n" + "="*70)
    print("EXTRACTOR DE VENTANAS P300")
    print("="*70)
    print(f"\n📁 Archivo: {file_path}")
    print(f"👤 Usuario: {USER}")
    print(f"🎯 Símbolo/Comando: {SYMBOL}")
    print(f"🔢 Ensayo: {TRIAL}")
    
    # Crear extractor
    extractor = P300WindowExtractor(data_path='.')
    
    # Analizar
    results = extractor.analyze_single_trial(
        file_path=file_path,
        channel_names=CHANNELS,
        save_results=True,
        out_dir=f'results/{USER}/{CATEGORY}'
    )
    
    if results:
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)
        
        # Mostrar características del P300
        p300_features = results['p300_features']
        
        print("\n📊 CARACTERÍSTICAS DEL P300 (0.5 - 1.2s):")
        print("-" * 70)
        print(p300_features.to_string(index=False))
        
        # Estadísticas
        print("\n📈 ESTADÍSTICAS GLOBALES:")
        print(f"   Amplitud P300 máxima: {p300_features['P300_Amplitude_Max'].max():.2f} µV")
        print(f"   Amplitud P300 mínima: {p300_features['P300_Amplitude_Min'].min():.2f} µV")
        print(f"   Amplitud P300 promedio: {p300_features['P300_Amplitude_Max'].mean():.2f} µV")
        print(f"   Latencia promedio: {p300_features['P300_Latency_Max'].mean():.3f} s")
        print(f"   RMS promedio: {p300_features['P300_RMS'].mean():.2f} µV")
        
        # Identificar canal con mayor P300
        max_idx = p300_features['P300_Amplitude_Max'].abs().idxmax()
        max_channel = p300_features.loc[max_idx, 'Channel']
        max_amplitude = p300_features.loc[max_idx, 'P300_Amplitude_Max']
        
        print(f"\n🎯 CANAL CON MAYOR P300: {max_channel}")
        print(f"   Amplitud: {max_amplitude:.2f} µV")
        
        print("\n" + "="*70)
        print("💾 Resultados guardados en:")
        print(f"   📊 results/{USER}/{CATEGORY}/{USER}_{SYMBOL}_{TRIAL}_p300_features.csv")
        print(f"   🖼️  Gráficas en la carpeta: results/{USER}/{CATEGORY}/")
        print("="*70 + "\n")
    else:
        print("\n❌ Error: No se pudo analizar el archivo")


if __name__ == "__main__":
    main()
