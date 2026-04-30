"""
Script de ejemplo: Análisis de P300 para múltiples ensayos y asociación con comandos
"""

from pathlib import Path
from P300_Window_Extractor import P300WindowExtractor
import pandas as pd
import numpy as np


def create_command_mapping():
    """
    Crea el mapeo entre archivos y comandos ejecutados
    """
    command_map = {
        'Letters': {
            'A': {'label': 'Letra A', 'type': 'escritura'},
            'B': {'label': 'Letra B', 'type': 'escritura'},
            'C': {'label': 'Letra C', 'type': 'escritura'},
            'D': {'label': 'Letra D', 'type': 'escritura'},
            'E': {'label': 'Letra E', 'type': 'escritura'},
            'F': {'label': 'Letra F', 'type': 'escritura'},
            'G': {'label': 'Letra G', 'type': 'escritura'},
            'H': {'label': 'Letra H', 'type': 'escritura'},
            'I': {'label': 'Letra I', 'type': 'escritura'},
            'J': {'label': 'Letra J', 'type': 'escritura'},
            'K': {'label': 'Letra K', 'type': 'escritura'},
            'L': {'label': 'Letra L', 'type': 'escritura'},
            'M': {'label': 'Letra M', 'type': 'escritura'},
            'N': {'label': 'Letra N', 'type': 'escritura'},
            'O': {'label': 'Letra O', 'type': 'escritura'},
            'P': {'label': 'Letra P', 'type': 'escritura'},
            'Q': {'label': 'Letra Q', 'type': 'escritura'},
            'R': {'label': 'Letra R', 'type': 'escritura'},
            'S': {'label': 'Letra S', 'type': 'escritura'},
            'T': {'label': 'Letra T', 'type': 'escritura'},
            'U': {'label': 'Letra U', 'type': 'escritura'},
            'V': {'label': 'Letra V', 'type': 'escritura'},
            'W': {'label': 'Letra W', 'type': 'escritura'},
            'X': {'label': 'Letra X', 'type': 'escritura'},
            'Y': {'label': 'Letra Y', 'type': 'escritura'},
            'Z': {'label': 'Letra Z', 'type': 'escritura'},
        },
        'Numbers': {
            '1': {'label': 'Número 1', 'type': 'números'},
            '2': {'label': 'Número 2', 'type': 'números'},
            '3': {'label': 'Número 3', 'type': 'números'},
            '4': {'label': 'Número 4', 'type': 'números'},
            '5': {'label': 'Número 5', 'type': 'números'},
            '6': {'label': 'Número 6', 'type': 'números'},
            '7': {'label': 'Número 7', 'type': 'números'},
            '8': {'label': 'Número 8', 'type': 'números'},
            '9': {'label': 'Número 9', 'type': 'números'},
        },
        'Controls': {
            '───': {'label': 'Control sin comando', 'type': 'control'},
            '⟵': {'label': 'Retroceso (Backspace)', 'type': 'control'},
            '↩': {'label': 'Enter/Enviar', 'type': 'control'},
        }
    }
    return command_map


def analyze_multiple_trials(user, category, symbols, num_trials=3, 
                           save_results=True):
    """
    Analiza múltiples ensayos para diferentes símbolos/comandos
    
    Args:
        user: Usuario (ej: 'User1')
        category: Categoría ('Letters', 'Numbers', 'Controls')
        symbols: Lista de símbolos a analizar (ej: ['A', 'H'])
        num_trials: Número de ensayos por símbolo
        save_results: Guardar resultados
    """
    extractor = P300WindowExtractor(data_path='.')
    command_map = create_command_mapping()
    
    all_results = []
    out_dir = Path('results') / user / category
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE P300 - {user} - {category}")
    print(f"{'='*70}")
    
    for symbol in symbols:
        print(f"\n📍 Analizando símbolo: {symbol}")
        print(f"   Comando: {command_map[category][symbol]['label']}")
        print(f"   {'-'*60}")
        
        for trial in range(num_trials):
            file_path = f'captures/{user}/{category}/{user}_{symbol}_{trial}.edf'
            
            # Verificar si el archivo existe
            if not Path(file_path).exists():
                print(f"   ⚠️  Archivo no encontrado: {file_path}")
                continue
            
            print(f"   Ensayo {trial}: ", end='', flush=True)
            
            try:
                # Analizar ensayo
                results = extractor.analyze_single_trial(
                    file_path=file_path,
                    save_results=False,
                    out_dir=out_dir
                )
                
                if results:
                    # Extraer características clave del P300
                    p300_features = results['p300_features']
                    
                    # Encontrar canal con mayor amplitud P300
                    max_amp_idx = p300_features['P300_Amplitude_Max'].abs().idxmax()
                    max_channel = p300_features.loc[max_amp_idx, 'Channel']
                    max_amplitude = p300_features.loc[max_amp_idx, 'P300_Amplitude_Max']
                    max_latency = p300_features.loc[max_amp_idx, 'P300_Latency_Max']
                    
                    all_results.append({
                        'User': user,
                        'Category': category,
                        'Symbol': symbol,
                        'Command': command_map[category][symbol]['label'],
                        'Trial': trial,
                        'Filename': results['filename'],
                        'Max_P300_Channel': max_channel,
                        'Max_P300_Amplitude_uV': max_amplitude,
                        'P300_Latency_s': max_latency,
                        'Mean_Amplitude_All_Channels': p300_features['P300_Amplitude_Max'].mean(),
                        'RMS_All_Channels': p300_features['P300_RMS'].mean()
                    })
                    
                    print(f"✓ Amplitud máxima: {max_amplitude:.2f}µV en {max_channel} @ {max_latency:.3f}s")
                else:
                    print("✗ Error en análisis")
                    
            except Exception as e:
                print(f"✗ Excepción: {str(e)}")
                continue
    
    # Guardar resultados en tabla
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print(f"\n{'='*70}")
        print("RESUMEN DE RESULTADOS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
        
        # Guardar CSV
        if save_results:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f'{user}_{category}_p300_summary.csv'
            results_df.to_csv(csv_path, index=False)
            print(f"\n💾 Resultados guardados en: {csv_path}")
        
        # Estadísticas por comando
        print(f"\n{'='*70}")
        print("ESTADÍSTICAS POR COMANDO")
        print(f"{'='*70}")
        stats_by_symbol = results_df.groupby('Symbol').agg({
            'Max_P300_Amplitude_uV': ['mean', 'std', 'min', 'max'],
            'P300_Latency_s': ['mean', 'std'],
        }).round(4)
        print(stats_by_symbol)
        
        return results_df
    else:
        print("⚠️  No se obtuvieron resultados")
        return None


def visualize_p300_by_command(results_df):
    """
    Visualiza comparación de P300 entre diferentes comandos
    """
    import matplotlib.pyplot as plt
    
    if results_df is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Amplitud máxima del P300 por símbolo
    ax1 = axes[0]
    symbols = results_df['Symbol'].unique()
    p300_amps = [results_df[results_df['Symbol'] == s]['Max_P300_Amplitude_uV'].values 
                 for s in symbols]
    
    bp1 = ax1.boxplot(p300_amps, labels=symbols, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Amplitud P300 (µV)', fontsize=11)
    ax1.set_xlabel('Símbolo/Comando', fontsize=11)
    ax1.set_title('Distribución de Amplitudes P300 por Comando', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfica 2: Latencia del P300 por símbolo
    ax2 = axes[1]
    latencies = [results_df[results_df['Symbol'] == s]['P300_Latency_s'].values 
                 for s in symbols]
    
    bp2 = ax2.boxplot(latencies, labels=symbols, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_ylabel('Latencia del Pico (s)', fontsize=11)
    ax2.set_xlabel('Símbolo/Comando', fontsize=11)
    ax2.set_title('Distribución de Latencias P300 por Comando', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/p300_command_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Gráfica de comparación guardada en: results/p300_command_comparison.png")
    plt.show()


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    # Analizar Letters
    print("\n\n" + "="*70)
    print("ANÁLISIS DE LETRAS")
    print("="*70)
    results_letters = analyze_multiple_trials(
        user='User1',
        category='Letters',
        symbols=['A', 'H'],
        num_trials=3,
        save_results=True
    )
    
    # Analizar Numbers
    print("\n\n" + "="*70)
    print("ANÁLISIS DE NÚMEROS")
    print("="*70)
    results_numbers = analyze_multiple_trials(
        user='User1',
        category='Numbers',
        symbols=['1', '5', '8', '9'],
        num_trials=3,
        save_results=True
    )
    
    # Analizar Controls
    print("\n\n" + "="*70)
    print("ANÁLISIS DE CONTROLES")
    print("="*70)
    results_controls = analyze_multiple_trials(
        user='User1',
        category='Controls',
        symbols=['───', '⟵', '↩'],
        num_trials=3,
        save_results=True
    )
    
    # Visualizar comparación
    if results_letters is not None:
        visualize_p300_by_command(results_letters)
