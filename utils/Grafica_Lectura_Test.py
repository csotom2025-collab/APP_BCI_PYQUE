import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def graficar_captura(user, letter, capture_number):
    """
    Grafica una captura específica del usuario
    """
    try:
        # Intentar Letter primero
        filename = f'captures/{user}/Letters/{user}_{letter}_{capture_number}.csv'
        
        # Si no existe, intentar Numbers
        if not os.path.exists(filename):
            filename = f'captures/{user}/Numbers/{user}_{letter}_{capture_number}.csv'
        
        # Si aún no existe, intentar Controls
        if not os.path.exists(filename):
            filename = f'captures/{user}/Controls/{user}_{letter}_{capture_number}.csv'
        
        if not os.path.exists(filename):
            print(f"❌ Archivo no encontrado: {filename}")
            return False
        
        df = pd.read_csv(filename)
        
        # Mostrar información del archivo
        print(f"\n✅ Cargado: {filename}")
        print(f"   Forma del dataset: {df.shape}")
        print(f"   Canales disponibles: {df.columns.tolist()}")

        # Crear gráficas por canal
        fig, axes = plt.subplots(8, 2, figsize=(14, 10))
        fig.suptitle(f'Señales EEG - {user} {letter} #{capture_number}', fontsize=16)

        # Aplanar el array de axes para iterar fácilmente
        axes = axes.flatten()

        df['Tm'] = [i for i in range(len(df))]
        # Plotear cada canal con Time en el eje X
        channels = [col for col in df.columns if col != 'Tm']
        for idx, column in enumerate(channels):
            axes[idx].plot(df['Tm'], df[column], linewidth=0.8, color='blue')
            axes[idx].set_title(f'Canal: {column}')
            axes[idx].set_xlabel('Tiempo (muestra)')
            axes[idx].set_ylabel('Amplitud')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)  # No bloquear para permitir más ventanas
        return True
        
    except Exception as e:
        print(f"❌ Error al procesar el archivo: {e}")
        return False

# Solicitar usuario una sola vez
numvber = input("\n📊 Ingrese el número de usuario: ").strip()
user = "User" + numvber

print(f"\n✓ Usuario seleccionado: {user}")
print("─" * 50)

# Loop para solicitar múltiples capturas
while True:
    print("\n📋 Solicitar nueva captura")
    print("(Escribe 'salir' en cualquier campo para terminar)")
    
    letra = input("Ingrese la letra/número a graficar (o 'salir'): ").strip().upper()
    if letra.lower() == 'salir':
        print("\n👋 Hasta luego!")
        break
    
    try:
        capture_number = int(input("Ingrese el número de captura: "))
        
        # Graficar la captura
        if graficar_captura(user, letra, capture_number):
            print("✓ Ventana abierta. Puedes solicitar otra captura o escribe 'salir'")
        
    except ValueError:
        print("❌ El número de captura debe ser un entero")
    except Exception as e:
        print(f"❌ Error: {e}")
