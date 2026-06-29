import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

def graficar_captura(user, letter, capture_number, apply_notch=False, clear_baseline=False):
    """
    Grafica una captura específica del usuario
    
    Args:
        user: Nombre del usuario
        letter: Letra o número a graficar
        capture_number: Número de captura
        apply_notch: Si es True, aplica filtro notch a 50 Hz para eliminar ruido de línea
        clear_baseline: Si es True, elimina la línea base de las señales
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
        if apply_notch:
            print(f"   ✓ Filtro notch aplicado (50 Hz)")
        if clear_baseline:
            print(f"   ✓ Eliminación de línea base aplicada")

        # Crear gráficas por canal
        fig, axes = plt.subplots(8, 2, figsize=(12, 8))
        fig.suptitle(f'Señales EEG - {user} {letter} #{capture_number}', fontsize=16)

        # Aplanar el array de axes para iterar fácilmente
        axes = axes.flatten()

        df['Tm'] = [i for i in range(len(df))]
        # Plotear cada canal con Time en el eje X
        channels = [col for col in df.columns if col != 'Tm']
        for idx, column in enumerate(channels):
            signal_data = df[column].values.copy()
            
            # Aplicar eliminación de línea base
            if clear_baseline:
                baseline = np.mean(signal_data[:int(len(signal_data) * 0.1)])
                signal_data = signal_data - baseline
            
            # Aplicar filtro notch a 50 Hz
            if apply_notch:
                fs = 250  # Frecuencia de muestreo (ajustar según tus datos)
                freq_notch = 50  # Frecuencia de línea (50 Hz)
                quality = 30  # Factor de calidad
                b, a = signal.iirnotch(freq_notch, quality, fs)
                signal_data = signal.filtfilt(b, a, signal_data)
            
            axes[idx].plot(df['Tm'], signal_data, linewidth=0.8, color='blue')
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

def graficar_captura_sobrepuesta(user, letter, capture_number, apply_notch=False, clear_baseline=False):
    """
    Grafica una captura específica del usuario en una nueva ventana sin bloquear.
    
    Args:
        user: Nombre del usuario
        letter: Letra o número a graficar
        capture_number: Número de captura
        apply_notch: Si es True, aplica filtro notch a 50 Hz para eliminar ruido de línea
        clear_baseline: Si es True, elimina la línea base de las señales
    """
    try:
        # Activar el modo interactivo de matplotlib para evitar bloqueos globales
        plt.ion() 
        
        # 1. Intentar Letter primero
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
        print(df.head())
        # Mostrar información del archivo
        print(f"\n✅ Cargado: {filename}")
        print(f"   Forma del dataset: {df.shape}")
        print(f"   Canales disponibles: {df.columns.tolist()}")
        if apply_notch:
            print(f"   ✓ Filtro notch aplicado (50 Hz)")
        if clear_baseline:
            print(f"   ✓ Eliminación de línea base aplicada")

        channels = [col for col in df.columns if col != 'Tm']
        print(channels)
        
        color_array = ['brown','orange','yellow','green','blue','purple','gray','white'] * 2
        plt.style.use('dark_background')
        
        # =========================================================================
        # CRUCIAL: Crear una NUEVA ventana (Figura) única para esta llamada
        # =========================================================================
        fig = plt.figure(figsize=(12, 6)) 
        fig.canvas.manager.set_window_title(f'EEG - {user} {letter} #{capture_number}')
        
        # Graficar los canales en la figura actual
        for idx, column in enumerate(channels):
            signal_data = df[column].values.copy()
            # Aplicar filtro notch a 50 Hz
            fs = 250  # Frecuencia de muestreo (ajustar según tus datos)
            if apply_notch:
                freq_notch = 60  # Frecuencia de línea (50 Hz)
                quality = 30  # Factor de calidad
                b, a = signal.iirnotch(freq_notch, quality, fs)
                signal_data = signal.filtfilt(b, a, signal_data)
            
            # Aplicar eliminación de línea base
            if clear_baseline:
                n_muestras_baseline = int(0.5 * fs) # 0.5 segundos * 250 Hz = 125 muestras
                # Calculamos el promedio de esas primeras 125 muestras en el arreglo 1D
                baseline_mean = np.mean(signal_data[:n_muestras_baseline])
                # Restamos el promedio a toda la señal
                signal_data = signal_data - baseline_mean
            
            
            plt.plot(df['Tm'], signal_data, linewidth=0.8, color=color_array[idx], label=column)
        
        plt.title(f'Señales EEG - {user} {letter} #{capture_number}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # =========================================================================
        # CRUCIAL: Dibujar la ventana sin bloquear la ejecución
        # =========================================================================
        plt.draw()             # Fuerza el renderizado de la nueva figura
        plt.pause(0.1)         # Pequeña pausa necesaria para que el backend de GUI procese el evento
        
        return True
        
    except Exception as e:
        print(f"❌ Error al procesar el archivo: {e}")
        return False

while True:  # 🔄 Bucle principal para el cambio de usuario
    # Solicitar usuario
    numvber = input("\n📊 Ingrese el número de usuario (o 'salir' para cerrar): ").strip()
    
    if numvber.lower() == 'salir':
        print("\n👋 ¡Hasta luego!")
        break
        
    if not numvber:
        print("❌ El número de usuario no puede estar vacío.")
        continue

    user = "User" + numvber
    print(f"\n✓ Usuario seleccionado: {user}")
    print("─" * 50)

    # Loop para solicitar múltiples capturas de ESTE usuario
    while True:
        print("\n📋 Solicitar nueva captura")
        print("(Escribe 'cambio' para cambiar de usuario, o 'salir' para terminar)")
        
        letra = input("Ingrese la letra/número a graficar: ").strip().upper()
        
        # Evaluar salidas o cambios de estado
        if letra.lower() == 'salir':
            print("\n👋 ¡Hasta luego!")
            exit() # Cierra el programa por completo
            
        if letra.lower() == 'cambio':
            print(f"\n🔄 Cambiando de usuario (Saliendo de {user})...")
            print("─" * 50)
            break # Rompe este bucle y regresa al inicio del bucle principal
        
        try:
            # Capturar la entrada del número como texto primero para checar 'cambio' o 'salir'
            num_cap_input = input("Ingrese el número de captura: ").strip().lower()
            
            if num_cap_input == 'salir':
                print("\n👋 ¡Hasta luego!")
                exit()
            if num_cap_input == 'cambio':
                print(f"\n🔄 Cambiando de usuario (Saliendo de {user})...")
                print("─" * 50)
                break
                
            capture_number = int(num_cap_input)
            
            # Graficar la captura
            if graficar_captura_sobrepuesta(user, letra, capture_number, apply_notch=False, clear_baseline=False):
                print("✓ Ventana abierta. Puedes solicitar otra captura, escribir 'cambio' o 'salir'")
            
        except ValueError:
            print("❌ El número de captura debe ser un entero")
        except Exception as e:
            print(f"❌ Error: {e}")
