import pandas as pd
import mne
import os
import matplotlib.pyplot as plt

def graficar_mne(user, letter, capture_number):
    try:
        # 1. Localizar el archivo (misma lógica que ya tenías)
        base_path = f'captures/{user}'
        subfolders = ['Letters', 'Numbers', 'Controls']
        filename = None
        
        for folder in subfolders:
            temp_path = f'{base_path}/{folder}/{user}_{letter}_{capture_number}.csv'
            if os.path.exists(temp_path):
                filename = temp_path
                break
        
        if not filename:
            print(f"❌ Archivo no encontrado.")
            return False

        # 2. Cargar datos
        df = pd.read_csv(filename)
        
        # Si tienes columnas que no son EEG (como 'Tm' o timestamps), hay que quitarlas
        # Asumiendo que tus columnas de canales se llaman 'CH1', 'CH2', etc.
        if 'Tm' in df.columns:
            df = df.drop(columns=['Tm'])
            
        # MNE espera que los datos estén en formato (canales, muestras) y en Voltios
        # Si tu ADS1299 entrega microvoltios, hay que multiplicar por 1e-6
        data = df.values.T #* 1e-6 
        
        # 3. Crear la información del dataset (Info Object)
        ch_names = df.columns.tolist()
        sfreq = 250  # La frecuencia de muestreo que configuraste en Arduino
        ch_types = ['eeg'] * len(ch_names)
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # 4. Crear el objeto Raw de MNE
        raw = mne.io.RawArray(data, info)
        
        # 5. Visualización interactiva
        # 'scalings' ajusta el zoom vertical automáticamente
        print(f"✅ Visualizando {filename} con MNE...")
        
        # Aplicar un filtro rápido para que se vea mejor en el plot
        raw.filter(l_freq=0.1, h_freq=40) 
        
        fig = raw.plot(
            duration=5,      # Cuántos segundos mostrar a la vez
            n_channels=16,   # Cuántos canales ver simultáneamente
            scalings='auto', # Ajuste automático de amplitud
            title=f'MNE Browser - {user}_{letter}_{capture_number}',
            show=True,
            block=True      # En MNE es mejor bloquear para interactuar con la ventana
        )
        
        return True

    except Exception as e:
        print(f"❌ Error MNE: {e}")
        return False


user_num = input("Número de usuario: ")
letra = input("Letra: ")
cap = input("Captura: ")
graficar_mne("User"+user_num, letra, cap)