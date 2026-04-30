from pathlib import Path
from EEdfReader import EEGEDFReader as eeg_reader

 # Listar archivos (ejemplo buscando sufijo '0')
#'Digit':['0','1','2','3','4','5','6','7','8','9'],
#'Char':['A','C','F','H','J','M','P','S','T','Y']
#'Controls':["───", "⟵", "↩"]       


user = "User3"
letter = "B"
type = "Letters"
file = f'captures/{user}/{type}/{user}_{letter}_0.edf'
#canales=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
canales = ["Oz","Po7","Po4","Po3","P4","P3","Po8","Pz","Fz","F2","F3","F4","AF3","Cz","AF4","F1"]
results_dir = Path('results') / user / type 
results_dir.mkdir(parents=True, exist_ok=True)
print(f"\nLeyendo archivo: {file}")
reader = eeg_reader(data_path='.')
eeg_data = reader.read_edf_file(filename=file)

if eeg_data is not None:
    metadata = eeg_data['metadata']
    print(f"\nInformación del archivo:")
    print(f"  - Canales: {metadata['n_channels']}")
    print(f"  - Duración: {metadata['duration']:.2f} segundos")
    print(f"  - Frecuencia de muestreo: {metadata['sampling_rate']} Hz")
    print(f"  - Etiquetas de canales: {metadata['signal_labels']}")
    print (f" - Número de muestras por canal: {eeg_data['signals'].shape[1]}")
    print(f"  - Muestras ejemplo (primer canal): {eeg_data['signals'][0, :10]}")
    
    # Preprocesar señales
    filtered_signals = reader.preprocess_eeg(signals=eeg_data['signals'], fs=metadata.get('sampling_rate'))
    eeg_data['filtered_signals'] = filtered_signals
    # Aplicar CAR (Common Average Reference)
    print("\nAplicando CAR (Common Average Reference)...")
    car_signals = reader.aplicar_car(filtered_signals)
    eeg_data['car_signals'] = car_signals
    print("CAR aplicado exitosamente")
    
    print(f"Extrayendo características...")                
    features_df = reader.extract_features(car_signals,
                                            channel_names=canales,
                                            available_channel_names=eeg_data['channel_names'],
                                            window_size=256, overlap=0.5)
    print(f"Características extraídas: {features_df.shape}")
    print(features_df.head())
    eeg_data['filtered_signals'] = car_signals  # actualizar para plots
    print("Generando gráficas (tiempo + PSD + bandas)...")
    reader.plot_channels_and_spectra(eeg_data, channel_names=canales,
                                        time_window=(0, 5), save=True, show=False,
                                        features_df=features_df, window_size=250, overlap=0.5)
    print("Generando gráficas de características por canal")
    reader.plot_feactureres(features_df, canales,filename=Path(str(file)).stem,
                                    window_size=256, overlap=0.5,
                                    out_dir=results_dir, save=True, show=False)
    print("Generando gráficas de espectrogramas por banda")
    bands = {
        'Rango interes': (13, 40)
    }
    for ch in canales:
        if ch not in eeg_data['channel_names']:
            print(f"Canal {ch} no encontrado, se omite espectrograma.")
            continue
        ch_idx = eeg_data['channel_names'].index(ch)
        ch_signal = filtered_signals[ch_idx, :]
        for bname, band in bands.items():
            print(f"  - Canal {ch}, en {bname}")
            reader.plot_espectrograma_banda(ch_signal, fs=metadata.get('sampling_rate'),
                                                band=band,bandName=bname, channel_name=ch,filename=Path(str(file)).stem,
                                                nperseg=256, noverlap=128, 
                                                out_dir=results_dir, save=True, show=False)
    print("Generando gráficas de potencias de banda")
    reader.plot_band_powers(features_df, eeg_data, canales,window_size=256, overlap=0.5,
                                out_dir=results_dir, save=True, show=False)
    print("podencias relativas de banda para el canal F3")
    reader.plot_power_relative_bandas(features_df, canales,filename=Path(str(file)).stem,
                                        window_size=256, overlap=0.5,
                                        out_dir=results_dir, save=True, show=False)
    output_filename = Path(str(file)).stem + '_features.csv'
    output_path = results_dir / output_filename
    features_df.to_csv(output_path, index=False)
    print(f"Características guardadas en: {output_path}") 
        