# Programa para conectar con un puerto serial y leer datos
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import numpy as np
from config import USE_16_CHANNELS, SERIAL_PORT, BAUD_RATE, TIMEOUT, UPDATE_INTERVAL, PLOT_HISTORY, OUTPUT_DIR, AUTO_TIMESTAMP

# Determinar número de canales basado en la configuración
NUM_CHANNELS = 16 if USE_16_CHANNELS else 8

# Crear headers dinámicamente
channel_headers = ["Tm"] + [f"ch{i+1}" for i in range(NUM_CHANNELS)]
header_str = ",".join(channel_headers)

# Crear nombre de archivo con timestamp opcional
timestamp = time.strftime("%Y%m%d_%H%M%S") if AUTO_TIMESTAMP else ""
filename_base = f'datosLectura_{NUM_CHANNELS}ch'
filename = f'{filename_base}_{timestamp}.csv' if timestamp else f'{filename_base}.csv'
if OUTPUT_DIR:
    filename = f'{OUTPUT_DIR}/{filename}'

# Crear un nuevo archivo CSV para almacenar los datos
datos = open(filename, 'w', encoding='utf-8')
datos.write(header_str + "\n")

# prepare an in-memory DataFrame for plotting (avoids re-reading disk)
df = pd.DataFrame(columns=channel_headers)

# Configuración del puerto serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)  # Usar configuración del archivo config.py
time.sleep(2)  # Espera a que el puerto se estabilice
ser.flush()
# Limpiar buffers de entrada/salida para evitar datos residuales
try:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
except Exception:
    # Algunas versiones/pkgs pueden no implementar reset_*; ignore si falla
    pass
time.sleep(0.05)
# Desechar cualquier línea parcial que haya quedado en el buffer
while ser.in_waiting:
    _ = ser.readline()
# Enviar el comando inicial para comenzar la transmisión de datos
ini = 'x'
ser.write(ini.encode('utf-8'))

# Crear figura con subplots usando Matplotlib (adaptable al número de canales)
channels = [f"ch{i+1}" for i in range(NUM_CHANNELS)]

# Determinar layout de subplots basado en número de canales
if NUM_CHANNELS == 8:
    rows, cols = 4, 2
    figsize = (14, 10)
elif NUM_CHANNELS == 16:
    rows, cols = 4, 4
    figsize = (16, 12)
else:
    # Layout genérico para otros números
    cols = min(4, NUM_CHANNELS)
    rows = (NUM_CHANNELS + cols - 1) // cols  # Ceiling division
    figsize = (4 * cols, 3 * rows)

fig, axes = plt.subplots(rows, cols, figsize=figsize)
fig.suptitle(f'Señales EEG por Canal - SignalTest (Tiempo Real) - {NUM_CHANNELS} Canales')

# Aplanar axes para acceso fácil, incluso si es 1D
if NUM_CHANNELS == 1:
    axes = [axes]
else:
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

# Si hay más subplots que canales, ocultar los extras
for i in range(len(axes)):
    if i >= NUM_CHANNELS:
        axes[i].set_visible(False)

# Control para cerrar cuando se cierre la ventana
running = True
def on_close(event):
    global running
    running = False

fig.canvas.mpl_connect('close_event', on_close)

# Activar modo interactivo
plt.ion()
plt.show()

update_count = 0
# --- NUEVAS VARIABLES PARA FRECUENCIA ---
sample_counter = 0
start_time_fs = time.time()
fs_real = 0
# ----------------------------------------

print(f"Abriendo gráfico - Modo {NUM_CHANNELS} canales")
print(f"Archivo de salida: {filename}")
print("Presiona Ctrl+C para detener...")

try:
    while running:
        if ser.in_waiting > 0:
            raw = ser.readline()
            line = raw.decode('utf-8', errors='replace').strip()
            if line.endswith(','):
                line = line[:-1]
            values = line.split(",")
            expected_values = NUM_CHANNELS + 1 
            
            if len(values) == expected_values:
                # 1. Guardar en archivo
                datos.write(f"{line}\n")
                
                # 2. CONTADOR DE FRECUENCIA
                sample_counter += 1
                current_time = time.time()
                elapsed = current_time - start_time_fs
                
                if elapsed >= 1.0:  # Cada segundo calculamos los Hz
                    fs_real = sample_counter / elapsed
                    print(f">>> Frecuencia de Muestreo Real: {fs_real:.2f} Hz")
                    sample_counter = 0
                    start_time_fs = current_time

                # 3. Procesar para el DataFrame
                try:
                    row = [float(v) if v.strip() else np.nan for v in values]
                    df.loc[len(df)] = row
                except Exception:
                    continue

            # Actualizar gráficos cada N muestras
            update_count += 1
            if update_count % UPDATE_INTERVAL == 0:
                # Opcional: Flush al archivo solo al actualizar gráfico para no saturar el disco
                datos.flush()
                
                df_plot = df.tail(PLOT_HISTORY)
                for idx, channel in enumerate(channels):
                    axes[idx].clear()
                    # Agregamos el FS al título del primer gráfico para verlo en la ventana
                    title_extra = f" ({fs_real:.1f} Hz)" if idx == 0 else ""
                    axes[idx].plot(df_plot['Tm'].values, df_plot[channel].values, 'b-', linewidth=1)
                    axes[idx].set_title(f'{channel}{title_extra}')
                    # ... (resto de tu configuración de ejes y grid)
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Límites de ejes (tu lógica original)
                    ymin, ymax = float(df_plot[channel].min()), float(df_plot[channel].max())
                    if ymax - ymin < 1e-6:
                        axes[idx].set_ylim(ymin - 0.5, ymax + 0.5)
                    else:
                        margin = (ymax - ymin) * 0.1
                        axes[idx].set_ylim(ymin - margin, ymax + margin)

                plt.tight_layout()
                plt.pause(0.001) # Pausa mínima para no bloquear la CPU

except KeyboardInterrupt:
    print("\nInterrupción del usuario.")
# ... (resto del finally)finally:
    datos.close()
    ser.close()
    print("Conexión cerrada.")
    
# Función de utilidad para mostrar configuración
def print_config_info():
    """Muestra información sobre la configuración actual"""
    print("\n" + "="*60)
    print("CONFIGURACIÓN ACTUAL:")
    print(f"Modo: {'16 canales' if USE_16_CHANNELS else '8 canales'}")
    print(f"Número de canales: {NUM_CHANNELS}")
    print(f"Puerto serial: {SERIAL_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Archivo de salida: {filename}")
    print(f"Headers: {header_str}")
    print(f"Layout de gráficos: {rows}x{cols}")
    print(f"Intervalo de actualización: cada {UPDATE_INTERVAL} muestras")
    print(f"Historial de gráfico: {PLOT_HISTORY} puntos")
    print(f"Directorio de salida: {OUTPUT_DIR if OUTPUT_DIR else 'actual'}")
    print(f"Timestamp automático: {'Sí' if AUTO_TIMESTAMP else 'No'}")
    print("="*60)
    
# Mostrar configuración al inicio
print_config_info()

