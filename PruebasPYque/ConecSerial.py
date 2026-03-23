# Programa para conectar con un puerto serial y leer datos
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import numpy as np

# Crear un nuevo archivo CSV para almacenar los datos
datos = open('datosLectura.csv', 'w', encoding='utf-8')
datos.write("Tm,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n")

# prepare an in-memory DataFrame for plotting (avoids re-reading disk)
df = pd.DataFrame(columns=["Tm","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"])

# Configuración del puerto serial
ser = serial.Serial('COM3', 230400, timeout=1)  # Ajusta el puerto y baudrate según tu configuración
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

# Crear figura con subplots usando Matplotlib (4 filas, 2 columnas)
channels = ["ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8"]
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('Señales EEG por Canal - SignalTest (Tiempo Real)')
axes = axes.flatten()  # Flatten para acceder más fácil

# Control para cerrar cuando se cierre la ventana
running = True
def on_close(event):
    global running
    running = False

fig.canvas.mpl_connect('close_event', on_close)

# Activar modo interactivo
plt.ion()
plt.show()
print("Abriendo gráfico - las gráficas se actualizarán en tiempo real")

update_count = 0
try:
    while running:
        if ser.in_waiting > 0:
            raw = ser.readline()
            line = raw.decode('utf-8', errors='replace').strip()
            # eliminar coma final extra si existe
            if line.endswith(','):
                line = line[:-1]
            values = line.split(",")
            if len(values) == 9:  # header has 9 columns (Tm + 8 canales)
                print(f"Datos recibidos: {line}")
                datos.write(f"{line}\n")
                datos.flush()

            # append the new row to a growing DataFrame instead of re-reading the file every loop
            try:
                if len(values) == 9:  # header has 9 columns (Tm + 8 canales)
                    # convert every value to float, fallback to NaN
                    row = []
                    for v in values:
                        try:
                            row.append(float(v))
                        except Exception:
                            row.append(np.nan)
                    # ensure correct length and append
                    if len(row) == len(df.columns):
                        df.loc[len(df)] = row
                    else:
                        continue
                else:
                    # skip malformed line
                    continue
            except Exception:
                # if conversion fails, ignore the row
                continue

            # Actualizar gráficos con Matplotlib - mostrar últimos 100 samples para mejor rendimiento
            update_count += 1
            if update_count % 100 == 0:  # Actualizar cada 100 líneas para mejor rendimiento
                df_plot = df.tail(100)  # keep only recent points
                for idx, channel in enumerate(channels):
                    axes[idx].clear()
                    axes[idx].plot(df_plot['Tm'].values, df_plot[channel].values, 'b-', linewidth=1)
                    axes[idx].set_title(f'Canal: {channel}')
                    axes[idx].set_xlabel('Tiempo')
                    axes[idx].set_ylabel('Amplitud')
                    axes[idx].grid(True, alpha=0.3)
                    # establish y limits with a small margin so constant values are still visible
                    # convert to float in case dtype is object
                    ymin = float(df_plot[channel].min())
                    ymax = float(df_plot[channel].max())
                    if ymax - ymin < 1e-6:  # nearly constant
                        mid = (ymin + ymax) / 2
                        axes[idx].set_ylim(mid - 0.5, mid + 0.5)
                    else:
                        margin = (ymax - ymin) * 0.1  # 10% margin
                        axes[idx].set_ylim(ymin - margin, ymax + margin)
                    # x limits
                    xmin = float(df_plot['Tm'].min())
                    xmax = float(df_plot['Tm'].max())
                    if xmax > xmin:
                        axes[idx].set_xlim(xmin, xmax)
                
                plt.tight_layout()
                plt.pause(0.01)  # Pequeña pausa para actualizar la ventana

except KeyboardInterrupt:
    print("Interrupción del usuario. Cerrando el puerto serial.")
finally:
    datos.close()
    ser.close()
    print("Conexión cerrada.")

