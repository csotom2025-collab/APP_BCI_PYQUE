import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo CSV

user = "User2"
letter = "N"
df = pd.read_csv(f'captures/{user}/Letters/{user}_{letter}_1.csv')
#df = pd.read_csv('EEGTestClean.csv')
# Mostrar las primeras filas para verificar
print(df.head())
print(f"\nforma del dataset: {df.shape}")
print(f"Canales disponibles: {df.columns.tolist()}")

# Crear gráficas por canal
fig, axes = plt.subplots(8, 2, figsize=(14, 10))
fig.suptitle('Señales EEG por Canal - SignalTest', fontsize=16)

# Aplanar el array de axes para iterar fácilmente
axes = axes.flatten()

df['Tm'] = [i for i in range(len(df))]
# Plotear cada canal con Time en el eje X
channels = [col for col in df.columns if col != 'Tm']
for idx, column in enumerate(channels):
    print(idx,column)
    axes[idx].plot(df['Tm'], df[column], linewidth=0.8, color='blue')
    axes[idx].set_title(f'Canal: {column}')
    axes[idx].set_xlabel('Tiempo (muestra)')
    axes[idx].set_ylabel('Amplitud')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
