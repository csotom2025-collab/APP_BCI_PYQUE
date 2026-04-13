# Configuración para ConecSerial.py
# Cambia estos valores según tus necesidades

# MODO DE CANALES
# True = 16 canales, False = 8 canales
USE_16_CHANNELS = True

# CONFIGURACIÓN SERIAL
SERIAL_PORT = 'COM6'
BAUD_RATE = 330400
TIMEOUT = 1

# CONFIGURACIÓN DE GRABACIÓN
UPDATE_INTERVAL = 100  # Actualizar gráficos cada N líneas
PLOT_HISTORY = 100     # Mantener últimos N puntos en el gráfico

# ARCHIVOS DE SALIDA
OUTPUT_DIR = ""  # Directorio de salida (vacío = directorio actual)
AUTO_TIMESTAMP = True  # Agregar timestamp al nombre del archivo