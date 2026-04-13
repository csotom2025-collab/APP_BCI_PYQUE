# Ejemplos de configuración para diferentes modos

# MODO 8 CANALES (por defecto)
USE_16_CHANNELS = False

# MODO 16 CANALES
# USE_16_CHANNELS = True

# CONFIGURACIONES DE EJEMPLO:

# Para hardware rápido (alta frecuencia de muestreo)
# USE_16_CHANNELS = True
# UPDATE_INTERVAL = 50
# PLOT_HISTORY = 200

# Para hardware lento (baja frecuencia de muestreo)
# USE_16_CHANNELS = False
# UPDATE_INTERVAL = 200
# PLOT_HISTORY = 50

# Para debugging (más información)
# USE_16_CHANNELS = False
# UPDATE_INTERVAL = 10
# PLOT_HISTORY = 500

# Para producción (optimizado)
# USE_16_CHANNELS = True
# UPDATE_INTERVAL = 100
# PLOT_HISTORY = 100
# AUTO_TIMESTAMP = True
# OUTPUT_DIR = "datos_produccion"