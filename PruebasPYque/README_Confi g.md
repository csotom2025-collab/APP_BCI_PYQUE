# Sistema de Configuración para ConecSerial.py

## Descripción
Este sistema permite cambiar fácilmente entre modos de 8 y 16 canales sin modificar el código principal.

## Archivo de Configuración: config.py

### Variables Principales
- `USE_16_CHANNELS`: Booleano para seleccionar modo (True=16 canales, False=8 canales)
- `SERIAL_PORT`: Puerto serial (ej: 'COM3', '/dev/ttyUSB0')
- `BAUD_RATE`: Velocidad de comunicación (ej: 230400, 115200)
- `TIMEOUT`: Timeout para lectura serial

### Configuración de Gráficos
- `UPDATE_INTERVAL`: Actualizar gráficos cada N líneas (mejor rendimiento)
- `PLOT_HISTORY`: Número de puntos recientes a mostrar en gráficos

### Configuración de Archivos
- `OUTPUT_DIR`: Directorio de salida (vacío = directorio actual)
- `AUTO_TIMESTAMP`: Agregar timestamp al nombre del archivo

## Uso

### Cambiar entre 8 y 16 canales:
```python
# En config.py
USE_16_CHANNELS = True   # Para 16 canales
USE_16_CHANNELS = False  # Para 8 canales
```

### Configuración personalizada:
```python
# Puerto serial diferente
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Gráficos más responsivos
UPDATE_INTERVAL = 50
PLOT_HISTORY = 200
```

## Archivos de Salida

### Modo 8 canales:
- `datosLectura_8ch.csv` o `datosLectura_8ch_20231201_143022.csv`

### Modo 16 canales:
- `datosLectura_16ch.csv` o `datosLectura_16ch_20231201_143022.csv`

## Layout de Gráficos

- **8 canales**: 4 filas × 2 columnas
- **16 canales**: 4 filas × 4 columnas

## Notas Técnicas

- El sistema detecta automáticamente el número de canales esperado
- Los headers CSV se generan dinámicamente
- Los gráficos se adaptan automáticamente al número de canales
- Se incluye validación de datos para evitar errores