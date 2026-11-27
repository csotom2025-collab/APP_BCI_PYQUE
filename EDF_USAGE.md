# Guía de Uso - Formato EDF

## Instalación
La librería **MNE** ha sido instalada satisfactoriamente. Para guardar datos en formato EDF, se utilizará MNE en lugar de pyedflib (que requería compilador C++).

## Características Agregadas

### 1. `save_capture_edf()` - Guardar captura en EDF
Guarda los datos capturados en formato EDF (European Data Format) usando MNE-Python.

**Características:**
- Convierte datos del DataFrame a formato EDF
- Automatiza la creación de canales con información de EEG
- Frecuencia de muestreo: 256 Hz (configurable en el código)
- Crea directorios automáticamente si no existen

### 2. `start_capture_edf()` - Iniciar captura con guardado automático en EDF
Similar a `start_capture()` pero guarda automáticamente en formato EDF.

**Parámetros:**
- `user` (str): Nombre del usuario (ej: "User0")
- `character_type` (str): Tipo de carácter (ej: "Letters", "Numbers", "Controls")
- `character` (str): Carácter específico (ej: "A", "1", "ESP")
- `duration` (int): Duración de la captura en segundos

**Ejemplo de uso:**
```python
controller = controllerSaveCapture(serial_monitor)
controller.start_capture_edf("User0", "Letters", "A", 5)  # Captura 5 segundos y guarda como EDF
```

### 3. Almacenamiento automático en ambos formatos
Si usas `start_capture()`, los datos se guardan en CSV.
Si usas `start_capture_edf()`, los datos se guardan en EDF.

## Funciones de Limpieza de Datos

### `clean_df_file(df)` - Limpieza automática
Elimina decimales innecesarios (.0) de los DataFrames:
- ✅ Convierte `1.0`, `2.0`, `3.0` → `1`, `2`, `3`
- ✅ Mantiene decimales reales como `1.23`, `4.56`
- ✅ Aplica automáticamente en guardado CSV y EDF

### `clean_existing_file(file_path)` - Limpiar archivo existente
```python
controller = controllerSaveCapture(serial_monitor)
controller.clean_existing_file("captures/User0/Letters/User0_A_0.csv")
```

### `clean_all_captures(user=None)` - Limpiar todos los archivos
```python
# Limpiar todos los archivos
controller.clean_all_captures()

# Limpiar solo archivos de un usuario
controller.clean_all_captures("User0")
```

## Estructura de archivos después de limpieza:
```
captures/
├── User0/
│   ├── Letters/
│   │   ├── User0_A_0.csv    # Ahora sin .0
│   │   └── User0_A_0.edf
```

## Especificaciones EDF

- **Formato:** European Data Format (EEG estándar)
- **Tipo de canal:** EEG (automático)
- **Frecuencia de muestreo:** 256 Hz
- **Nombres de canales:** Se toman del CSV original (máximo 16 caracteres)
- **Escala de datos:** Automáticamente convertida a microvoltios (µV)
- **Límite de valores:** ±1,000,000 µV (para compatibilidad EDF)
- **Respaldo automático:** Si EDF falla, se guarda como CSV

## Normalización Automática

El código automáticamente:
- ✅ Detecta y limpia valores NaN e infinitos
- ✅ Escala datos extremadamente grandes (>1M) a rango típico de EEG (1000 µV)
- ✅ Convierte volts a microvoltios cuando es necesario
- ✅ Amplifica datos muy pequeños (<1mV) para mejor resolución
- ✅ Limita valores a ±30,000 µV para compatibilidad EDF
- ✅ Acorta nombres de canales si exceden 16 caracteres
- ✅ Usa float32 para mejor compatibilidad
- ✅ Proporciona respaldo CSV si EDF falla
- ✅ Muestra estadísticas de depuración durante el proceso

## Configuración Personalizada

Para cambiar la frecuencia de muestreo, edita el valor `sfreq` en `save_capture_edf()`:

```python
sfreq = 256  # Cambiar este valor según tu hardware
```

## Requisitos

- pandas
- numpy
- mne
- PyQt6

Todas las librerías están instaladas en el ambiente virtual.

## Ventajas del Formato EDF

- ✅ Compatibilidad con software de análisis de EEG
- ✅ Estándar internacional para datos biomédicos
- ✅ Soporte en herramientas como EEGLAB, Brainstorm
- ✅ Preserva metadatos de canales y frecuencia de muestreo
- ✅ Mejor compresión que CSV para grandes volúmenes
