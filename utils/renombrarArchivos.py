import os
import pandas as pd
ruta_carpeta= "captures/UserJorge"
subcarpetas = os.listdir(ruta_carpeta)
print(subcarpetas)
nuevoNombre = "Jorge"
subcarpetas = ["Letters"]
for subcarpeta in subcarpetas:
    path= os.path.join(ruta_carpeta, subcarpeta)
    print(path)
    for nombre_archivo in os.listdir(path):
        # 1. Separar nombre y extensión
        nombre_base, extension = os.path.splitext(nombre_archivo)
        extension = extension[1:]  # Eliminar el punto inicial

        # 2. Extraer partes del nombre original
        nombre_base = nombre_base.split("_")
        letra = nombre_base[1]
        trial = nombre_base[2]

        # 3. Definir el nuevo nombre y las rutas
        nuevo_nombre = f"User{nuevoNombre}_{letra}_{trial}.{extension}"
        ruta_original = os.path.join(ruta_carpeta, subcarpeta, nombre_archivo)
        ruta_nueva = os.path.join(ruta_carpeta, subcarpeta, nuevo_nombre)

        # 4. Definir las columnas
        headers = ['Tm', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']

        # 5. Leer el archivo usando la ruta original antes de renombrarlo
        # Usamos 'names' para asignar los encabezados correctamente
        archivo = pd.read_csv(ruta_original, names=headers, header=0, sep=',')

        # 6. Modificar la columna 'Tm' con el rango de 1 al total de filas
        archivo['Tm'] = range(1, len(archivo) + 1)

        # 7. Guardar el archivo directamente en la NUEVA ruta con el nuevo nombre
        archivo.to_csv(ruta_nueva, index=False)

        # 8. Eliminar el archivo original si no quieres dejar un duplicado
        os.remove(ruta_original)