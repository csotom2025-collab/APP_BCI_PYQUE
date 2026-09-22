import os
ruta_carpeta= "captures/UserJorge"
subcarpetas = os.listdir(ruta_carpeta)
print(subcarpetas)
nuevoNombre = "Jorge"
subcarpetas = ["Letters"]
for subcarpeta in subcarpetas:
    path= os.path.join(ruta_carpeta, subcarpeta)
    print(path)
    for nombre_archivo in os.listdir(path):
        ##RENOnmbrar archivo
        nombre_base, extension = os.path.splitext(nombre_archivo)
        extension = extension[1:]  # Eliminar el punto inicial de la extensión
        nombre_base = nombre_base.split("_")
        letra = nombre_base[1]
        trial = nombre_base[2]
        nuevo_nombre = f"User{nuevoNombre}_{letra}_{trial}.{extension}"
        ruta_file = os.path.join(ruta_carpeta, subcarpeta, nombre_archivo)
        print(ruta_file,nuevo_nombre)
        os.rename(ruta_file, os.path.join(ruta_carpeta, subcarpeta, nuevo_nombre))