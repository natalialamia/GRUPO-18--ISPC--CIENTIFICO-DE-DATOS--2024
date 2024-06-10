import os
import mysql.connector
from PIL import Image

# Conectar a la base de datos MySQL
conexion = mysql.connector.connect(
    host="localhost",
    user="root",
    password="mariano",
    database="recomendaciones"
)
cursor = conexion.cursor()

# Crear la tabla "ciudades"
cursor.execute("""
    CREATE TABLE IF NOT EXISTS mar (
        id_ciudad INT AUTO_INCREMENT PRIMARY KEY,
        nombre VARCHAR(255),
        ubicacion VARCHAR(255),
        clima VARCHAR(255),
        paisajes LONGBLOB
    )
""")

# Ruta de la carpeta con imágenes
ruta_imagenes = 'C:\\Users\\Usuario\\Desktop\\archivos\\images\\zonas\\mar'

# Itera sobre las imágenes en la carpeta
for filename in os.listdir(ruta_imagenes):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Abre la imagen
        ruta_imagen = os.path.join(ruta_imagenes, filename)
        imagen = Image.open(ruta_imagen)

        # Obtiene información de la imagen (nombre, ubicación, clima)
        nombre_ciudad = filename.split('_')[0]
        ubicacion_ciudad = 'Ubicación de ' + nombre_ciudad
        clima_ciudad = 'Clima de ' + nombre_ciudad

        # Guarda la imagen en un objeto bytes
        imagen_bytes = None
        with open(ruta_imagen, 'rb') as file:
            imagen_bytes = file.read()

        # Inserta un nuevo registro en la tabla "ciudades"
        sql = "INSERT INTO mar (nombre, ubicacion, clima, paisajes) VALUES (%s, %s, %s, %s)"
        valores = (nombre_ciudad, ubicacion_ciudad, clima_ciudad, imagen_bytes)
        cursor.execute(sql, valores)

# Commit y cierra la conexión
conexion.commit()
cursor.close()
conexion.close()