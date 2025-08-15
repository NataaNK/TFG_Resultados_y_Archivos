import os
import re

# Ruta a la carpeta que contiene los archivos .txt
folder_path = r"H:\MasOrange\IMAGENES\AAOptimizacion\SalidasConTexto100\4_1_piqe_registro_optuna_ConTexto100\salidas_optuna_contexto100"

# Inicializa el contador de tiempo total
total_seconds = 0.0

# Expresión regular para extraer el número de segundos
time_pattern = re.compile(r"Tiempo de ejecución:\s*([\d.]+)\s*segundos")

# Recorre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                for line in file:
                    match = time_pattern.search(line)
                    if match:
                        total_seconds += float(match.group(1))
                        break  # Asumimos solo una línea con tiempo por archivo
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")

# Imprime el total de segundos
print(f"Tiempo total de ejecución: {total_seconds:.2f} segundos")
