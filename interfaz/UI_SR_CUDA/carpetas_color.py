import os

# Constantes
BASE_FOLDER = "H:/MasOrange/troncho_portal/resultados_subdataset/WaveMixV2/COLOR"  # Cambia esta ruta a la carpeta deseada
PREFIX = "WaveMixV2"  # Cambia el prefijo según lo necesites

# Lista de sufijos para las carpetas
SUFFIXES = [
    "BlancoNegroBlur", "BlancoNegroCompressed", "BlancoNegroNoise", "BlancoNegroResized",
    "ColorBlur", "ColorCompressed", "ColorNoise", "ColorResized",
    "SepiaBlur", "SepiaCompressed", "SepiaNoise", "SepiaResized"
]

# Crear las carpetas
os.makedirs(BASE_FOLDER, exist_ok=True)  # Asegurar que la carpeta base existe

for suffix in SUFFIXES:
    folder_name = f"{PREFIX}_{suffix}"
    folder_path = os.path.join(BASE_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Carpeta creada: {folder_path}")
