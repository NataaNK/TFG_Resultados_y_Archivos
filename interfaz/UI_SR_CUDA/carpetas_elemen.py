import os

# Constantes
BASE_FOLDER = r"H:\MasOrange\troncho_portal\resultados_subdataset\WaveMix\ELEMENTOS"  # Cambia esta ruta a la carpeta deseada
PREFIX = "WaveMix"  # Cambia el prefijo según lo necesites

# Lista de sufijos para las carpetas
SUFFIXES = [
    "DibujoBlur", "DibujoCompressed", "DibujoNoise", "DibujoResized",
    "ElemCartoonBlur", "ElemCartoonCompressed", "ElemCartoonNoise", "ElemCartoonResized",
    "FondosCargadosBlur", "FondosCargadosCompressed", "FondosCargadosNoise", "FondosCargadosResized",
    "MinimalistaBlur", "MinimalistaCompressed", "MinimalistaNoise", "MinimalistaResized",
    "RetratoBlur", "RetratoCompressed", "RetratoNoise", "RetratoResized"
]

# Crear las carpetas
os.makedirs(BASE_FOLDER, exist_ok=True)  # Asegurar que la carpeta base existe

for suffix in SUFFIXES:
    folder_name = f"{PREFIX}_{suffix}"
    folder_path = os.path.join(BASE_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Carpeta creada: {folder_path}")
