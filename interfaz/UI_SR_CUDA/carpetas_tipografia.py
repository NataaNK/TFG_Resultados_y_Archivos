import os

# Constantes
BASE_FOLDER = r"H:\MasOrange\troncho_portal\resultados_subdataset\WaveMix\TIPOGRAFIA"  # Cambia esta ruta a la carpeta deseada
PREFIX = "WaveMix"  # Cambia el prefijo según lo necesites

# Lista de sufijos para las carpetas
SUFFIXES = [
    "CartoonBlur", "CartoonCompressed", "CartoonNoise", "CartoonResized",
    "FuturistasBlur", "FuturistasCompressed", "FuturistasNoise", "FuturistasResized",
    "ManuscritasBlur", "ManuscritasCompressed", "ManuscritasNoise", "ManuscritasResized",
    "MaquinaEscribirBlur", "MaquinaEscribirCompressed", "MaquinaEscribirNoise", "MaquinaEscribirResized",
    "PsicodelicasBlur", "PsicodelicasCompressed", "PsicodelicasNoise", "PsicodelicasResized",
    "SansSerifGruesasBlur", "SansSerifGruesasCompressed", "SansSerifGruesasNoise", "SansSerifGruesasResized",
    "SerifBlur", "SerifCompressed", "SerifNoise", "SerifResized"
]

# Crear las carpetas
os.makedirs(BASE_FOLDER, exist_ok=True)  # Asegurar que la carpeta base existe

for suffix in SUFFIXES:
    folder_name = f"{PREFIX}_{suffix}"
    folder_path = os.path.join(BASE_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Carpeta creada: {folder_path}")
