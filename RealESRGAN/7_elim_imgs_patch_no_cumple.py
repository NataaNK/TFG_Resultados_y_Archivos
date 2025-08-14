import os
from PIL import Image

# Carpetas de imágenes
hr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\HR"
lr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\LR"

# Tamaño mínimo requerido para las imágenes LR
patch_size = 64

print("Revisando imágenes LR que son demasiado pequeñas:")

# Lista para llevar registro de los archivos eliminados
deleted_files = []

for filename in os.listdir(lr_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        lr_path = os.path.join(lr_folder, filename)
        try:
            with Image.open(lr_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Error al abrir {filename}: {e}")
            continue

        # Si la imagen LR es demasiado pequeña, se elimina
        if w < patch_size or h < patch_size:
            print(f"Eliminando {filename} (LR): {w}x{h}")
            try:
                os.remove(lr_path)
                deleted_files.append(filename)
            except Exception as e:
                print(f"Error al eliminar {filename} de LR: {e}")

# Ahora elimina las imágenes correspondientes en HR, asumiendo que tienen el mismo nombre
for filename in deleted_files:
    hr_path = os.path.join(hr_folder, filename)
    if os.path.exists(hr_path):
        try:
            os.remove(hr_path)
            print(f"Eliminado {filename} (HR) porque su par LR fue eliminado.")
        except Exception as e:
            print(f"Error al eliminar {filename} de HR: {e}")
    else:
        print(f"{filename} no se encontró en HR.")

print("Proceso completado.")
