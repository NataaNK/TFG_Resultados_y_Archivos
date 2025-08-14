import os

# Directorios de entrenamiento
train_hr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\HR"
train_lr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\LR"

# Directorios de validación
val_hr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\HR_val"
val_lr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\LR_val"

# Archivos de salida para el meta info
output_file_train = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\meta_info_train.txt"
output_file_val = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\meta_info_val.txt"

# Extensiones aceptadas
extensions = ('.png', '.jpg', '.jpeg')

def get_pairs(hr_folder, lr_folder):
    """
    Busca en la carpeta HR los archivos de imagen y, para cada uno, intenta encontrar el archivo
    correspondiente en la carpeta LR comparando el nombre base (independientemente de la extensión).
    Devuelve una lista de tuplas (nombre_HR, nombre_LR).
    """
    pairs = []
    # Lista y ordena los archivos en HR
    for filename in sorted(os.listdir(hr_folder)):
        if not filename.lower().endswith(extensions):
            continue
        base = os.path.splitext(filename)[0]
        lr_file = None
        # Busca en LR un archivo que tenga el mismo nombre base con alguna extensión aceptada
        for ext in extensions:
            candidate = base + ext
            if os.path.exists(os.path.join(lr_folder, candidate)):
                lr_file = candidate
                break
        if lr_file is not None:
            pairs.append((filename, lr_file))
        else:
            print(f"Warning: No se encontró archivo LR correspondiente para {filename} en {lr_folder}")
    return pairs

# Obtén los pares para entrenamiento y validación
train_pairs = get_pairs(train_hr_folder, train_lr_folder)
val_pairs = get_pairs(val_hr_folder, val_lr_folder)

# Escribe el meta info para entrenamiento
with open(output_file_train, 'w') as f:
    for hr, lr in train_pairs:
        f.write(f"{hr}, {lr}\n")
print("Archivo meta_info_train.txt generado con éxito.")

# Escribe el meta info para validación
with open(output_file_val, 'w') as f:
    for hr, lr in val_pairs:
        f.write(f"{hr}, {lr}\n")
print("Archivo meta_info_val.txt generado con éxito.")
