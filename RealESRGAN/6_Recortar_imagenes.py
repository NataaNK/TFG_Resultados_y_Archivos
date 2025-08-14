import os
from PIL import Image

# Factor de escala esperado
scale = 4

def center_crop(img, new_width, new_height):
    """Realiza un recorte centrado a las dimensiones dadas."""
    width, height = img.size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

def process_and_overwrite_folder(folder):
    """
    Procesa todas las imágenes en 'folder':
      - Calcula las nuevas dimensiones: (floor(width/scale)*scale, floor(height/scale)*scale)
      - Si no cumplen, recorta la imagen de forma centrada
      - Sobrescribe el archivo original.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for filename in os.listdir(folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        filepath = os.path.join(folder, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                new_width = (width // scale) * scale
                new_height = (height // scale) * scale

                if new_width != width or new_height != height:
                    cropped_img = center_crop(img, new_width, new_height)
                    cropped_img.save(filepath)
                    print(f"Recortada {filename}: {width}x{height} -> {new_width}x{new_height}")
                else:
                    print(f"{filename} ya cumple con las dimensiones ({width}x{height}).")
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

if __name__ == '__main__':
    # Carpetas de imágenes HR para entrenamiento y validación
    train_hr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\HR"
    val_hr_folder = r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\HR_val"

    print("Procesando carpeta de entrenamiento HR (sobrescribiendo)...")
    process_and_overwrite_folder(train_hr_folder)

    print("\nProcesando carpeta de validación HR_val (sobrescribiendo)...")
    process_and_overwrite_folder(val_hr_folder)

    print("\nProceso completado.")
