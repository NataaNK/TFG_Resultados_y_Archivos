import os
from PIL import Image

# Ruta de entrada y salida
input_path = r"E:\MasOrange\caratulas24\caratulas24_hd"
output_path = r"E:\MasOrange\caratulas24\caratulas24_lr"

# Crear la carpeta de salida si no existe
os.makedirs(output_path, exist_ok=True)

def downscale_image(image, output_file):
    # Reducción de resolución por un factor de 4
    lr_image = image.resize((image.width // 4, image.height // 4), Image.Resampling.LANCZOS)
    # Guardar la imagen reescalada
    lr_image.save(output_file, "JPEG", quality=90)

def process_images(input_path, output_path):
    for file_name in os.listdir(input_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            input_file = os.path.join(input_path, file_name)
            output_file = os.path.join(output_path, file_name)

            # Abrir la imagen
            with Image.open(input_file) as img:
                downscale_image(img, output_file)

# Ejecutar el procesamiento
process_images(input_path, output_path)

print(f"Procesamiento completo. Las imágenes reescaladas se guardaron en: {output_path}")
