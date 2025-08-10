from PIL import Image
import os

def resize_images(input_folder, output_folder, resolutions):
    # Crea la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Recorre todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Verifica si el archivo es una imagen
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Abre la imagen
            with Image.open(file_path) as img:
                # Redimensiona y guarda la imagen para cada resolución
                for width, height in resolutions:
                    resized_img = img.resize((width, height), Image.LANCZOS)
                    # Construye el nombre de archivo de salida con el sufijo de resolución
                    output_filename = f"{os.path.splitext(filename)[0]}_{width}x{height}{os.path.splitext(filename)[1]}"
                    output_path = os.path.join(output_folder, output_filename)
                    # Guarda la imagen redimensionada
                    resized_img.save(output_path)


input_folder = 'natalia/originals'
output_folder = 'natalia/resized'
resolutions = [
    (120, 160), (180, 240), (240, 320), (300, 400), (360, 480),  # Pequeñas
    (480, 640), (600, 800), (720, 960), (768, 1024), (900, 1200),  # Medianas
    (1080, 1440), (1200, 1600), (1440, 1920), (1536, 2048), (1600, 2560),  # HD
    (2160, 2880), (2400, 3200), (2880, 3840), (3200, 4000), (3840, 5120)  # Ultra altas
]


resize_images(input_folder, output_folder, resolutions)
