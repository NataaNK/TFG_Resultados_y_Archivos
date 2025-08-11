import os
import numpy as np
from PIL import Image, ImageFilter

# =============================================================================
# CONSTANTE que puedes modificar con la ruta de tu directorio de imágenes
# =============================================================================
INPUT_DIR = r"H:/MasOrange/IMAGENES/Sub-datasets/COLOR/BlancoNegro/BlancoNegroOriginalResized"  # <-- MODIFICA ESTA RUTA

# =============================================================================
# Función para aplicar ruido gaussiano a una imagen PIL usando numpy
# =============================================================================
def add_gaussian_noise(pil_image, mean=0, var=10):
    """
    Agrega ruido gaussiano a la imagen.
    :param pil_image: Imagen tipo PIL
    :param mean: Media de la distribución normal
    :param var: Varianza de la distribución
    :return: Nueva imagen PIL con ruido gaussiano aplicado
    """
    # Convertir imagen a array numpy
    np_image = np.array(pil_image).astype(np.float32)
    
    # Generar ruido gaussiano
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, np_image.shape)
    
    # Agregar el ruido a la imagen
    noised_image = np_image + noise
    
    # Asegurarnos de que los valores estén en el rango [0, 255]
    noised_image = np.clip(noised_image, 0, 255)
    
    # Convertir de vuelta a uint8
    noised_image = noised_image.astype(np.uint8)
    
    # Crear una imagen PIL desde el array
    return Image.fromarray(noised_image)

def main():
    # =============================================================================
    # Verificar si la carpeta de entrada existe
    # =============================================================================
    if not os.path.isdir(INPUT_DIR):
        print(f"La ruta '{INPUT_DIR}' no es un directorio válido.")
        return
    
    # =============================================================================
    # Obtener el nombre base de la carpeta (por ejemplo "MiCarpeta" si la ruta es "/alguna/ruta/MiCarpeta")
    # =============================================================================
    base_dir_name = os.path.basename(os.path.normpath(INPUT_DIR))
    
    # =============================================================================
    # Construir las rutas de salida
    # =============================================================================
    output_dir_blur       = os.path.join(os.path.dirname(INPUT_DIR), base_dir_name + "Blur")
    output_dir_noise      = os.path.join(os.path.dirname(INPUT_DIR), base_dir_name + "Noise")
    output_dir_compression= os.path.join(os.path.dirname(INPUT_DIR), base_dir_name + "Compression")
    
    # Crear los directorios de salida si no existen
    os.makedirs(output_dir_blur, exist_ok=True)
    os.makedirs(output_dir_noise, exist_ok=True)
    os.makedirs(output_dir_compression, exist_ok=True)
    
    # =============================================================================
    # Extensiones válidas
    # =============================================================================
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    # =============================================================================
    # Recorrer todos los archivos del directorio de entrada
    # =============================================================================
    for filename in os.listdir(INPUT_DIR):
        # Verificar que sea una imagen con extensión válida
        if filename.lower().endswith(valid_extensions):
            # Construir la ruta completa al archivo
            file_path = os.path.join(INPUT_DIR, filename)
            
            # Abrir la imagen
            with Image.open(file_path) as img:
                # Asegurarnos de que la imagen está en modo RGB (para consistencia)
                img = img.convert("RGB")
                
                # =============================================================
                # 1) Desenfoque Gaussiano (Blur)
                # =============================================================
                # Aplicamos, por ejemplo, un blur de radio 2
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=1))
                
                # Guardar la imagen en la carpeta de blur
                # Se añade el sufijo "_blur" antes de la extensión
                name, ext = os.path.splitext(filename)
                blur_output_path = os.path.join(output_dir_blur, name + "_blur.jpg")
                
                # Guardamos en formato JPG (puedes ajustar calidad si deseas)
                blurred_img.save(blur_output_path, "JPEG", quality=90)
                
                # =============================================================
                # 2) Compresión JPEG
                # =============================================================
                # Guardamos la imagen con menor calidad para simular compresión
                # (puedes ajustar la calidad a tu gusto)
                compressed_output_path = os.path.join(output_dir_compression, name + "_compressed.jpg")
                img.save(compressed_output_path, "JPEG", quality=30)  # por ejem, calidad=30
                
                # =============================================================
                # 3) Ruido Gaussiano
                # =============================================================
                # Agregamos ruido gaussiano a la imagen
                noisy_img = add_gaussian_noise(img, mean=0, var=150)
                
                # Guardar la imagen con ruido
                noise_output_path = os.path.join(output_dir_noise, name + "_noise.jpg")
                noisy_img.save(noise_output_path, "JPEG", quality=90)
    
    print("Proceso finalizado. Se crearon y guardaron las imágenes deformadas en:")
    print(f" - {output_dir_blur}")
    print(f" - {output_dir_compression}")
    print(f" - {output_dir_noise}")

if __name__ == "__main__":
    main()
