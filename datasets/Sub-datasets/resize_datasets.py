import os
from PIL import Image

# CONSTANTE QUE PUEDES MODIFICAR
INPUT_FOLDER = "H:/MasOrange/IMAGENES/Sub-datasets/COLOR/BlancoNegro/BlancoNegroOriginal"

def main():
    # Obtener el nombre base de la carpeta de entrada
    folder_name = os.path.basename(INPUT_FOLDER.rstrip(os.sep))
    
    # Construir el path de la carpeta de salida (por ejemplo, "MiCarpetaResized")
    parent_dir = os.path.dirname(INPUT_FOLDER.rstrip(os.sep))
    output_folder_name = folder_name + "Resized"
    OUTPUT_FOLDER = os.path.join(parent_dir, output_folder_name)
    
    # Crear la carpeta de salida si no existe
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    
    # Extensiones válidas
    valid_extensions = (".png", ".jpg", ".jpeg")
    
    # Listar los archivos de imagen en la carpeta de entrada
    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(valid_extensions)
    ]
    
    # Procesar cada imagen
    for image_file in image_files:
        input_path = os.path.join(INPUT_FOLDER, image_file)
        
        with Image.open(input_path) as img:
            # Obtener dimensiones originales
            width, height = img.size
            
            # Calcular dimensiones nuevas (1/4)
            new_width = width // 4
            new_height = height // 4
            
            # Redimensionar la imagen
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Separar nombre y extensión
            filename, extension = os.path.splitext(image_file)
            
            # Construir el nombre de salida con el sufijo "_resized"
            output_filename = f"{filename}_resized{extension}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Guardar la imagen
            resized_img.save(output_path)
            
            print(f"Imagen procesada: {image_file} -> {output_filename}")

if __name__ == "__main__":
    main()
