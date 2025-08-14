import os
import argparse
from aura_sr import AuraSR
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Procesar imágenes con AuraSR")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio de entrada donde se encuentran las imágenes a procesar."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio de salida donde se guardarán las imágenes procesadas."
    )
    return parser.parse_args()

def load_image_from_path(path):
    """Cargar una imagen desde una ruta."""
    return Image.open(path)

def main():
    # Parsear argumentos
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar el modelo AuraSR
    print("Cargando modelo AuraSR...")
    aura_sr = AuraSR.from_pretrained(model_id="fal-ai/AuraSR", device="cuda")

    # Obtener lista de imágenes en el directorio de entrada
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if not image_files:
        print("No se encontraron imágenes en el directorio de entrada.")
        return

    # Procesar imágenes
    print(f"Procesando {len(image_files)} imágenes con AuraSR...")
    for idx, filename in enumerate(image_files, start=1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_x4.jpg")

        try:
            # Cargar imagen
            image = load_image_from_path(input_path).convert("RGB")

            # Procesar imagen con AuraSR
            print(f"[{idx}/{len(image_files)}] Procesando imagen: {filename}")
            upscaled_image = aura_sr.upscale_4x(image)

            # Guardar imagen procesada
            upscaled_image.save(output_path)
            print(f"Imagen procesada y guardada: {output_path}")
        except Exception as e:
            print(f"Error al procesar la imagen {filename}: {e}")

if __name__ == "__main__":
    main()
