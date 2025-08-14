import os
import subprocess
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Procesar imágenes con Real-ESRGAN.")
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

def main():
    # Parsear argumentos
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Parámetros del comando
    model = "RealESRGAN_x4plus"
    scale_factor = 4.0  # Factor de escala
    suffix = "_x4"

    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar las imágenes en la carpeta
    images = sorted(os.listdir(input_dir))  # Ordenar para procesarlas en orden

    for image in images:
        # Procesar solo archivos con extensiones válidas
        if image.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Ruta completa del archivo de entrada
            input_path = os.path.join(input_dir, image)

            # Generar nombre de archivo de salida
            base_name, ext = os.path.splitext(image)
            output_name = f"{base_name}{suffix}{ext}"
            output_path = os.path.join(output_dir, output_name)

            # Construir el comando
            command = [
                sys.executable, "inference_realesrgan.py",
                "-n", model,
                "-i", input_path,
                "-o", output_dir,
                "--fp32",
                "--outscale", str(scale_factor)
            ]

            # Ejecutar el comando
            try:
                print(f"Procesando: {image} -> {output_name}")
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error al procesar {image}: {e}")

    print("Procesamiento completo de todas las imágenes.")

if __name__ == "__main__":
    main()
