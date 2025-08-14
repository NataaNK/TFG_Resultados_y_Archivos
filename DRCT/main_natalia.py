import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Procesar imágenes con DRCT.")
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
    args = parser.parse_args()

    # Parámetros del modelo
    model_path = r".\experiments\pretrained_models\DRCT-L_X4.pth"  # CAMBIAR ESTO CON CADA FACTOR DE ESCALA

    # Crear el directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Obtener todas las imágenes del directorio de entrada
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png'))]

    # Procesar cada imagen
    for idx, filename in enumerate(image_files, start=1):
        input_path = os.path.join(args.input_dir, filename)
        output_path = args.output_dir  # Directorio de salida, no se incluye nombre del archivo

        command = [
            sys.executable, "inference.py",  
            "--input", input_path,
            "--output", output_path,
            "--model_path", model_path,
            "--scale", "4"  # Escala por factor X
        ]

        # Imprimir el progreso
        print(f"Procesando imagen {idx}/{len(image_files)}: {filename}")

        # Ejecutar el comando
        try:
            subprocess.run(command, check=True)
            print(f"Imagen procesada y guardada en: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error al procesar la imagen {filename}: {e}")

if __name__ == "__main__":
    main()
