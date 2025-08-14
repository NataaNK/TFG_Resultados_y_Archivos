import argparse
import os
import sys
import yaml
import subprocess

def update_yaml(input_dir, yaml_path):
    """Actualizar el campo dataroot_lq en el archivo YAML."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Actualizar el campo dataroot_lq
    if 'datasets' in config and 'test' in config['datasets']:
        config['datasets']['test']['dataroot_lq'] = input_dir

    # Guardar el archivo actualizado
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(config, file)

def main():
    parser = argparse.ArgumentParser(description="Ejecutar SwinFIR con ruta dinámica para input_dir.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Ruta al directorio de entrada que se usará en dataroot_lq."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="H:\\MasOrange\\SwinFIR\\options\\test\\SwinFIR\\NATALIA_x4.yml",
        help="Ruta al archivo YAML que será modificado."
    )
    args = parser.parse_args()

    # Actualizar el YAML con la ruta proporcionada
    print(f"Actualizando {args.yaml_path} con dataroot_lq: {args.input_dir}")
    try:
        update_yaml(args.input_dir, args.yaml_path)
    except Exception as e:
        print(f"Error al actualizar el archivo YAML: {e}")
        return

    # Construir y ejecutar el comando
    command = [
        sys.executable, "swinfir/test.py",
        "-opt", args.yaml_path
    ]
    try:
        print(f"Ejecutando comando: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")

if __name__ == "__main__":
    main()
