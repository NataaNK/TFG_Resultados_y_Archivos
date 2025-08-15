# %% [code]
import os
import sys
import subprocess
import optuna
import cv2
import numpy as np
import time
import shutil

# Importar la función piqe desde el paquete pypiqe
from pypiqe import piqe

# Constantes para configuración de rutas
INPUT_FOLDER = "H:/MasOrange/IMAGENES/AAOptimizacion/Final150/Final150Compressed"       # Carpeta de entrada con imágenes
OUTPUT_FOLDER = "H:/MasOrange/IMAGENES/AAOptimizacion/salidas_compressed"         # Carpeta donde se guardan los resultados temporales
MEJOR_SALIDA = "H:/MasOrange/IMAGENES/AAOptimizacion/mejor_salida_compressed"     # Carpeta donde se copia la mejor imagen de cada entrada

# Variable global para la imagen actual
CURRENT_INPUT_IMAGE = None

# -----------------------------------------------------------------------------
def compute_piqe(image_path):
    """
    Calcula el PIQE real para la imagen utilizando la función piqe del paquete pypiqe.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    # La función piqe retorna (score, activityMask, artifactsMask, noiseMask)
    score, _, _, _ = piqe(img)
    return score

# -----------------------------------------------------------------------------
def objective(trial):
    global CURRENT_INPUT_IMAGE  # Usar la imagen actual asignada en el loop
    # Parámetros sugeridos por Optuna
    model_name = trial.suggest_categorical(
        "model_name",
        ["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B",
         "realesr-animevideov3", "realesr-general-x4v3"]
    )
    denoise_strength = trial.suggest_float("denoise_strength", 0.0, 1.0)
    tile = trial.suggest_int("tile", 0, 512)
    tile_pad = trial.suggest_int("tile_pad", 0, 50)
    pre_pad = trial.suggest_int("pre_pad", 0, 50)
    face_enhance = trial.suggest_categorical("face_enhance", [False, True])
    fp32 = trial.suggest_categorical("fp32", [False, True])
    alpha_upsampler = trial.suggest_categorical("alpha_upsampler", ["realesrgan", "bicubic"])

    # Forzar parámetros por defecto en el trial 0
    if trial.number == 0:
        model_name = "RealESRGAN_x4plus"
        denoise_strength = 0.5
        tile = 0
        tile_pad = 10
        pre_pad = 0
        face_enhance = False
        fp32 = False
        alpha_upsampler = "realesrgan"

    # Construir el comando de inferencia
    command = [
        sys.executable, "inference_realesrgan.py",
        "-i", CURRENT_INPUT_IMAGE,
        "-n", model_name,
        "-o", OUTPUT_FOLDER,
        "--denoise_strength", str(denoise_strength),
        "-s", "4",  # Factor de escala fijo
        "--tile", str(tile),
        "--tile_pad", str(tile_pad),
        "--pre_pad", str(pre_pad),
        "--alpha_upsampler", alpha_upsampler
    ]
    if face_enhance:
        command.append("--face_enhance")
    if fp32:
        command.append("--fp32")

    print("Ejecutando comando:")
    print(" ".join(command))

    start_time = time.time()
    try:
        # Ejecutar la inferencia y capturar la salida
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Generar el nombre de la imagen de salida automáticamente:
        # Se asume que la imagen generada se llama "<base_name>_out<ext>"
        base_name, ext = os.path.splitext(os.path.basename(CURRENT_INPUT_IMAGE))
        original_output_image = os.path.join(OUTPUT_FOLDER, f"{base_name}_out{ext}")
        new_output_image = f"{base_name}_out_trial{trial.number}{ext}"
        new_output_image_path = os.path.join(OUTPUT_FOLDER, new_output_image)

        # Renombrar el archivo de salida para evitar sobreescritura
        os.rename(original_output_image, new_output_image_path)
        trial.set_user_attr("output_image_path", new_output_image_path)

        # Calcular el PIQE usando la imagen renombrada
        score = compute_piqe(new_output_image_path)
        print(f"Trial {trial.number}: PIQE score = {score}")
        print(f"Trial {trial.number}: Tiempo de ejecución = {elapsed_time:.2f} segundos")

        log_content = (
            f"Trial number: {trial.number}\n"
            "Parameters:\n"
            f"  model_name: {model_name}\n"
            f"  denoise_strength: {denoise_strength}\n"
            f"  tile: {tile}\n"
            f"  tile_pad: {tile_pad}\n"
            f"  pre_pad: {pre_pad}\n"
            f"  face_enhance: {face_enhance}\n"
            f"  fp32: {fp32}\n"
            f"  alpha_upsampler: {alpha_upsampler}\n"
            f"PIQE Score: {score}\n"
            f"Tiempo de ejecución: {elapsed_time:.2f} segundos\n\n"
            "Subprocess Output:\n"
            f"{result.stdout}\n"
        )
        if result.stderr:
            log_content += "Subprocess Error Output:\n" + result.stderr

    except Exception as e:
        elapsed_time = time.time() - start_time
        base_name, ext = os.path.splitext(os.path.basename(CURRENT_INPUT_IMAGE))
        log_content = (
            f"Trial number: {trial.number}\n"
            "Parameters:\n"
            f"  model_name: {model_name}\n"
            f"  denoise_strength: {denoise_strength}\n"
            f"  tile: {tile}\n"
            f"  tile_pad: {tile_pad}\n"
            f"  pre_pad: {pre_pad}\n"
            f"  face_enhance: {face_enhance}\n"
            f"  fp32: {fp32}\n"
            f"  alpha_upsampler: {alpha_upsampler}\n"
            f"Error occurred: {str(e)}\n"
            f"Tiempo de ejecución hasta error: {elapsed_time:.2f} segundos\n"
        )
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Se retorna un valor alto para penalizar el trial
        score = 1e6

    # Guardar el log para el trial en la misma carpeta de salida
    log_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_trial_{trial.number}.txt")
    with open(log_file, "w") as f:
        f.write(log_content)

    return score

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Crear los directorios de salida si no existen
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(MEJOR_SALIDA, exist_ok=True)

    # Obtener la lista de imágenes en INPUT_FOLDER (se filtran extensiones válidas)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    input_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                          if os.path.splitext(f)[1].lower() in valid_exts])

    # Procesar cada imagen de la carpeta de entrada
    for filename in input_files:
        CURRENT_INPUT_IMAGE = os.path.join(INPUT_FOLDER, filename)
        print(f"\nProcesando imagen: {CURRENT_INPUT_IMAGE}")

        # Crear un estudio de Optuna para la imagen actual
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        print("----- Mejor resultado para esta imagen -----")
        print("Mejores parámetros encontrados:")
        print(study.best_params)
        print("Mejor PIQE score:", study.best_value)

        # Copiar la imagen del mejor trial a MEJOR_SALIDA con un nombre que incluya el base name
        best_trial = study.best_trial
        best_image_path = best_trial.user_attrs.get("output_image_path", None)
        if best_image_path and os.path.exists(best_image_path):
            destino = os.path.join(MEJOR_SALIDA, os.path.basename(best_image_path))
            shutil.copy(best_image_path, destino)
            print(f"La mejor imagen se ha copiado a: {destino}")
        else:
            print("No se encontró la imagen del mejor trial para copiarla.")
