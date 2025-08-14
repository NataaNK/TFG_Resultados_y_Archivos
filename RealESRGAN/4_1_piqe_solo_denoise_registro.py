# %% [code]
import os
import sys
import subprocess
import optuna
import cv2
import numpy as np
import time
import shutil
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Para guardar las imágenes: pip install kaleido scikit-learn plotly pybrisque

# Importar la función piqe desde el paquete pypiqe
from pypiqe import piqe

# Importar visualizaciones e importancia de parámetros de Optuna
import optuna.visualization as vis
from optuna.importance import get_param_importances

# Constantes para configuración de rutas
INPUT_FOLDER = "H:/MasOrange/IMAGENES/AAOptimizacion/FinetuningDataset/ConTexto100LR"       # Carpeta de entrada con imágenes
OUTPUT_FOLDER = "H:/MasOrange/IMAGENES/AAOptimizacion/SalidasConTexto100/4_1_piqe_registro_optuna_ConTexto100/salidas_optuna_contexto100"         # Carpeta donde se guardan los resultados temporales
MEJOR_SALIDA = "H:/MasOrange/IMAGENES/AAOptimizacion/SalidasConTexto100/4_1_piqe_registro_optuna_ConTexto100/mejor_salida_optuna_contexto100"      # Carpeta donde se copia la mejor imagen de cada entrada

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

    # Fijamos todos los parámetros excepto denoise_strength
    model_name = "realesr-general-x4v3"
    tile = 0
    tile_pad = 10
    pre_pad = 0
    face_enhance = False
    fp32 = False
    alpha_upsampler = "realesrgan"

    # Único parámetro a optimizar: denoise_strength
    # Forzar en trial 0 el valor 0.5; en los demás se sugiere
    if trial.number == 0:
        denoise_strength = 0.5
    else:
        denoise_strength = trial.suggest_float("denoise_strength", 0.0, 1.0)
    # Guardamos el valor en user_attr para luego consultarlo
    trial.set_user_attr("denoise_strength", denoise_strength)

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
        score = 1e6  # Penalización

    # Guardar el log para el trial en la misma carpeta de salida
    log_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_trial_{trial.number}.txt")
    with open(log_file, "w") as f:
        f.write(log_content)

    return score

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    start_total = time.time()
    # Crear los directorios de salida si no existen
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(MEJOR_SALIDA, exist_ok=True)

    # Definir la ruta del CSV para registrar los valores óptimos de denoise_strength
    csv_file = os.path.join(MEJOR_SALIDA, "valores_optimos_todas_imgs.csv")

    # Si el archivo no existe, escribir el encabezado
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["imagen", "denoise_strength_optimo"])

    # Procesar cada imagen en INPUT_FOLDER (filtrando extensiones válidas)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    input_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                          if os.path.splitext(f)[1].lower() in valid_exts])

    # Para registrar los valores óptimos para el gráfico final
    registros = []

    for filename in input_files:
        CURRENT_INPUT_IMAGE = os.path.join(INPUT_FOLDER, filename)
        base_name, ext = os.path.splitext(os.path.basename(CURRENT_INPUT_IMAGE))
        print(f"\nProcesando imagen: {CURRENT_INPUT_IMAGE}")

        # Si ya existe una imagen procesada para este base_name en MEJOR_SALIDA, se omite la optimización
        existing = glob.glob(os.path.join(MEJOR_SALIDA, f"{base_name}_out_trial*{ext}"))
        if existing:
            print(f"La imagen {base_name} ya fue procesada. Se omite la optimización.")
            continue

        # Crear un estudio de Optuna para la imagen actual (optimización sólo de denoise_strength)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=25)

        print("----- Mejor resultado para esta imagen -----")
        print("Mejores parámetros encontrados:")
        print(study.best_params)
        print("Mejor PIQE score:", study.best_value)

        # Seleccionar el trial óptimo (en este caso, optimizamos solo denoise_strength)
        best_trial = study.best_trial

        # Registrar el valor óptimo para esta imagen: se guarda el valor de denoise_strength en user_attrs
        dn_optimo = best_trial.user_attrs.get("denoise_strength", None)
        registros.append({"imagen": base_name, "denoise_strength_optimo": dn_optimo})

        # Escribir el registro inmediatamente en el CSV (para conservar entre ejecuciones)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([base_name, dn_optimo])

        # Análisis de importancia de parámetros y visualización
        try:
            param_importances = get_param_importances(study)
            most_important = max(param_importances, key=param_importances.get)
            print(f"El parámetro más importante para {base_name} es: {most_important} (importancia: {param_importances[most_important]:.4f})")

            fig_history = vis.plot_optimization_history(study)
            history_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_optimization_history.png")
            fig_history.write_image(history_file)

            fig_importance = vis.plot_param_importances(study)
            importance_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_param_importance.png")
            fig_importance.write_image(importance_file)

            fig_parallel = vis.plot_parallel_coordinate(study)
            parallel_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_parallel_coordinate.png")
            fig_parallel.write_image(parallel_file)

            for param in study.best_params.keys():
                fig_slice = vis.plot_slice(study, params=[param])
                slice_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_slice_{param}.png")
                fig_slice.write_image(slice_file)
        except Exception as e:
            print(f"Error al generar gráficos: {e}")

        # Copiar la imagen del mejor trial a MEJOR_SALIDA con un nombre que incluya el base name
        best_image_path = best_trial.user_attrs.get("output_image_path", None)
        if best_image_path and os.path.exists(best_image_path):
            destino = os.path.join(MEJOR_SALIDA, os.path.basename(best_image_path))
            shutil.copy(best_image_path, destino)
            print(f"La mejor imagen se ha copiado a: {destino}")
        else:
            print("No se encontró la imagen del mejor trial para copiarla.")

    # Una vez procesadas todas las imágenes, generar un gráfico que muestre los valores óptimos de denoise_strength en cada imagen
    try:
        df_opt = pd.read_csv(csv_file)
        plt.figure(figsize=(10,6))
        plt.scatter(df_opt["imagen"], df_opt["denoise_strength_optimo"], color="blue")
        plt.xlabel("Imagen")
        plt.ylabel("Denoise Strength Óptimo")
        plt.xticks(rotation=90)
        plt.title("Valores Óptimos de denoise_strength en cada Imagen")
        plt.tight_layout()
        grafico_csv_path = os.path.join(MEJOR_SALIDA, "grafico_valores_optimos.png")
        plt.savefig(grafico_csv_path)
        plt.close()
        print(f"Gráfico de valores óptimos guardado en: {grafico_csv_path}")
    except Exception as e:
        print(f"Error al generar gráfico global: {e}")

    total_elapsed = time.time() - start_total
    print(f"\nTiempo total de ejecución del script: {total_elapsed:.2f} segundos")

# %%
