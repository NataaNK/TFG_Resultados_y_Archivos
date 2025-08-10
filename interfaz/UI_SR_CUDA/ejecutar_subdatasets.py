import os
import shutil
import subprocess
import time

# Diccionario con información sobre los modelos y sus rutas de salida
MODELOS = {
    "AURA-SR": {
        "path": "H:/MasOrange/CUDA/aura-sr-CUDA",
        "command": ".\\main_NATALIA.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/aura-sr-CUDA/scaled",
        "venv_python": "H:/MasOrange/CUDA/aura-sr-CUDA/venv/Scripts/python.exe"
    },
    "STABLE-SR": {
        "path": "D:/GIAA5/Natalia/StableSR",
        "command": "scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ./stablesr_turbo.ckpt --init-img {input_dir} --outdir {output_dir} --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt ./vqgan_cfw_00011.ckpt --colorfix_type wavelet --upscale 4",
        "output_dir": "D:/GIAA5/Natalia/StableSR/scaled",
        "venv_python": "D:/GIAA5/Natalia/StableSR/venv/Scripts/python.exe"
    },
    "IPG": {
        "path": "H:/MasOrange/CUDA/IPG_CUDA",
        "command": "./LowLevel/IPG/exec_NATALIA.py --opt H:/MasOrange/CUDA/IPG_CUDA/LowLevel/IPG/options/test_NATALIA_x4.yml --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/IPG_CUDA/LowLevel/IPG/results/test_NATALIA_x4/visualization/CustomTestSet",
        "venv_python": "H:/MasOrange/CUDA/IPG_CUDA/venv/Scripts/python.exe"
    },
    "REAL-ESRGAN": {
        "path": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA",
        "command": "main_natalia.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA/scaled",
        "venv_python": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA/venv/Scripts/python.exe"
    },
    "SWIN-FIR": {
        "path": "H:/MasOrange/CUDA/SwinFIR-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/SwinFIR-CUDA/results/SwinFIR_SRx4_Test_CPU/visualization/SingleImageTest",
        "venv_python": "H:/MasOrange/CUDA/SwinFIR-CUDA/venv/Scripts/python.exe"
    },
    "HAT": {
        "path": "H:/MasOrange/CUDA/HAT-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/HAT-CUDA/results/HAT_L_SRx4_Test/visualization/Posters44",
        "venv_python": "H:/MasOrange/CUDA/HAT-CUDA/venv/Scripts/python.exe"
    },
    "DRTC": {
        "path": "H:/MasOrange/CUDA/DRCT-CUDA",
        "command": "main_natalia.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/DRCT-CUDA/scaled",
        "venv_python": "H:/MasOrange/CUDA/DRCT-CUDA/venv/Scripts/python.exe"
    },
    "HMA": {
        "path": "H:/MasOrange/CUDA/HMA-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/HMA-CUDA/results/HMA_test/visualization/Custom",
        "venv_python": "H:/MasOrange/CUDA/HMA-CUDA/venv/Scripts/python.exe"
    },
    "WAVE-MIX": {
        "path": "H:/MasOrange/CUDA/WaveMixSR",
        "command": "main_NATALIA.py --input_dir {input_dir} --output_dir {output_dir} --weights \"saved_model_weights/bsd100_2x_y_df2k_33.2.pth\"",
        # Aquí NO es necesario mover los resultados, porque ya se guardan en el output que indiquemos
        # Así que podemos usar directamente el mismo output_dir que se pasa por parámetro.
        "output_dir": "", 
        "venv_python": "H:/MasOrange/CUDA/WaveMixSR/venv/Scripts/python.exe"
    }
}

# =============================================================================
# AJUSTA ESTAS TRES CONSTANTES A TUS NECESIDADES
# =============================================================================

# 1. Lista de rutas de entrada (donde están las imágenes originales).
INPUT_DIRS = [
    r"H:\MasOrange\IMAGENES\AAOptimizacion\FinetuningDataset\ConTexto100LR",
]

# 2. Lista de rutas de salida (donde quieres que se guarden las imágenes procesadas).
OUTPUT_DIRS = [
    r"H:\MasOrange\IMAGENES\AAOptimizacion\SalidasConTexto100\RealESRGANPlusx4_ConTexto100",
]

# 3. Modelo de SR que quieres utilizar (debe ser una de las claves del diccionario MODELOS).
MODELO = "REAL-ESRGAN"
# =============================================================================


def ejecutar_superescalado(input_dir, output_dir, modelo):
    """
    Ejecuta la superresolución para un modelo dado y mueve los resultados
    a la carpeta de salida especificada, salvo para modelos que ya guardan
    directamente en la carpeta de salida (por ejemplo, WAVE-MIX).
    """
    if modelo not in MODELOS:
        raise ValueError(f"Modelo {modelo} no encontrado en el diccionario MODELOS.")

    modelo_info = MODELOS[modelo]
    model_path = modelo_info["path"]
    model_command = modelo_info["command"]
    model_output_dir = modelo_info["output_dir"]  # Carpeta donde el modelo deja los resultados (si aplica)
    venv_python = modelo_info["venv_python"]

    # Validar que el ejecutable python exista
    if not os.path.isfile(venv_python):
        raise FileNotFoundError(f"No se encontró el ejecutable de Python: {venv_python}")

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"La carpeta de entrada no existe: {input_dir}")

    # Asegurarse de que la carpeta de salida final exista
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Cambiar al directorio del modelo
    os.chdir(model_path)

    # Construir el comando
    # Para algunos modelos, model_output_dir se utiliza dentro del comando. En WAVE-MIX usamos directamente {output_dir}.
    # Si "model_output_dir" no se usa, se puede poner "" o la misma {output_dir}.
    command = model_command.format(input_dir=input_dir, output_dir=output_dir)
    full_command = f"\"{venv_python}\" {command}"

    # Medir el tiempo de ejecución
    start_time = time.time()
    if modelo == "IPG":
        # El caso IPG requiere un comando específico
        activate_venv = os.path.join(model_path, "venv", "Scripts", "activate")
        eval_cmd = (
            f"cmd /c \"{activate_venv} && "
            f"python -m basicsr.test --opt {model_path}/LowLevel/IPG/options/test_NATALIA_x4.yml\""
        )
        process = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        # Resto de modelos
        process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()
    end_time = time.time()
    duration = end_time - start_time

    if process.returncode != 0:
        msg_error = (
            f"\nERROR ejecutando el modelo {modelo}:\n"
            f"Comando: {full_command}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )
        raise RuntimeError(msg_error)

    # Si el modelo NO es WAVE-MIX, movemos las imágenes procesadas desde el directorio interno al de salida.
    # (WAVE-MIX ya guarda directamente sus resultados en output_dir, por lo que no es necesario mover).
    if modelo != "WAVE-MIX":
        for file in os.listdir(model_output_dir):
            full_file_path = os.path.join(model_output_dir, file)
            if os.path.isfile(full_file_path):
                shutil.move(full_file_path, output_dir)

    # Retornar tiempo y logs
    return duration, stdout, stderr


def main():
    # Ruta absoluta del fichero de log (salida.txt) en el mismo directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "salida.txt")
    
    # Abrimos el fichero en modo "append" para ir agregando texto.
    with open(log_path, "a", encoding="utf-8") as log_file:
        if len(INPUT_DIRS) != len(OUTPUT_DIRS):
            mensaje_error = "Las listas INPUT_DIRS y OUTPUT_DIRS deben tener la misma longitud."
            print(mensaje_error)
            log_file.write(mensaje_error + "\n")
            return

        for i, (input_dir, output_dir) in enumerate(zip(INPUT_DIRS, OUTPUT_DIRS), start=1):
            # Generar cabecera
            header = (
                f"\n---------------------------------------------\n"
                f"PROCESANDO LOTE #{i}\n"
                f"Directorio de entrada : {input_dir}\n"
                f"Directorio de salida  : {output_dir}\n"
                f"Usando modelo         : {MODELO}\n"
            )
            print(header)
            log_file.write(header)

            try:
                duracion, stdout, _ = ejecutar_superescalado(input_dir, output_dir, MODELO)
                resumen = (
                    f"Tiempo de ejecución (lote #{i}): {duracion:.2f} segundos.\n"
                    f"Salida estándar del proceso (resumen):\n"
                    f"{stdout.strip()[:500]} ...\n"
                )
                print(resumen)
                log_file.write(resumen)
            except Exception as e:
                error_text = f"Ocurrió un error en el lote #{i}:\n{e}\n"
                print(error_text)
                log_file.write(error_text)


if __name__ == "__main__":
    main()
