import os
import shutil
import gradio as gr
import subprocess
from tkinter import Tk, filedialog
import time

# Diccionario con información sobre los modelos y sus rutas de salida
MODELOS = {
    "AURA-SR": {
        "path": "H:/MasOrange/CUDA/aura-sr-CUDA",
        "command": ".\\main_NATALIA.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/aura-sr-CUDA/scaled",
        "venv": "H:/MasOrange/CUDA/aura-sr-CUDA/venv/Scripts/activate"
    },
    "STABLE-SR": {
        "path": "D:/GIAA5/Natalia/StableSR",
        "command": "scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ./stablesr_turbo.ckpt --init-img {input_dir} --outdir {output_dir} --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples 1 --vqgan_ckpt ./vqgan_cfw_00011.ckpt --colorfix_type wavelet --upscale 4",
        "output_dir": "D:/GIAA5/Natalia/StableSR/scaled",
        "venv": "D:/GIAA5/Natalia/StableSR/venv/Scripts/activate"
    },
    "IPG": {
        "path": "H:/MasOrange/CUDA/IPG_CUDA",
        "command": "./LowLevel/IPG/exec_NATALIA.py --opt H:/MasOrange/CUDA/IPG_CUDA/LowLevel/IPG/options/test_NATALIA_x4.yml --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/IPG_CUDA/LowLevel/IPG/results/test_NATALIA_x4/visualization/CustomTestSet",
        "venv": "H:/MasOrange/CUDA/IPG_CUDA/venv/Scripts/activate"
    },
    "REAL-ESRGAN": {
        "path": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA",
        "command": "main_natalia.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA/scaled",
        "venv": "H:/MasOrange/CUDA/Real-ESRGAN-CUDA/venv/Scripts/activate"
    },
    "SWIN-FIR": {
        "path": "H:/MasOrange/CUDA/SwinFIR-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/SwinFIR-CUDA/results/SwinFIR_SRx4_Test_CPU/visualization/SingleImageTest",
        "venv": "H:/MasOrange/CUDA/SwinFIR-CUDA/venv/Scripts/activate"
    },
    "HAT": {
        "path": "H:/MasOrange/CUDA/HAT-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/HAT-CUDA/results/HAT_L_SRx4_Test/visualization/Posters44",
        "venv": "H:/MasOrange/CUDA/HAT-CUDA/venv/Scripts/activate"
    },
    "DRTC": {
        "path": "H:/MasOrange/CUDA/DRCT-CUDA",
        "command": "main_natalia.py --input_dir {input_dir} --output_dir {output_dir}",
        "output_dir": "H:/MasOrange/CUDA/DRCT-CUDA/scaled",
        "venv": "H:/MasOrange/CUDA/DRCT-CUDA/venv/Scripts/activate"
    },
    "HMA": {
        "path": "H:/MasOrange/CUDA/HMA-CUDA",
        "command": "main_natalia.py --input_dir {input_dir}",
        "output_dir": "H:/MasOrange/CUDA/HMA-CUDA/results/HMA_test/visualization/Custom",
        "venv": "H:/MasOrange/CUDA/HMA-CUDA/venv/Scripts/activate"
    }
}

# Variable global para manejar el proceso actual
current_process = None

def seleccionar_carpeta():
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal
    carpeta = filedialog.askdirectory()
    root.destroy()
    return carpeta

def ejecutar_superescalado(carpeta_input, modelo, carpeta_output):
    global current_process

    if modelo not in MODELOS:
        return f"Modelo {modelo} no encontrado."

    modelo_info = MODELOS[modelo]
    input_dir = carpeta_input
    output_dir = modelo_info["output_dir"]
    venv_python = os.path.join(modelo_info["path"], "venv", "Scripts", "python.exe")  # Ruta completa al ejecutable Python

    # Validar que el archivo python.exe exista
    if not os.path.isfile(venv_python):
        return f"No se encontró el ejecutable Python: {venv_python}"

    try:
        # Validar carpetas de entrada y salida
        if not os.path.isdir(input_dir):
            return f"La carpeta de entrada no existe: {input_dir}"
        if not os.path.isdir(os.path.dirname(output_dir)):
            return f"La carpeta de salida no es válida: {output_dir}"

        # Cambiar al directorio del modelo
        os.chdir(modelo_info["path"])

        # Construir el comando
        command = modelo_info["command"].format(input_dir=input_dir, output_dir=output_dir)
        full_command = f"\"{venv_python}\" {command}"

        # Medir el tiempo de ejecución
        if modelo == "IPG":
            # Activar el entorno virtual de IPG antes de ejecutar el comando
            activate_venv = os.path.join(modelo_info["path"], "venv", "Scripts", "activate")
            eval_cmd = f"cmd /c \"{activate_venv} && python -m basicsr.test --opt {modelo_info['path']}/LowLevel/IPG/options/test_NATALIA_x4.yml\""
            
            start_time = time.time()  # Inicio del tiempo
            current_process = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = current_process.communicate()
            end_time = time.time()  # Fin del tiempo
        else:
            start_time = time.time()  # Inicio del tiempo
            current_process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = current_process.communicate()
            end_time = time.time()  # Fin del tiempo

        # Calcular la duración
        duration = end_time - start_time

        # Verificar si el proceso fue cancelado
        if current_process is None:
            return "Ejecución cancelada."

        # Capturar errores y mostrar más detalles
        if current_process.returncode != 0:
            return f"Error ejecutando el modelo {modelo}:\nComando: {full_command}\nSalida estándar:\n{stdout}\nErrores:\n{stderr}"

        # Mover imágenes procesadas a la carpeta de salida final
        for file in os.listdir(output_dir):
            full_file_path = os.path.join(output_dir, file)
            if os.path.isfile(full_file_path):
                shutil.move(full_file_path, carpeta_output)

        current_process = None
        return f"Imágenes procesadas y movidas a {carpeta_output}.\nSalida del modelo:\n{stdout.strip()}\n-------------------------------------------------------------------------------------------------\n⚠️ Tiempo de ejecución: {duration:.2f} segundos. ⚠️\n-------------------------------------------------------------------------------------------------"

    except Exception as e:
        current_process = None
        return f"Error inesperado al ejecutar el modelo {modelo}:\n{str(e)}"

def cancelar_ejecucion():
    global current_process
    if current_process:
        current_process.terminate()
        current_process = None
        return "Ejecución cancelada correctamente."
    return "No hay ejecución en curso para cancelar."

# Crear la interfaz
with gr.Blocks() as demo:
    gr.Markdown("## Interfaz para Superescalar Imágenes con Modelos de SR")

    carpeta_input_btn = gr.Button("Seleccionar Carpeta de Imágenes (Input)")
    carpeta_input_path = gr.Textbox(label="Carpeta Seleccionada (Input)", interactive=False)

    carpeta_input_btn.click(fn=seleccionar_carpeta, inputs=[], outputs=carpeta_input_path)

    modelo = gr.Dropdown(choices=list(MODELOS.keys()), label="Modelo de SR")

    carpeta_output_btn = gr.Button("Seleccionar Carpeta de Salida (Output)")
    carpeta_output_path = gr.Textbox(label="Carpeta Seleccionada (Output)", interactive=False)

    carpeta_output_btn.click(fn=seleccionar_carpeta, inputs=[], outputs=carpeta_output_path)

    resultado = gr.Textbox(label="Resultado", interactive=False)

    ejecutar_btn = gr.Button("Ejecutar")
    ejecutar_btn.click(ejecutar_superescalado, inputs=[carpeta_input_path, modelo, carpeta_output_path], outputs=resultado)

    cancelar_btn = gr.Button("Cancelar")
    cancelar_btn.click(fn=cancelar_ejecucion, inputs=[], outputs=resultado)

demo.launch()