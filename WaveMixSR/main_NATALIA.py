# test_custom_x4_via_2passes.py
import os
import argparse

import torch
import torchvision.transforms as transforms
import kornia
from PIL import Image

# Importa la arquitectura WaveMix normal (x2). 
# Vamos a encadenar dos pasadas x2 => x4.
from wavemixsr.model import WaveMixSR

def main():
    # 1) Parseamos los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(
        description="Script de inferencia para super-resolución x4 con WaveMixSR (usando 2 pasadas x2)."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Carpeta con las imágenes LR (baja resolución)."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Carpeta para guardar las imágenes SR (super-resolución)."
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="saved_model_weights/bsd100_2x_y_df2k_33.2.pth",
        help="Ruta al archivo .pth con los pesos del modelo (x2)."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    weights_path = args.weights

    # 2) Seleccionar dispositivo (GPU si disponible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3) Instanciar el modelo x2 con los mismos parámetros usados en entrenamiento
    model = WaveMixSR(
        depth = 4,
        mult = 1,
        ff_channel = 144,
        final_dim = 144,
        dropout = 0.3
    ).to(device)

    # 4) Cargar los pesos
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 5) Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # 6) Preparar transformaciones
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # 7) Recorrer todos los ficheros en la carpeta de entrada
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue  # Ignorar archivos que no sean imágenes

        lr_path = os.path.join(input_dir, filename)
        img = Image.open(lr_path).convert('RGB')

        # Convertir a tensor y luego a YCbCr (igual que en test.py)
        img_tensor = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W]
        img_ycbcr = kornia.color.rgb_to_ycbcr(img_tensor)

        with torch.no_grad():
            # Primer paso: x2
            sr_ycbcr_x2 = model(img_ycbcr)
            # Segundo paso: x2 de la salida anterior => x4 total
            sr_ycbcr_x4 = model(sr_ycbcr_x2)
        
        # Convertir de vuelta a RGB (usamos la salida x4)
        sr_rgb_x4 = kornia.color.ycbcr_to_rgb(sr_ycbcr_x4)

        # Guardar la imagen SR final x4
        sr_pil = to_pil(sr_rgb_x4.squeeze(0).cpu())
        sr_pil.save(os.path.join(output_dir, filename))

    print("¡Proceso finalizado! Las imágenes SR x4 se han guardado en:", output_dir)

if __name__ == "__main__":
    main()
