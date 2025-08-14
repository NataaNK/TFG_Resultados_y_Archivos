import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Número de GPUs disponibles:", torch.cuda.device_count())
print("GPU actual:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Ninguna")
