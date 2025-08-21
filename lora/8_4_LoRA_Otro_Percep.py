import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Importar modelo VGG19 para la pérdida perceptual
import torchvision.models as models

# Definición simplificada de SRVGGNetCompact (basada en Real-ESRGAN)
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1))
        if act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        else:
            activation = nn.ReLU(inplace=True)
        self.body.append(activation)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1))
            if act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            else:
                activation = nn.ReLU(inplace=True)
            self.body.append(activation)
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), kernel_size=3, stride=1, padding=1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = nn.functional.interpolate(x, scale_factor=self.upsampler.upscale_factor, mode='nearest')
        out = out + base
        return out

# Importar loralib (asegúrate de tenerlo instalado: pip install loralib)
try:
    import loralib as lora
except ImportError:
    raise ImportError("Necesitas instalar loralib. Puedes hacerlo con: pip install loralib")

from pypiqe import piqe

# --- Función para calcular PSNR ---
def calculate_psnr(sr, hr, max_val=255.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

# --- Función para calcular PIQE ---
def compute_piqe(np_img):
    if np_img.max() <= 1.0:
        np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
    score, _, _, _ = piqe(np_img)
    return score

# --- Dataset emparejado ---
class PairedImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.filenames = sorted(os.listdir(hr_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr_path = os.path.join(self.hr_dir, fname)
        lr_path = os.path.join(self.lr_dir, fname)
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        return lr_img, hr_img

# --- Función para convertir una capa Conv2d en una capa LoRA usando loralib ---
def convert_conv2d_to_lora(conv_module, r, lora_alpha):
    k = conv_module.kernel_size[0] if isinstance(conv_module.kernel_size, tuple) else conv_module.kernel_size
    new_conv = lora.Conv2d(
        conv_module.in_channels,
        conv_module.out_channels,
        kernel_size=k,
        stride=conv_module.stride,
        padding=conv_module.padding,
        bias=(conv_module.bias is not None),
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        merge_weights=True
    )
    new_conv.conv.weight.data.copy_(conv_module.weight.data)
    if conv_module.bias is not None:
        new_conv.conv.bias.data.copy_(conv_module.bias.data)
    return new_conv

# --- Función recursiva para reemplazar módulos Conv2d por sus versiones LoRA ---
def replace_conv_with_lora(module, r=8, lora_alpha=8):
    if len(list(module.named_children())) == 0:
        return
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and not isinstance(child, lora.Conv2d):
            if not hasattr(child, 'weight') or child.weight is None:
                print(f"DEBUG: La capa '{name}' no tiene 'weight', omitiendo reemplazo.")
                continue
            try:
                new_conv = convert_conv2d_to_lora(child, r, lora_alpha)
                setattr(module, name, new_conv)
                print(f"DEBUG: Reemplazada capa '{name}' con LoRA (r={r}, lora_alpha={lora_alpha})")
            except Exception as e:
                print(f"DEBUG: Error al reemplazar la capa '{name}': {e}")
        else:
            if len(list(child.named_children())) > 0:
                replace_conv_with_lora(child, r, lora_alpha)
            else:
                print(f"DEBUG: Módulo '{name}' no es Conv2d o ya es LoRA.")

# --- Función para imprimir la estructura del modelo ---
def debug_model_structure(model):
    print("DEBUG: Estructura del modelo:")
    for name, module in model.named_modules():
        params = list(module._parameters.keys()) if hasattr(module, "_parameters") else []
        print(f"  {name}: {module.__class__.__name__}, _parameters: {params}")

# --- Función para congelar todos los parámetros excepto los de LoRA ---
def mark_only_lora_trainable(model):
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            print(f"DEBUG: Parámetro entrenable: {name} (shape: {param.shape})")
        else:
            param.requires_grad = False

# --- Función para paddear tensores a tamaño máximo en el batch (múltiplo de upscale)
def pad_to_max(tensors, upscale=4):
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    new_h = ((max_h + upscale - 1) // upscale) * upscale
    new_w = ((max_w + upscale - 1) // upscale) * upscale
    padded = []
    for t in tensors:
        pad_h = new_h - t.shape[1]
        pad_w = new_w - t.shape[2]
        padded_t = nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded.append(padded_t)
    return torch.stack(padded)

# --- Función de collate personalizada ---
def collate_fn(batch):
    lr_list = [item[0].contiguous() for item in batch]
    hr_list = [item[1].contiguous() for item in batch]
    lr_batch = pad_to_max(lr_list, upscale=4)
    desired_h = lr_batch.shape[2] * 4
    desired_w = lr_batch.shape[3] * 4
    padded_hr = []
    for t in hr_list:
        pad_h = desired_h - t.shape[1]
        pad_w = desired_w - t.shape[2]
        padded_t = nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_hr.append(padded_t)
    hr_batch = torch.stack(padded_hr)
    return lr_batch, hr_batch

# --- Función para validación ---
def validate(model, dataloader, device):
    model.eval()
    psnr_list = []
    piqe_list = []
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            sr_imgs_np = sr_imgs.detach().cpu().clamp(0, 1).numpy()
            hr_imgs_np = hr_imgs.detach().cpu().clamp(0, 1).numpy()
            for sr, hr in zip(sr_imgs_np, hr_imgs_np):
                sr = np.transpose(sr, (1, 2, 0)) * 255.0
                hr = np.transpose(hr, (1, 2, 0)) * 255.0
                sr = sr.astype(np.uint8)
                hr = hr.astype(np.uint8)
                psnr_val = calculate_psnr(torch.tensor(sr, dtype=torch.float32),
                                            torch.tensor(hr, dtype=torch.float32),
                                            max_val=255.0)
                piqe_val = compute_piqe(sr)
                psnr_list.append(psnr_val)
                piqe_list.append(piqe_val)
    model.train()
    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
    avg_piqe = sum(piqe_list) / len(piqe_list) if piqe_list else 0.0
    return avg_psnr, avg_piqe

##########################################
# AÑADIDO: Pérdida Perceptual basada en VGG19
##########################################
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=35):
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*[vgg19[i] for i in range(feature_layer)]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, sr, hr):
        sr_norm = (sr - self.mean.to(sr.device)) / self.std.to(sr.device)
        hr_norm = (hr - self.mean.to(hr.device)) / self.std.to(hr.device)
        sr_features = self.feature_extractor(sr_norm)
        hr_features = self.feature_extractor(hr_norm)
        loss = nn.functional.l1_loss(sr_features, hr_features)
        return loss

##########################################
# AÑADIDO: Pérdida de Bordes (Edge Loss) con Sobel
##########################################
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        kernel_x = torch.tensor([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, sr, hr):
        sr_gray = 0.2989 * sr[:,0:1,:,:] + 0.5870 * sr[:,1:2,:,:] + 0.1140 * sr[:,2:3,:,:]
        hr_gray = 0.2989 * hr[:,0:1,:,:] + 0.5870 * hr[:,1:2,:,:] + 0.1140 * hr[:,2:3,:,:]
        sr_grad_x = nn.functional.conv2d(sr_gray, self.kernel_x, padding=1)
        sr_grad_y = nn.functional.conv2d(sr_gray, self.kernel_y, padding=1)
        hr_grad_x = nn.functional.conv2d(hr_gray, self.kernel_x, padding=1)
        hr_grad_y = nn.functional.conv2d(hr_gray, self.kernel_y, padding=1)
        loss_x = nn.functional.l1_loss(sr_grad_x, hr_grad_x)
        loss_y = nn.functional.l1_loss(sr_grad_y, hr_grad_y)
        return loss_x + loss_y

##########################################
# AÑADIDO: Discriminador simple y GAN Loss
##########################################
class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(SimpleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*4, 1, 3, stride=1, padding=1)
        )
    def forward(self, x):
        return self.model(x)

##########################################
# Fin de las modificaciones GAN
##########################################

# --- Función main con argumentos ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Real-ESRGAN x4v3 con LoRA, Perceptual, Edge y GAN Loss")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directorio con imágenes HR para entrenamiento")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directorio con imágenes LR para entrenamiento")
    parser.add_argument("--val_hr_dir", type=str, default=None, help="Directorio con imágenes HR para validación")
    parser.add_argument("--val_lr_dir", type=str, default=None, help="Directorio con imágenes LR para validación")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directorio para guardar checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=8, help="Rank de LoRA")
    parser.add_argument("--lora_alpha", type=int, default=8, help="Factor de escala LoRA")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de entrenamiento")
    parser.add_argument("--save_interval", type=int, default=1, help="Época cada cuánto guardar el modelo")
    parser.add_argument("--denoise_strength", type=float, default=0.5, help="Fusión de pesos: valor entre 0 y 1")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="Factor de ponderación para la pérdida perceptual")
    parser.add_argument("--lambda_edge", type=float, default=0.1, help="Factor de ponderación para la pérdida de bordes")
    parser.add_argument("--lambda_gan", type=float, default=0.01, help="Factor de ponderación para la GAN Loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = PairedImageDataset(args.hr_dir, args.lr_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = PairedImageDataset(args.val_hr_dir, args.val_lr_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    else:
        val_loader = None

    print("DEBUG: Cargando pesos pre-entrenados para realesr-general-x4v3.pth...")
    normal_weight_path = os.path.join("weights", "realesr-general-x4v3.pth")
    wdn_weight_path = os.path.join("weights", "realesr-general-wdn-x4v3.pth")
    if not os.path.isfile(normal_weight_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos normal en {normal_weight_path}")
    if not os.path.isfile(wdn_weight_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos WDN en {wdn_weight_path}")
    state_orig = torch.load(normal_weight_path, map_location="cpu")
    state_denoise = torch.load(wdn_weight_path, map_location="cpu")
    key = None
    if isinstance(state_orig, dict):
        if "params_ema" in state_orig:
            key = "params_ema"
        elif "params" in state_orig:
            key = "params"
        state_orig = state_orig[key] if key else state_orig
    if isinstance(state_denoise, dict):
        if "params_ema" in state_denoise:
            key = "params_ema"
        elif "params" in state_denoise:
            key = "params"
        state_denoise = state_denoise[key] if key else state_denoise

    combined_state = {}
    for k, v in state_orig.items():
        if k in state_denoise:
            combined_state[k] = (1 - args.denoise_strength) * v + args.denoise_strength * state_denoise[k]
        else:
            combined_state[k] = v

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    model.load_state_dict(combined_state, strict=True)
    print("DEBUG: Modelo cargado con éxito con denoise_strength =", args.denoise_strength)
    debug_model_structure(model)

    replace_conv_with_lora(model, r=args.r, lora_alpha=args.lora_alpha)
    mark_only_lora_trainable(model)
    model = model.to(device)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DEBUG: Parámetros entrenables totales (LoRA): {total_trainable}")
    if total_trainable == 0:
        raise ValueError("No se encontraron parámetros entrenables (LoRA).")

    # Configurar discriminador y GAN Loss
    discriminator = SimpleDiscriminator().to(device)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    bce_loss = nn.BCEWithLogitsLoss()

    # Configurar las otras pérdidas y optimizador del generador
    criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss(feature_layer=35).to(device)
    edge_criterion = EdgeLoss().to(device)
    lambda_perceptual = args.lambda_perceptual
    lambda_edge = args.lambda_edge
    lambda_gan = args.lambda_gan
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs+1):
        model.train()
        discriminator.train()
        epoch_loss = 0.0
        start_time = time.time()
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            ### Actualización del Discriminador ###
            optimizer_d.zero_grad()
            with torch.cuda.amp.autocast():
                sr_imgs = model(lr_imgs)
                d_real = discriminator(hr_imgs)
                d_fake = discriminator(sr_imgs.detach())
                real_labels = torch.ones_like(d_real)
                fake_labels = torch.zeros_like(d_fake)
                d_loss = bce_loss(d_real, real_labels) + bce_loss(d_fake, fake_labels)
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)

            ### Actualización del Generador ###
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                sr_imgs = model(lr_imgs)
                loss_l1 = criterion(sr_imgs, hr_imgs)
                loss_perc = perceptual_criterion(sr_imgs, hr_imgs)
                loss_edge = edge_criterion(sr_imgs, hr_imgs)
                d_fake_for_g = discriminator(sr_imgs)
                gan_loss = bce_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
                g_loss = loss_l1 + lambda_perceptual * loss_perc + lambda_edge * loss_edge + lambda_gan * gan_loss
            scaler.scale(g_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += g_loss.item() * lr_imgs.size(0)
        epoch_loss /= len(train_dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} - Pérdida Total: {epoch_loss:.4f}. Tiempo: {elapsed:.2f} s")
        if val_loader is not None:
            avg_psnr, avg_piqe = validate(model, val_loader, device)
            print(f"  -> Validación: PSNR = {avg_psnr:.2f} dB, PIQE = {avg_piqe:.2f}")
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_lora_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"DEBUG: Modelo guardado en {ckpt_path}")
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
