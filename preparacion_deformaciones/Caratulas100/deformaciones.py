import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

def apply_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def apply_blur(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def apply_compression(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return compressed_image

def apply_lighting_changes(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    factor = random.uniform(0.5, 1.5)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

def resize_image(image, scale=0.25):
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size)
    return resized_image

def downscale_image(image, scale=0.33):
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    downscaled_image = cv2.resize(image, new_size)
    return downscaled_image

def process_images(input_folder, output_folder, deform_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(deform_folder):
        os.makedirs(deform_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Original base name
            base_name = os.path.splitext(filename)[0]

            # Resize
            resized = resize_image(image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_resized.jpg"), resized)

            # Downscale for other transformations
            downscaled = downscale_image(image)
            downscaled_path = os.path.join(deform_folder, f"{base_name}_downscaled.jpg")
            cv2.imwrite(downscaled_path, downscaled)

            # Load downscaled image for further transformations
            downscaled_image = cv2.imread(downscaled_path)

            # Noise
            noisy = apply_noise(downscaled_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_noise.jpg"), noisy)

            # Blur
            blurred = apply_blur(downscaled_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_blur.jpg"), blurred)

            # Compression
            compressed = apply_compression(downscaled_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_compression.jpg"), compressed)

            # Lighting Changes
            lighting_changed = apply_lighting_changes(downscaled_image)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_lighting.jpg"), lighting_changed)

if __name__ == "__main__":
    input_folder = "H:\MasOrange\IMAGENES\Caratulas120_originals"  # Replace with your input folder path
    output_folder = "H:\MasOrange\IMAGENES\Caratulas120"  # Replace with your output folder path
    deform_folder = "H:\MasOrange\IMAGENES\Caratulas120_originals_deformaciones"  # Folder for downscaled images
    process_images(input_folder, output_folder, deform_folder)
