import argparse
import cv2
import glob
import numpy as np
import os
import torch

from drct.archs.DRCT_arch import DRCT  # Asegúrate de que esta importación funcione

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or folder')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image(s)')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor: 1, 2, 3, 4')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = DRCT(
        upscale=args.scale, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.0,
        depths=[6] * 12, embed_dim=180, num_heads=[6] * 12, gc=32,
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")

    # Process input
    input_paths = glob.glob(os.path.join(args.input, '*')) if os.path.isdir(args.input) else [args.input]
    print(f"Processing {len(input_paths)} input(s)...")

    for idx, path in enumerate(input_paths):
        print(f"Processing image {idx + 1}/{len(input_paths)}: {path}")
        imgname = os.path.splitext(os.path.basename(path))[0]

        # Load image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Could not load image {path}")
            continue
        print(f"Image loaded: {path}, shape: {img.shape}")

        # Normalize and prepare image for model
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // 16 + 1) * 16 - h_old
                w_pad = (w_old // 16 + 1) * 16 - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img, model, args)
                output = output[..., :h_old * args.scale, :w_old * args.scale]
        except Exception as e:
            print(f"Error during inference on {path}: {e}")
            continue

        # Convert and save output image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert back to BGR
        output = (output * 255.0).round().astype(np.uint8)

        output_path = os.path.join(args.output, f"{imgname}_DRCT-L_X4.png")
        if output is None or output.size == 0:
            print(f"Error: Output is empty for image {path}")
            continue

        if cv2.imwrite(output_path, output):
            print(f"Image saved successfully: {output_path}")
        else:
            print(f"Error: Could not save image to {output_path}")


def test(img, model, args):
    if args.tile is None:
        # Process the whole image
        return model(img)
    else:
        # Process in tiles
        b, c, h, w = img.size()
        tile = min(args.tile, h, w)
        stride = tile - args.tile_overlap
        sf = args.scale

        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

        return E.div_(W)


if __name__ == '__main__':
    main()
