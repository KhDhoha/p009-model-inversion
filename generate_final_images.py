"""
Generate and save final images from optimized latent vectors.
"""
import torch
import os
from pathlib import Path
from utils.stylegan import load_generator, create_image
from PIL import Image
import numpy as np

# Configuration
latent_file = "results/optimized_w_selected_iqlsgitx.pt"  # Change run_id as needed
output_dir = "results/final_images"
target_class = 4

# Load generator
print("Loading StyleGAN2 generator...")
G = load_generator('stylegan2-ada-pytorch/ffhq.pkl')

# Load optimized latents
print(f"Loading latents from {latent_file}...")
if not os.path.exists(latent_file):
    # Find the latest optimized_w_selected file
    import glob
    files = glob.glob("results/optimized_w_selected_*.pt")
    if files:
        latent_file = sorted(files)[-1]
        print(f"Found: {latent_file}")
    else:
        print(f"ERROR: No optimized_w_selected files found in results/")
        exit(1)

w_final = torch.load(latent_file, map_location='cuda')
print(f"Loaded {w_final.shape[0]} latent vectors")

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Generate and save images
device = 'cuda'  # load_generator puts it on CUDA automatically

print(f"\nGenerating {w_final.shape[0]} images...")
for idx in range(w_final.shape[0]):
    w = w_final[idx:idx+1]
    
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const', force_fp32=True)
    
    # Normalize to [0, 1]
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    # Convert to PIL and save
    img_pil = Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    save_path = f"{output_dir}/class{target_class}_img_{idx:03d}.png"
    img_pil.save(save_path)
    
    if (idx + 1) % 10 == 0:
        print(f"  Generated {idx + 1}/{w_final.shape[0]} images")

print(f"\nDone! Images saved to {output_dir}/")
print(f"Total images: {w_final.shape[0]}")
