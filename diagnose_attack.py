"""
Diagnose attack pipeline - show images at each stage
"""
import torch
import sys
sys.path.insert(0, 'stylegan2-ada-pytorch')
from pathlib import Path
from utils.stylegan import load_generator, create_image
import yaml

print("="*60)
print("ATTACK PIPELINE DIAGNOSTIC")
print("="*60)

# Load config
with open('config/attacking/comparison_attack.yaml') as f:
    config_dict = yaml.safe_load(f)

target_class = config_dict['attack']['targets']
print(f"\n✓ Config loaded: target=class {target_class}")

# Load StyleGAN2
print("\nLoading models...")
G = load_generator('stylegan2-ada-pytorch/ffhq.pkl')
print(f"✓ StyleGAN2 loaded")

# Check target class name
idx2cls = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 
           5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'else'}

print(f"\n🎯 Target class: {target_class} ({idx2cls.get(target_class, 'UNKNOWN')})")

device = 'cuda'

# Check initial latents saved
print("\n" + "="*60)
print("STAGE 1-2: Initial & Optimized Latent Files")
print("="*60)

import glob

init_files = sorted(glob.glob("results/init_w_*.pt"))
if init_files:
    init_latent_file = init_files[-1]
    print(f"\n✓ Found initial latents: {init_latent_file}")
    
    w_init = torch.load(init_latent_file, map_location=device)
    print(f"  Shape: {w_init.shape} (num candidates x latent_dim)")
    print(f"  These are the FILTERED candidates (high confidence on target class)")
else:
    print("\n⚠ No initial latents found - attack not started yet")

opt_files = sorted(glob.glob("results/optimized_w_*.pt"))
if opt_files:
    opt_latent_file = opt_files[-1]
    print(f"\n✓ Found optimized latents: {opt_latent_file}")
    
    w_opt = torch.load(opt_latent_file, map_location=device)
    print(f"  Shape: {w_opt.shape} (num candidates x latent_dim)")
    print(f"  These have been optimized to MAXIMIZE target class confidence")
else:
    print("\n⚠ No optimized latents found - attack optimization stage not completed")

sel_files = sorted(glob.glob("results/optimized_w_selected_*.pt"))
if sel_files:
    sel_latent_file = sel_files[-1]
    print(f"\n✓ Found selected latents: {sel_latent_file}")
    
    w_sel = torch.load(sel_latent_file, map_location=device)
    print(f"  Shape: {w_sel.shape} (final filtered by robustness test)")
    print(f"  These are ROBUST to random crops/flips while keeping target confidence")
else:
    print("\n⚠ No selected latents found - final selection stage not completed")

# Check final images in results/target_X/
print("\n" + "="*60)
print("STAGE 3: Final PNG Images")
print("="*60)

target_dir = f"results/target_{target_class}"
if Path(target_dir).exists():
    final_images = sorted(Path(target_dir).glob("*.png"))
    print(f"\n✓ Found {len(final_images)} final PNG images in {target_dir}")
    print("\nFinal images (these are what to evaluate):")
    for img_file in final_images:
        size = img_file.stat().st_size / 1024  # KB
        print(f"  {img_file.name} ({size:.1f} KB)")
else:
    print(f"\n⚠ No final images in {target_dir}")

print("\n" + "="*60)
print("ATTACK PIPELINE EXPLANATION")
print("="*60)
print("""
The model inversion attack works in 3 stages:

1️⃣  INITIAL SELECTION (find_initial_w)
   └─ Sample hundreds of random StyleGAN images
   └─ Keep only top N with highest confidence on TARGET class
   └─ Result: init_w_*.pt (filtered starting points)

2️⃣  OPTIMIZATION (optimize.py) 
   └─ For each initial latent, apply gradient descent:
      └─ Loss = -log(model_confidence_on_target_class)
      └─ Gradient update: w_new = w_old + learning_rate * ∇loss
      └─ Repeat for N epochs until convergence
   └─ Result: optimized_w_*.pt (highly confident on target)

3️⃣  FINAL SELECTION (final_selection.py)
   └─ Test robustness: apply random crops/flips to each image
   └─ Keep only images that STAY confident after distortions
   └─ Result: PNG files in results/target_X/ 

KEY INSIGHT: Images change between runs because final_selection
uses RANDOMNESS to test robustness. Each run randomly samples
different crops/flips, so different images get selected as "final".

TO FIX THIS:
- Set samples_per_target: 1 in config to keep only 1 image
- Or ignore the randomness (it's intentional for robustness)

QUALITY ISSUES:
If final images don't look like target class:
✗ Increase num_epochs: 50 (more optimization iterations)
✗ Increase search_space_size: 1000 (better initial sampling)  
✗ Verify target_class is correct in data/my_dataset/
✗ Check if model is overfitting (train on more diverse data)
""")
