# compare_models_sanity.py
import argparse
import glob
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms as T
from collections import Counter

from utils.attack_config_parser import AttackConfigParser
from utils.training_config_parser import TrainingConfigParser
from models.classifier import Classifier
from datasets.custom_subset import ClassSubset
from metrics.accuracy import Accuracy

def find_latest_defender_checkpoint(save_root, arch='resnet50'):
    candidates = sorted(glob.glob(os.path.join(save_root, f"{arch}_*")), reverse=True)
    if not candidates:
        return None
    latest = candidates[0]
    pths = glob.glob(os.path.join(latest, "Classifier_*.pth"))
    return pths[0] if pths else None

def clean_and_load_state(model, checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    new_state = {}
    for k, v in state.items():
        if k.startswith("model._orig_mod."):
            new_k = "model." + k[len("model._orig_mod."):]
        elif k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state[new_k] = v
    res = model.load_state_dict(new_state, strict=False)
    return res

def evaluate_model_on_class(model, dataset, class_idx, batch_size, device):
    subset = ClassSubset(dataset, [class_idx])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    acc_metric = Accuracy()
    confidences = []
    model.to(device).eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            confidences.append(probs.gather(1, y.unsqueeze(1)).squeeze(1).cpu())
            acc_metric.update(out.cpu(), y.cpu())
    acc = acc_metric.compute_metric()
    mean_conf = torch.cat(confidences).mean().item() if confidences else 0.0
    return acc, mean_conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack-config', default='config/attacking/my_attacking.yaml')
    parser.add_argument('--defender-config', default='config/training/defender_training.yaml')
    parser.add_argument('--defender-checkpoint', default=None)
    parser.add_argument('--image-root', default='data/my_dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Load configs
    attack_cfg = args.attack_config
    defender_cfg = args.defender_config

    # Build test transform from training config (ensures same preprocessing)
    train_cfg_parser = TrainingConfigParser(defender_cfg)
    test_transform = train_cfg_parser.create_transformations(mode='test', normalize=True)

    # Dataset
    dataset = ImageFolder(root=args.image_root, transform=test_transform)
    class_to_idx = dataset.class_to_idx
    if not class_to_idx:
        print("No classes found under", args.image_root)
        return

    # Vulnerable model loaded via AttackConfigParser (handles key remapping)
    attack_parser = AttackConfigParser(attack_cfg)
    vuln_model = attack_parser.create_target_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vuln_model.to(device).eval()

    # Defender model (optional)
    defender_ckpt = args.defender_checkpoint
    if defender_ckpt is None:
        defend_save_root = train_cfg_parser._config['training']['save_path']
        defender_ckpt = find_latest_defender_checkpoint(defend_save_root, arch=train_cfg_parser._config['model']['architecture'])
    defender_model = None
    if defender_ckpt:
        defend_model = train_cfg_parser.create_model()
        res = clean_and_load_state(defend_model, defender_ckpt)
        print("Defender load result:", res)
        defend_model.to(device).eval()
        defender_model = defend_model
        print("Using defender checkpoint:", defender_ckpt)
    else:
        print("No defender checkpoint found; skipping defender evaluation.")

    # Report mapping and per-class metrics
    print("\nClass mapping (name -> idx):")
    for name, idx in class_to_idx.items():
        print(f"  {name} -> {idx}")

    print("\nEvaluating per-class (accuracy, mean_target_confidence):")
    print("Class\tVulnAcc\tVulnConf\tDefAcc\tDefConf")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        vuln_acc, vuln_conf = evaluate_model_on_class(vuln_model, dataset, idx, args.batch_size, device)
        if defender_model:
            def_acc, def_conf = evaluate_model_on_class(defender_model, dataset, idx, args.batch_size, device)
        else:
            def_acc, def_conf = (None, None)
        print(f"{class_name}\t{vuln_acc:.4f}\t{vuln_conf:.4f}\t{'' if def_acc is None else f'{def_acc:.4f}'}\t{'' if def_conf is None else f'{def_conf:.4f}'}")

if __name__ == "__main__":
    main()