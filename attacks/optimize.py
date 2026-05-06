from losses.poincare import poincare_loss
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import wandb
import os


class Optimization():
    def __init__(self, target_model, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_model
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.num_ws = num_ws
        self.clip = config.attack['clip']



    def optimize(self, w_batch, targets_batch, num_epochs):
        optimizer = self.config.create_optimizer(params=[w_batch.requires_grad_()])
        scheduler = self.config.create_lr_scheduler(optimizer)
    
        idx_to_class = getattr(self.config, "idx_to_class", None)
        # Use the first target as the folder name (assumes all targets in batch are the same)
        target_folder = os.path.join(self.config.output_dir, f"target_{int(targets_batch[0].item())}")
        os.makedirs(target_folder, exist_ok=True)
    
        for i in range(num_epochs):
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)
    
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(imgs)
            else:
                discriminator_loss = torch.tensor(0.0)
    
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)
    
            outputs = self.target(imgs)
            target_loss = poincare_loss(outputs, targets_batch).mean()
    
            optimizer.zero_grad()
            loss = target_loss + discriminator_loss * self.discriminator_weight
            loss.backward()
            optimizer.step()
    
            if scheduler:
                scheduler.step()
    
            # Live preview: log only the first image, always overwritten
            if self.config.log_progress and torch.cuda.current_device() == 0:
                with torch.no_grad():
                    img = imgs[0].detach().cpu()
                    wandb.log({"live_preview": wandb.Image(img, caption=f"Iter {i}")}, commit=True)
    
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t mean_conf={mean_conf:.4f}'
                    )
    
                # Save per-epoch results for each sample in the batch
                imgs_epoch = imgs.detach().cpu()
                for idx, (img, target) in enumerate(zip(imgs_epoch, targets_batch)):
                    class_idx = int(target.item())
                    class_name = idx_to_class.get(class_idx, str(class_idx)) if idx_to_class else str(class_idx)
                    filename = f"{class_name}_epoch_{i}_sample_{idx}.png"
                    vutils.save_image(img, os.path.join(target_folder, filename), normalize=True, value_range=(-1, 1))
    
        # Save final results for each target class
        if torch.cuda.current_device() == 0:
            imgs = self.synthesize(w_batch, num_ws=self.num_ws)
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)
            imgs = imgs.detach().cpu()
            for idx, (img, target) in enumerate(zip(imgs, targets_batch)):
                class_idx = int(target.item())
                class_name = idx_to_class.get(class_idx, str(class_idx)) if idx_to_class else str(class_idx)
                filename = f"{class_name}_final_sample_{idx}.png"
                vutils.save_image(img, os.path.join(target_folder, filename), normalize=True, value_range=(-1, 1))
    
        return w_batch.detach()   

    def synthesize(self, w, num_ws):
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
        return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
