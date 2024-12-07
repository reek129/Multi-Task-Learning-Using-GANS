import os
from glob import glob
import time

import numpy as np
import h5py
import cv2

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,Subset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

from torchvision.models import mobilenet_v3_small
from sklearn.metrics import jaccard_score
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models import vgg16, VGG16_Weights



import csv

from datetime import datetime

from PIL import Image
import torch.nn.functional as F


from labels import labels


curr_dir=os.getcwd()
root= os.path.join(curr_dir,"cityscapes_dataset")
curr_dir,root


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def plot_all_losses(epoch, train_losses,valid_losses,save_dir):
    # Plot training and validation losses
    for key in train_losses.keys():
        plt.figure()
        plt.plot(train_losses[key], label=f"Train {key}")
        plt.plot(valid_losses[key], label=f"Valid {key}")
        plt.xlabel("Epoch")
        plt.ylabel(key.replace("_", " ").title())
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{key}_loss_after_epoch_{epoch}.png"))
        plt.close()

def save_training_visualization_as_gif2(epoch, inputs, seg_output, depth_output, seg_labels, depth_labels):
    inputs = inputs.detach().cpu()
    seg_output = torch.argmax(seg_output, dim=1).detach().cpu()
    depth_output = depth_output.detach().cpu()
    seg_labels = seg_labels.detach().cpu()
    depth_labels = depth_labels.detach().cpu()
    
#     inputs_rgb = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)  # Normalize inputs to [0, 1]
    
#     # Normalize depth maps for visualization
#     depth_labels_vis = (depth_labels - depth_labels.min()) / (depth_labels.max() - depth_labels.min() + 1e-5)
#     depth_preds_vis = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min() + 1e-5)



    batch_size = min(4, inputs.size(0))  # Limit to 4 samples for visualization
    fig, axes = plt.subplots(batch_size, 5, figsize=(15, 4 * batch_size))

    for i in range(batch_size):
        
        inputs_temp = inputs[i]
        # print(f"inputs_temp: {inputs_temp.shape}")
        inputs_rgb = (inputs_temp - inputs_temp.min()) / (inputs_temp.max() - inputs_temp.min() + 1e-5)  # Normalize inputs to [0, 1]
        
        depth_labels_vis = (depth_labels[i] - depth_labels[i].min()) / (depth_labels[i].max() - depth_labels[i].min() + 1e-5)
        depth_preds = depth_output[i]
        depth_preds_vis = (depth_preds - depth_preds.min()) / (depth_preds.max() - depth_preds.min() + 1e-5)
        # print(f"depth_labels_vis: {depth_labels_vis.shape}")
        # print(f"depth_preds_vis: {depth_preds_vis.shape}")

    
        
        # Row 1: Ground truth
        axes[i, 0].imshow(inputs_rgb.permute(1, 2, 0))
        axes[i, 0].set_title("RGB Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(seg_labels[i], cmap="tab20")
        axes[i, 1].set_title("GT Segmentation")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(depth_labels_vis.squeeze(), cmap="inferno")
        axes[i, 2].set_title("GT Depth")
        axes[i, 2].axis("off")

        # Row 2: Predictions
        axes[i, 3].imshow(seg_output[i], cmap="tab20")
        axes[i, 3].set_title("Generated Segmentation")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(depth_preds_vis.squeeze(), cmap="inferno")
        axes[i, 4].set_title("Generated Depth")
        axes[i, 4].axis("off")
        
    # Remove axes for cleaner visualization
    for ax in axes.flat:
        ax.axis("off")


    # plt.tight_layout()
    fig.tight_layout()
    fig.canvas.draw()
    
    # # Save current epoch as an image for GIF
    # epoch_img_path = os.path.join(gif_path, f"epoch_{epoch}.png")
    # os.makedirs(gif_path, exist_ok=True)
    # plt.savefig(epoch_img_path)
    # plt.close()
    
    
    # return epoch_img_path
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to buffer_rgba
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA has 4 channels
    plt.close(fig)

    # Convert to PIL.Image for GIF
    frame_rgb = frame[:, :, :3] 

    # Return as PIL.Image for GIF creation
    # return Image.fromarray(frame)
    return Image.fromarray(frame_rgb)




class PerceptualLoss(nn.Module):
    def __init__(self, pretrained_model="vgg16", layers=["relu3_3"], device="cuda"):
        """
        Perceptual loss class.

        Args:
            pretrained_model (str): Pretrained model to use (e.g., "vgg16").
            layers (list of str): Layers to extract features from.
            device (str): Device to load the pretrained model on ("cuda" or "cpu").
        """
        super().__init__()

        # Load pretrained model
        if pretrained_model == "vgg16":
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        else:
            raise ValueError(f"Unsupported pretrained model: {pretrained_model}")

        # Freeze the parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Select layers
        self.layers = layers
        self.feature_extractor = nn.ModuleDict({
            layer: vgg[:i] for i, layer in enumerate(vgg._modules.keys()) if layer in self.layers
        })

    def forward(self, generated, target):
        """
        Compute perceptual loss between generated and target images.

        Args:
            generated (torch.Tensor): Generated image batch.
            target (torch.Tensor): Target image batch.

        Returns:
            torch.Tensor: MSE loss between extracted features.
        """
        loss = 0.0
        for layer_name, extractor in self.feature_extractor.items():
            gen_features = extractor(generated)
            target_features = extractor(target)
            loss += F.mse_loss(gen_features, target_features)
        return loss

def scale_invariant_depth_loss(pred, target, lambda_weight=0.1):
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=False)
    
    diff = pred - target
    n = diff.numel()
    mse = torch.sum(diff**2) / n
    scale_invariant = mse - (lambda_weight / (n**2)) * (torch.sum(diff))**2
    return scale_invariant

def depth_smoothness_loss(pred, img, alpha=1.0):
    depth_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    depth_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    img_grad_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    smoothness_x = depth_grad_x * torch.exp(-alpha * img_grad_x)
    smoothness_y = depth_grad_y * torch.exp(-alpha * img_grad_y)
    return smoothness_x.mean() + smoothness_y.mean()


def inv_huber_loss(pred, target, delta=0.1):
    """
    Inverse Huber loss for depth prediction.
    Args:
        pred (Tensor): Predicted depth map.
        target (Tensor): Ground truth depth map.
        delta (float): Threshold for switching between quadratic and linear terms.
    Returns:
        Tensor: Inverse Huber loss.
    """
    abs_diff = torch.abs(pred - target)
    delta_tensor = torch.tensor(delta, dtype=abs_diff.dtype, device=abs_diff.device)  # Convert delta to tensor
    quadratic = torch.minimum(abs_diff, delta_tensor)
    linear = abs_diff - quadratic
    return (0.5 * quadratic**2 + delta_tensor * linear).mean()


def mean_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    intersection = torch.logical_and(pred == target, target != 255).float()  # Ignore class 255
    union = torch.logical_or(pred == target, target != 255).float()
    iou = torch.sum(intersection) / torch.sum(union)
    return iou



def contrastive_loss(pred, target, margin=1.0):
    """
    Contrastive loss to ensure the depth map predictions are closer to the target.
    """
    # Flatten the tensors for element-wise operations
    pred_flat = pred.view(pred.size(0), -1)  # Flatten except for the batch dimension
    target_flat = target.view(target.size(0), -1)  # Flatten except for the batch dimension

    # Compute the pairwise distances
    distances = torch.sqrt(torch.sum((pred_flat - target_flat) ** 2, dim=1))  # Batch-wise distances

    # Create labels for contrastive loss
    labels = (torch.abs(pred_flat - target_flat).mean(dim=1) < margin).float()  # Batch-wise labels

    # Calculate contrastive loss
    similar_loss = labels * distances**2
    dissimilar_loss = (1 - labels) * torch.clamp(margin - distances, min=0)**2
    loss = (similar_loss + dissimilar_loss).mean()

    return loss


def dice_loss(predictions, targets, smooth=1e-6):
    """
    Calculate Dice Loss for segmentation.
    Args:
        predictions (torch.Tensor): The predicted segmentation map (logits or probabilities).
                                    Shape: [batch_size, num_classes, height, width]
        targets (torch.Tensor): The ground truth segmentation map (one-hot encoded or integer labels).
                                Shape: [batch_size, height, width]
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: Dice Loss (scalar).
    """
    # Convert integer labels to one-hot if needed
    if predictions.shape != targets.shape:
        targets = F.one_hot(targets, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
    
    # Apply softmax to predictions for multi-class segmentation
    predictions = torch.softmax(predictions, dim=1)
    
    # Flatten tensors to calculate intersection and union
    predictions_flat = predictions.view(predictions.shape[0], predictions.shape[1], -1)
    targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
    
    # Calculate intersection and union
    intersection = (predictions_flat * targets_flat).sum(dim=-1)
    union = predictions_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
    
    # Calculate Dice Coefficient
    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    
    # Dice Loss
    return 1 - dice_coeff.mean()


def initialize_optimizers_and_schedulers(model, lr_gen=1e-4, lr_disc=1e-4, weight_decay=1e-4):
    """
    Initialize optimizers and schedulers for all generators and discriminators.
    
    Args:
        model (nn.Module): MultiTaskModel instance.
        lr_gen (float): Learning rate for generators.
        lr_disc (float): Learning rate for discriminators.
        weight_decay (float): Weight decay for optimizers.
    
    Returns:
        dict: Optimizers and schedulers for generators and discriminators.
    """
    # Optimizers for shared generator
    optimizer_shared_gen = torch.optim.AdamW(
        model.feature_generator.parameters(),
        lr=lr_gen,
        weight_decay=weight_decay
    )
    scheduler_shared_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_shared_gen, T_max=50, eta_min=1e-6
    )

    # Optimizer and scheduler for the shared generator's refinement layer
    optimizer_shared_refine = torch.optim.AdamW(
        model.shared_generator.parameters(),
        lr=lr_gen,
        weight_decay=weight_decay
    )
    scheduler_shared_refine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_shared_refine, T_max=50, eta_min=1e-6
    )

    # Optimizers and schedulers for task-specific generators
    optimizer_seg_gen = torch.optim.AdamW(
        model.seg_output_layer.parameters(),
        lr=lr_gen,
        weight_decay=weight_decay
    )
    scheduler_seg_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_seg_gen, mode='min', factor=0.5, patience=5
    )

    optimizer_depth_gen = torch.optim.AdamW(
        model.depth_output_layer.parameters(),
        lr=lr_gen,
        weight_decay=weight_decay
    )
    scheduler_depth_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_depth_gen, mode='min', factor=0.5, patience=5
    )

    # Optimizers and schedulers for task-specific discriminators
    optimizer_seg_disc = torch.optim.AdamW(
        model.seg_discriminator.parameters(),
        lr=lr_disc,
        weight_decay=weight_decay
    )
    scheduler_seg_disc = torch.optim.lr_scheduler.StepLR(
        optimizer_seg_disc, step_size=20, gamma=0.1
    )

    optimizer_depth_disc = torch.optim.AdamW(
        model.depth_discriminator.parameters(),
        lr=lr_disc,
        weight_decay=weight_decay
    )
    scheduler_depth_disc = torch.optim.lr_scheduler.StepLR(
        optimizer_depth_disc, step_size=20, gamma=0.1
    )

    # Optimizer and scheduler for the multi-task discriminator
    optimizer_multi_task_disc = torch.optim.AdamW(
        model.multi_task_discriminator.parameters(),
        lr=lr_disc,
        weight_decay=weight_decay
    )
    scheduler_multi_task_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_multi_task_disc, T_max=50, eta_min=1e-6
    )

    return {
        "optimizers": {
            "shared_gen": optimizer_shared_gen,
            "shared_refine": optimizer_shared_refine,
            "seg_gen": optimizer_seg_gen,
            "depth_gen": optimizer_depth_gen,
            "seg_disc": optimizer_seg_disc,
            "depth_disc": optimizer_depth_disc,
            "multi_task_disc": optimizer_multi_task_disc
        },
        "schedulers": {
            "shared_gen": scheduler_shared_gen,
            "shared_refine": scheduler_shared_refine,
            "seg_gen": scheduler_seg_gen,
            "depth_gen": scheduler_depth_gen,
            "seg_disc": scheduler_seg_disc,
            "depth_disc": scheduler_depth_disc,
            "multi_task_disc": scheduler_multi_task_disc
        }
    }


def train_model_with_adversarial_loss_tracking(
    model, train_loader, valid_loader, num_epochs, device, opt_sched, save_dir="results"
):
    """
    Trains a multi-task model with adversarial feedback and tracks losses.
    
    Args:
        model: Multi-task model with integrated generators and discriminators.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        num_epochs: Number of epochs to train.
        device: Device for training ("cuda" or "cpu").
        opt_sched: Dictionary of optimizers and schedulers for generators and discriminators.
        save_dir: Directory to save results.
    
    Returns:
        train_losses, valid_losses: Lists of losses for training and validation.
    """
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare CSV file for loss tracking
    csv_path = os.path.join(save_dir, f"loss_tracking_{timestamp}.csv")
    gif_path = os.path.join(save_dir, f"training_visualization_{timestamp}.gif")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_seg_loss", "train_depth_loss", "train_combined_loss",
            "train_adv_loss", 
            # "train_seg_iou",
            "valid_seg_loss", "valid_depth_loss", "valid_combined_loss",
            "valid_adv_loss", 
            # "valid_seg_iou"
        ])

    # Initialize tracking variables
    train_losses = {"seg": [], "depth": [], "combined": [], "adv": []}
    valid_losses = {"seg": [], "depth": [], "combined": [], "adv": []}
    best_combined_loss = float("inf")
    gif_frames =[]
    perceptual_loss_fn = PerceptualLoss(pretrained_model="vgg16").to(device)

    # Start training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train = {key: 0.0 for key in train_losses.keys()}
        num_batches = 0
        
        
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as pbar:
            for batch in train_loader:
                inputs, seg_labels, depth_labels = (
                    batch["left"].to(device),
                    batch["mask"].to(device),
                    batch["depth"].to(device),
                )
                input_size = inputs.size()[-2:]

                # Preprocess seg_labels to one-hot encoding
                if seg_labels.size(1) == 1:  # If class indices are given
                    seg_labels = torch.nn.functional.one_hot(seg_labels.squeeze(1), num_classes=20)
                    seg_labels = seg_labels.permute(0, 3, 1, 2).float().to(device)  # Convert to [B, C, H, W]

                # Ensure depth_labels has correct dimensions
                if depth_labels.dim() == 5:  # If depth_labels has extra dimensions
                    depth_labels = depth_labels.squeeze(2)

                # Zero gradients
                for optimizer in opt_sched["optimizers"].values():
                    optimizer.zero_grad()

                # Forward pass with discriminator outputs
                outputs = model(
                    inputs,
                    input_size=input_size,
                    seg_labels=seg_labels,
                    depth_labels=depth_labels,
                    return_discriminator_outputs=True,
                )

                # Generator losses
                seg_loss = nn.CrossEntropyLoss()(outputs["seg_output"], seg_labels) + \
                           dice_loss(outputs["seg_output"], seg_labels)
                depth_loss = scale_invariant_depth_loss(outputs["depth_output"], depth_labels) + \
                             inv_huber_loss(outputs["depth_output"], depth_labels) + \
                             depth_smoothness_loss(outputs["depth_output"], inputs)
                
                seg_perceptual_loss = perceptual_loss_fn(outputs["seg_output"], seg_labels.unsqueeze(1))
                depth_perceptual_loss = perceptual_loss_fn(outputs["depth_output"], depth_labels)
                
                seg_loss = seg_loss + 0.1 * seg_perceptual_loss
                depth_loss = depth_loss + 0.1 * depth_perceptual_loss
                

                adv_loss = -(
                    torch.mean(outputs["seg_real_disc"]) +
                    torch.mean(outputs["depth_real_disc"]) +
                    torch.mean(outputs["combined_real_disc"])
                )

                combined_loss = seg_loss + depth_loss + 0.01 * adv_loss

                # Backpropagation for generators
                combined_loss.backward(retain_graph=True)
                # opt_sched["optimizers"]["generator"].step()
                opt_sched["optimizers"]["shared_gen"].step()
                opt_sched["optimizers"]["shared_refine"].step()
                opt_sched["optimizers"]["seg_gen"].step()
                opt_sched["optimizers"]["depth_gen"].step()


                # Update task-specific discriminators
                for task, disc_optimizer in [
                    ("seg", "seg_disc"),
                    ("depth", "depth_disc"),
                ]:
                    opt_sched["optimizers"][disc_optimizer].zero_grad()
                    real_disc_loss = torch.mean(
                        (outputs[f"{task}_real_disc"] - 1) ** 2
                    )
                    fake_disc_loss = torch.mean(
                        (outputs[f"{task}_fake_disc"].detach()) ** 2
                    )
                    disc_loss = (real_disc_loss + fake_disc_loss) / 2
                    disc_loss.backward()
                    opt_sched["optimizers"][disc_optimizer].step()

                # Update multi-task discriminator
                opt_sched["optimizers"]["multi_task_disc"].zero_grad()
                real_combined_loss = torch.mean(
                    (outputs["combined_real_disc"] - 1) ** 2
                )
                fake_combined_loss = torch.mean(
                    (outputs["combined_fake_disc"].detach()) ** 2
                )
                combined_disc_loss = (real_combined_loss + fake_combined_loss) / 2
                combined_disc_loss.backward()
                opt_sched["optimizers"]["multi_task_disc"].step()

                # Update training metrics
                epoch_train["seg"] += seg_loss.item()
                epoch_train["depth"] += depth_loss.item()
                epoch_train["combined"] += combined_loss.item()
                epoch_train["adv"] += adv_loss.item()
                # epoch_train["iou"] += mean_iou(outputs["seg_output"], seg_labels, num_classes=20).item()
                num_batches += 1

            # Average training metrics
            for key in epoch_train.keys():
                train_losses[key].append(epoch_train[key] / num_batches)

            # Validation loop
            model.eval()
            epoch_valid = {key: 0.0 for key in valid_losses.keys()}
            num_valid_batches = 0

            with torch.no_grad():
                for batch in valid_loader:
                    inputs, seg_labels, depth_labels = (
                        batch["left"].to(device),
                        batch["mask"].to(device),
                        batch["depth"].to(device),
                    )
                    input_size = inputs.size()[-2:]

                    # Preprocess seg_labels to one-hot encoding
                    if seg_labels.size(1) == 1:
                        seg_labels = torch.nn.functional.one_hot(seg_labels.squeeze(1), num_classes=20)
                        seg_labels = seg_labels.permute(0, 3, 1, 2).float().to(device)

                    # Ensure depth_labels has correct dimensions
                    if depth_labels.dim() == 5:
                        depth_labels = depth_labels.squeeze(2)


                    # Forward pass
                    outputs = model(
                        inputs,
                        input_size=input_size,
                        seg_labels=seg_labels,
                        depth_labels=depth_labels,
                        return_discriminator_outputs=True,
                    )

                    # Validation loss calculations
                    seg_loss = nn.CrossEntropyLoss()(outputs["seg_output"], seg_labels) + \
                               dice_loss(outputs["seg_output"], seg_labels)
                    depth_loss = scale_invariant_depth_loss(outputs["depth_output"], depth_labels) + \
                                 inv_huber_loss(outputs["depth_output"], depth_labels) + \
                                 depth_smoothness_loss(outputs["depth_output"], inputs)
                    
                    seg_perceptual_loss = perceptual_loss_fn(outputs["seg_output"], seg_labels.unsqueeze(1))
                    depth_perceptual_loss = perceptual_loss_fn(outputs["depth_output"], depth_labels)

                    seg_loss = seg_loss + 0.1 * seg_perceptual_loss
                    depth_loss = depth_loss + 0.1 * depth_perceptual_loss


                    adv_loss = -(
                        torch.mean(outputs["seg_real_disc"]) +
                        torch.mean(outputs["depth_real_disc"]) +
                        torch.mean(outputs["combined_real_disc"])
                    )

                    combined_loss = seg_loss + depth_loss + 0.01 * adv_loss

                    # Update validation metrics
                    epoch_valid["seg"] += seg_loss.item()
                    epoch_valid["depth"] += depth_loss.item()
                    epoch_valid["combined"] += combined_loss.item()
                    epoch_valid["adv"] += adv_loss.item()
                    # epoch_valid["iou"] += mean_iou(outputs["seg_output"], seg_labels, num_classes=20).item()
                    num_valid_batches += 1

        # Average validation metrics
        for key in epoch_valid.keys():
            valid_losses[key].append(epoch_valid[key] / num_valid_batches)

        # Save best model
        valid_combined_loss = epoch_valid["combined"] / num_valid_batches
        if valid_combined_loss < best_combined_loss:
            best_combined_loss = valid_combined_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch+1} with combined loss {best_combined_loss:.4f}")
            
        frame = save_training_visualization_as_gif2(epoch, inputs, outputs["seg_output"], outputs["depth_output"], torch.argmax(seg_labels, dim=1), depth_labels)
        gif_frames.append(frame)
        
        

        # Append metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                epoch_train["seg"] / num_batches,
                epoch_train["depth"] / num_batches,
                epoch_train["combined"] / num_batches,
                epoch_train["adv"] / num_batches,
                # epoch_train["iou"] / num_batches,
                epoch_valid["seg"] / num_valid_batches,
                epoch_valid["depth"] / num_valid_batches,
                epoch_valid["combined"] / num_valid_batches,
                epoch_valid["adv"] / num_valid_batches,
                # epoch_valid["iou"] / num_valid_batches,
            ])
            
        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs} Results:")

        # Print training losses
        print(f"  Train Losses - Segmentation: {epoch_train['seg']/num_batches:.4f}, Depth: {epoch_train['depth']/num_batches:.4f}, "
              f"Combined: {epoch_train['combined']/num_batches:.4f}, Adversarial: {epoch_train['adv']/num_batches:.4f}")

        # Print validation losses
        print(f"  Valid Losses - Segmentation: {epoch_valid['seg']/ num_valid_batches:.4f}, Depth: {epoch_valid['depth']/ num_valid_batches:.4f}, "
              f"Combined: {epoch_valid['combined']/ num_valid_batches:.4f}, Adversarial: {epoch_valid['adv']/ num_valid_batches:.4f}")


        # Update schedulers
        for name, scheduler in opt_sched["schedulers"].items():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Pass the appropriate metric to ReduceLROnPlateau
                scheduler.step(valid_losses["combined"][-1])  # Use the most recent validation combined loss
            else:
                scheduler.step()
            
        if epoch %10 == 0:
            gif_path2 =os.path.join(save_dir,f"viz_epoch_{epoch}.gif")
            gif_frames[0].save(gif_path2, save_all=True, append_images=gif_frames[1:], duration=500, loop=0)
            plot_all_losses(epoch, train_losses,valid_losses,save_dir)
            
    
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=500, loop=0)
    print(f"Training visualization saved as GIF at {gif_path}")

    
    return train_losses, valid_losses


from model import MultiTaskModel, MobileNetV3Backbone

BATCH_SIZE = 8
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


mobilenet_backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# encoder = MobileNetV3Backbone(mobilenet_backbone.features)
model = MultiTaskModel(backbone=mobilenet_backbone.features, num_seg_classes=20, depth_channels=1)
model.to(DEVICE)

# Initialize optimizers and schedulers
opt_sched = initialize_optimizers_and_schedulers(model)

# Access optimizers
optimizers = opt_sched["optimizers"]
schedulers = opt_sched["schedulers"]