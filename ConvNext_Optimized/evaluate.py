import os
import sys

sys.path.insert(0, "/home/parinayok/nutrition5k/OpenSeeD")
sys.path.append("/home/parinayok/nutrition5k")

import logging
from typing import Callable

import init_config
import torch
from custom_utils import get_loss
from dataset import Metadata, collate_fn, make_dataset
from model import get_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

logger = logging.getLogger()


def calculate_smape(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    dataset,
    device: torch.device,
) -> dict[str, float]:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE) for each category.
    
    Args:
        outputs (dict[str, torch.Tensor]): Model outputs with keys = [cal, mass, fat, carb, protein]
        metadata (list[Metadata]): Target values represented by a list of metadata (normalized)
        dataset: Dataset object with mean_metadata and std_metadata attributes
        device (torch.device): Device
        
    Returns:
        smape_dict (dict[str, float]): Dictionary of SMAPE values for each category
    """
    smape_dict = {}
    categories = ["cal", "mass", "fat", "carb", "protein"]
    
    for key in categories:
        # Get predictions (normalized) and squeeze to remove last dimension
        pred_normalized = outputs[key].squeeze(-1).cpu()
        
        # Get ground truth (normalized)
        target_normalized = torch.tensor([met.__getattribute__(key) for met in metadata])
        
        # Denormalize predictions: actual = normalized * std + mean
        mean = dataset.mean_metadata.__getattribute__(key)
        std = dataset.std_metadata.__getattribute__(key)
        pred_actual = pred_normalized * std + mean
        target_actual = target_normalized * std + mean
        
        # Calculate SMAPE: mean(|actual - predicted| / ((|actual| + |predicted|) / 2)) * 100
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        absolute_error = torch.abs(target_actual - pred_actual)
        denominator = (torch.abs(target_actual) + torch.abs(pred_actual)) / 2.0 + epsilon
        percentage_error = absolute_error / denominator
        smape = percentage_error.mean().item() * 100.0
        
        smape_dict[key] = smape
    
    return smape_dict


def evaluate(
    config: CN,
    model: nn.Module,
    loss_func: Callable,
    criterion: nn.Module,
    device: torch.device,
):
    print(" ".join(config.TITLE))
    batch_size = config.TRAIN.BATCH_SIZE
    resize_size = (config.EVAL.HEIGHT, config.EVAL.WIDTH)

    # dataset preparation
    dataset = make_dataset(config)
    dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # evaluation
    running_loss = 0.0
    running_loss_multi = {}
    # Store all predictions and targets for SMAPE calculation
    all_outputs = {key: [] for key in ["cal", "mass", "fat", "carb", "protein"]}
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            rgb_img = batch["rgb_img"]
            depth_img = batch["depth_img"]
            mask: torch.Tensor = batch["mask"]
            metadata: list[Metadata] = batch["metadata"]
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            mask = mask.to(device)
            resize = transforms.Resize(resize_size)
            rgb_img = resize(rgb_img)
            depth_img = resize(depth_img)
            mask = resize(mask)
            outputs = model(rgb_img, depth_img, mask=mask)
            loss_multi = loss_func(outputs, metadata, device)
            loss = sum(loss_multi.values())
            assert isinstance(loss, torch.Tensor)
            running_loss += loss.item() * len(rgb_img)
            for key in loss_multi.keys():
                if key in running_loss_multi.keys():
                    running_loss_multi[key] += loss_multi[key].item() * len(rgb_img)
                else:
                    running_loss_multi[key] = loss_multi[key].item() * len(rgb_img)
            
            # Store outputs and metadata for SMAPE calculation
            for key in ["cal", "mass", "fat", "carb", "protein"]:
                if key in outputs:
                    all_outputs[key].append(outputs[key].cpu())
            all_metadata.extend(metadata)

    # log
    running_loss /= len(dataset["test"])
    logger.info(f"loss: {running_loss:.4f}")
    for key in running_loss_multi.keys():
        running_loss_multi[key] /= len(dataset["test"])
        if key == "ingrs":
            logger.info(f"{key} percent loss: {running_loss_multi[key]:.4f}")
            continue
        mean = dataset["test"].mean_metadata.__getattribute__(key)
        std = dataset["test"].std_metadata.__getattribute__(key)
        logger.info(f"{key} loss: {running_loss_multi[key] * std:.4f}")
        logger.info(f"{key} percent loss: {running_loss_multi[key] * std / mean:.4f}")
    
    # Calculate and print SMAPE for each category
    # Concatenate all outputs
    concatenated_outputs = {
        key: torch.cat(all_outputs[key], dim=0) for key in all_outputs.keys()
    }
    smape_dict = calculate_smape(concatenated_outputs, all_metadata, dataset["test"], device)
    
    print("\n" + "="*50)
    print("SMAPE (Symmetric Mean Absolute Percentage Error) Results:")
    print("="*50)
    print(f"Cal loss SMAPE: {smape_dict['cal']:.4f}%")
    print(f"Mass loss SMAPE: {smape_dict['mass']:.4f}%")
    print(f"Fat loss SMAPE: {smape_dict['fat']:.4f}%")
    print(f"Carb loss SMAPE: {smape_dict['carb']:.4f}%")
    print(f"Protein loss SMAPE: {smape_dict['protein']:.4f}%")
    print("="*50 + "\n")
    
    # Also log SMAPE values
    logger.info("SMAPE Results:")
    logger.info(f"Cal loss SMAPE: {smape_dict['cal']:.4f}%")
    logger.info(f"Mass loss SMAPE: {smape_dict['mass']:.4f}%")
    logger.info(f"Fat loss SMAPE: {smape_dict['fat']:.4f}%")
    logger.info(f"Carb loss SMAPE: {smape_dict['carb']:.4f}%")
    logger.info(f"Protein loss SMAPE: {smape_dict['protein']:.4f}%")


def main():
    # init config
    _, config = init_config.get_arguments()
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    # prepare the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(config, device)
    model.to(device)
    model.eval()
    
    # Load state dict - use strict=False to handle regressor architecture differences
    # This allows loading old models trained with Regressor even if code has EnhancedRegressor
    try:
        model.load_state_dict(torch.load(config.SAVE_PATH), strict=True)
    except RuntimeError as e:
        # If strict loading fails, try with strict=False (ignores missing/unexpected keys)
        # This is useful when loading old models or models with different regressor architecture
        logger.warning(f"Strict loading failed: {e}")
        logger.warning("Attempting to load with strict=False (will ignore mismatched regressor keys)")
        model.load_state_dict(torch.load(config.SAVE_PATH), strict=False)

    # loss function and criterion
    loss_func = get_loss(config)
    criterion = nn.L1Loss()

    # init logger file path
    log_path = os.path.join("log", os.path.splitext(config.SAVE_PATH)[0] + "_eval.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_config.init_logger(os.path.dirname(log_path), os.path.basename(log_path))
    init_config.set_random_seed(config.TRAIN.SEED)
    logger.info(config.dump())

    # evaluate
    evaluate(config, model, loss_func, criterion, device=device)


if __name__ == "__main__":
    main()
