import io
import multiprocessing
import os
import warnings
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.*")

import logging

import hydra
import timm
import torch
import torch.optim as optim
from einops import rearrange, reduce
from omegaconf import DictConfig, OmegaConf
from torch import nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from dataset import DataData

import wandb

IS_LOCAL = bool(int(os.environ.get("THINKCELL_LOCAL", default="0")))
to_np = lambda x: x.detach().cpu().numpy()
logger = logging.getLogger("train")

@hydra.main(version_base=None, config_path="conf", config_name="config_local")
def load_config(hydra_config: DictConfig) -> DictConfig:
    global cfg
    cfg = hydra_config

load_config()

def update_frequency_parameters(parameter_names: List[str]) -> None:
    """
    Takes config frequency parameters in number of images and updates them to be in number of steps.
    """
    def calculate_frequency_in_steps(frequency_in_images: int) -> int:
        return max(1, frequency_in_images // cfg.hparams.batch_size)

    for param_name in parameter_names:
        current_value = getattr(cfg.hparams, param_name)
        new_value = calculate_frequency_in_steps(current_value)
        setattr(cfg.hparams, param_name, new_value)
        logger.info(f"Updated {param_name} from {current_value} (images) to {new_value} (steps).")


update_frequency_parameters([
    "log_loss_frequency",
    "validation_frequency",
    "save_checkpoint_frequency"
])

device = torch.device(cfg.hparams.device if torch.cuda.is_available() else "cpu")  # use gpu if possible

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Sequence):
        return type(data)(to_device(item, device) for item in data)
    elif data is None:
        return None
    else:
        raise TypeError(f"Input type {type(data)} not supported")


def set_class_fn(cls):
    """
    Decorator that wraps around a function and adds this function to the given class.
    """
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
    """Returns the image_batch as a tensor with shape [B,C,H,W]
    and the bboxes as a tuple with tensor items."""
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels


def tensor_denormalize(tensor):
    tensor = tensor.cpu()

    # Undo normalization and clip values to ensure they are between 0 and 1
    mean = np.array([0.8525, 0.8530, 0.8474])
    std = np.array([0.3088, 0.3043, 0.3135])
    tensor = tensor.clone()  # avoid changing the tensor in-place
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # inplace undo normalization

    return tensor


def create_image_grid(images, rows, cols):
    w, h = images[0][0].size
    grid = Image.new("RGBA", size=(cols * w, rows * h))
    for row in range(rows):
        for col in range(cols):
            img = images[row][col]
            grid.paste(img, box=(col * w, row * h))
    return grid


def log_images_to_wandb(model, dl, global_step) -> None:
    # Create and log the image grids
    N_rows = 1
    combined_images = [[] for _ in range(N_rows)]

    model.eval()

    it_ = iter(dl)
    img_batch, labels = next(it_)
    img_batch = img_batch.to(device)
    labels = labels.to(device)
    del it_

    def save_fig_to_pil_image():
        """Save the current figure to a PIL image."""
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert("RGBA")
        buf.close()
        return img
    
    with torch.no_grad():
        prediction = torch.argmax(inference_step(model, img_batch), dim=1)

    for i in range(img_batch.shape[0]):
        # Start probs
        plt.imshow(
            to_np(rearrange(tensor_denormalize(img_batch[i]), "c h w -> h w c")),
        )
        # End probs
        label = "cat" if prediction[i].item() == 1 else "dog"
        plt.figtext(0.5, 0.01, label, ha='center', fontsize=12)
        combined_images[0].append(save_fig_to_pil_image())
        plt.close()
    
    # Log the image grids
    grid_test = create_image_grid(combined_images, rows=N_rows, cols=img_batch.size(0))
    wandb.log(
            {
                "image_pred": [wandb.Image(grid_test)],
        },
        step=global_step,
        commit=False,
    )
    model.train()


@torch.compile(disable=cfg.hparams.debug)
def training_step(model, img_batch, labels, loss_fn, optimizer):
    # img_batch = model.preprocess_forward(img_batch)
    outputs = model(img_batch)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.hparams.clip_grad_norm)
    optimizer.step()
    return loss.item()


@torch.compile(disable=cfg.hparams.debug)
def inference_step(model, img_batch):
    return model(img_batch)

def val_step(model, val_loader, criterion, step):
    # Validation phase
    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = 0
        for i, (img, label) in enumerate(tqdm(val_loader, leave=False)):
            img = img.to(device)
            label = label.to(device)

            outputs = model(img)

            loss = criterion(outputs, label)
            val_loss += loss.item()

            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs == label).sum().item()
            
        val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_loader.dataset)

        wandb.log({
            "Validation Loss": val_loss,
            "Accuracy": accuracy,
        }, step=step)

def train() -> None:

    dataset_train = DataData(cfg.dataset.root_dir, cfg.hparams.img_size)
    dataset_val = DataData(cfg.dataset.root_dir, cfg.hparams.img_size)
    train_dl = DataLoader(
        dataset_train,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.hparams.num_workers,
    )
    val_dl = DataLoader(
        dataset_val,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.hparams.num_workers,
    )

    model = timm.create_model("resnet18", pretrained=True, num_classes=cfg.model.nb_classes).to(cfg.hparams.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.hparams.max_lr, weight_decay=cfg.hparams.weight_decay)
    # scheduler = ExponentialLR(optimizer, gamma=cfg.hparams.exponential_lr_decay)

    if cfg.hparams.finetune_pretrained_model:
        logger.info("[Fine-tuning a pretrained model]")
        checkpoint = torch.load(cfg.hparams.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    wandb.init(
        project="hackatum",
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    run_id = wandb.run.id

    checkpoint_comparison_fn = np.less

    global_step = 0
    loss_dict = defaultdict(float)

    best_checkpoint_score = np.inf
    for epoch in range(cfg.hparams.num_epochs):
        logger.info(f"[Starting Epoch {epoch}]")

        for batch_idx, (
            img_batch,
            labels,
        ) in enumerate(tqdm(train_dl)):

            img_batch, labels = to_device((img_batch, labels), device)

            # Use the compiled training step
            loss = training_step(model, img_batch, labels, loss_fn, optimizer)

            loss_dict["total"] += loss
            loss_dict["n"] += 1

            # if (global_step > 0 and global_step % cfg.hparams.lr_decay_frequency == 0):
            #     current_lr = optimizer.param_groups[0]["lr"]
            #     wandb.log({"learning_rate": current_lr}, step=global_step)

            #     scheduler.step()
            
            if global_step > 0 and global_step % cfg.hparams.log_loss_frequency == 0:
                try:
                    loss_dict["mean_loss"] = loss_dict["total"] / loss_dict["n"]
                    loss_dict["total"] = 0
                    loss_dict["n"] = 0
                except ZeroDivisionError:
                    pass
                wandb.log({"loss": loss_dict["mean_loss"]}, step=global_step)

            if (global_step > 0 and global_step % cfg.hparams.validation_frequency == 0):
                log_images_to_wandb(model, val_dl, global_step)
                val_step(model, val_dl, loss_fn, global_step)

            if (global_step > 0 and global_step % cfg.hparams.save_checkpoint_frequency == 0):
                if cfg.hparams.save_weights:
                    checkpoint_data = {
                        "run_id": run_id,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss_dict["mean_loss"],
                    }
                    current_checkpoint_score = loss_dict["mean_loss"]
                    if checkpoint_comparison_fn(current_checkpoint_score, best_checkpoint_score):
                        best_checkpoint_score = current_checkpoint_score
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save(checkpoint_data, f"checkpoints/checkpoint_run_{run_id}.pt")
                        logger.info(f"Best Checkpoint saved with score: {current_checkpoint_score}")

            global_step += 1
    wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")
    logger.info(f"Device: {device}, CPU Count: {multiprocessing.cpu_count()}")
    train()
