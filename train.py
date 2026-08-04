"""
==============================================================================
GEMO: A Deep Learning Method for Brain Fiber Classification and Tract
Segmentation Using Geometrical and Morphological Features

DOI:        https://doi.org/10.1016/j.acra.2026.07.024

Email       : Amin_br@yahoo.com
GitHub      : https://github.com/amin-barati/GEMO

==============================================================================
"""


from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import build_datasets
from streamline_model import StreamlineClassifier, build_label_mapping, count_trainable_parameters
from utils import (
    AverageMeter,
    compute_feature_normalization_stats,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_label_map,
    set_seed,
)

logger = logging.getLogger("train")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    grad_clip_norm: float = None,
    log_every_n_steps: int = 
) -> tuple:
    
    
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for step, (images, features, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images, features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        loss_meter.update(loss.item(), n=batch_size)
        acc_meter.update(accuracy, n=batch_size)

        if is_train and log_every_n_steps and step % log_every_n_steps == 0:
            logger.info("  step %d: loss=%.4f acc=%.4f", step, loss_meter.avg, acc_meter.avg)

    return loss_meter.avg, acc_meter.avg


def main(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    logger.info("Building label map from %s ...", cfg.data.trk_dir)
    label_map = build_label_mapping(cfg.data.trk_dir, separator=cfg.data.separator)
    logger.info("Found %d classes: %s", len(label_map), label_map)
    save_label_map(label_map, os.path.join(cfg.train.checkpoint_dir, "label_map.json"))

    if len(label_map) != cfg.model.num_classes:
        logger.warning(
            "cfg.model.num_classes=%d but %d classes were found in %s; using the discovered count.",
            cfg.model.num_classes,
            len(label_map),
            cfg.data.trk_dir,
        )
        cfg.model.num_classes = len(label_map)

    logger.info("Computing feature normalization statistics from %s ...", cfg.data.features_h5_path)
    feature_mean, feature_std = compute_feature_normalization_stats(cfg.data.features_h5_path, cfg.data.feature_names)
    logger.info("Feature mean: %s", feature_mean.tolist())
    logger.info("Feature std:  %s", feature_std.tolist())

    train_dataset, val_dataset = build_datasets(cfg, label_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,  
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),

    )

    model = StreamlineClassifier(
        num_handcrafted_features=cfg.model.num_handcrafted_features,
        num_classes=cfg.model.num_classes,
        cnn_base_channels=cfg.model.cnn_base_channels,
        cnn_embedding_dim=cfg.model.cnn_embedding_dim,
        feature_hidden_dims=cfg.model.feature_hidden_dims,
        feature_embedding_dim=cfg.model.feature_embedding_dim,
        classifier_hidden_dims=cfg.model.classifier_hidden_dims,
        dropout=cfg.model.dropout,
        fusion_mode=cfg.model.fusion_mode,
    ).to(device)
    model.set_feature_normalization_stats(feature_mean, feature_std)
    logger.info("Model has %d trainable parameters.", count_trainable_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma)

    start_epoch = 0
    best_val_accuracy = 0.0
    resume_path = os.path.join(cfg.train.checkpoint_dir, "last.pt")
    if os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, optimizer, device=str(device))
        start_epoch = checkpoint.get("epoch", 0)
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        logger.info("Resuming training from epoch %d.", start_epoch)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch_start = time.time()
        logger.info("Epoch %d/%d", epoch + 1, cfg.train.num_epochs)

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=cfg.train.grad_clip_norm,
            log_every_n_steps=cfg.train.log_every_n_steps,
        )
        logger.info("  train: loss=%.4f acc=%.4f", train_loss, train_acc)

        if (epoch + 1) % cfg.train.val_every_n_epochs == 0:
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
            logger.info("  val:   loss=%.4f acc=%.4f", val_loss, val_acc)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                save_checkpoint(
                    os.path.join(cfg.train.checkpoint_dir, "best.pt"),
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    best_val_accuracy=best_val_accuracy,
                )

        scheduler.step()
        save_checkpoint(resume_path, model, optimizer, epoch=epoch + 1, best_val_accuracy=best_val_accuracy)
        logger.info("Epoch %d finished in %.1fs", epoch + 1, time.time() - epoch_start)

    logger.info("Training complete. Best validation accuracy: %.4f", best_val_accuracy)


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train the dual-branch streamline classifier.")
    parser.add_argument("--trk-dir", default=None)
    parser.add_argument("--features-h5", default=None)
    parser.add_argument("--bounds-h5", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.trk_dir is not None:
        cfg.data.trk_dir = args.trk_dir
    if args.features_h5 is not None:
        cfg.data.features_h5_path = args.features_h5
    if args.bounds_h5 is not None:
        cfg.data.bounds_h5_path = args.bounds_h5
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_epochs is not None:
        cfg.train.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.device is not None:
        cfg.train.device = args.device
    return cfg


if __name__ == "__main__":
    main(_parse_args())
