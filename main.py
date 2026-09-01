"""PGMAN training entry point."""

from __future__ import annotations

import datetime
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from common.k_fold import read_csv
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import AverageMeter, get_spatial_transform, local2global_path
from datasets.dataset import get_data_loader, get_training_set, get_validation_set
from models.pgman import PGMAN
from opts import parse_opts
from train import train_epoch
from transforms.audio import TSNAudio
from transforms.target import ClassLabel
from transforms.temporal import TSN
from validation import val_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(opt) -> PGMAN:
    model = PGMAN(
        num_frames=opt.n_frames,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        need_audio=opt.need_audio,
        need_text=opt.need_text,
    )
    return model.cuda()


def print_network(model: torch.nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")


def restore_checkpoint(model, optimizer, checkpoint_path: str) -> int:
    if not checkpoint_path:
        return 1
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 1))


def build_loaders(opt):
    target_transform = ClassLabel()
    temporal_transform = TSN(n_frames=opt.n_frames, center=False)
    audio_transform = TSNAudio(n_frames=opt.n_frames, center=False)

    training_data = get_training_set(
        opt,
        get_spatial_transform(opt, "train"),
        temporal_transform,
        target_transform,
        audio_transform,
    )
    validation_data = get_validation_set(
        opt,
        get_spatial_transform(opt, "test"),
        temporal_transform,
        target_transform,
        audio_transform,
    )
    return (
        get_data_loader(opt, training_data, shuffle=True),
        get_data_loader(opt, validation_data, shuffle=False),
    )


def run(opt) -> float:
    local2global_path(opt)
    set_seed(opt.seed)
    total_acc = AverageMeter()

    print(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    folds = read_csv(opt.fold_csv, opt.annotation_path, k=1)
    for fold, _ in enumerate(folds, start=1):
        print(f"# {'-' * 32} fold {fold} {'-' * 32} #")
        model = build_model(opt)
        print_network(model)

        criterion = get_loss(opt).cuda()
        optimizer = get_optim(opt, model.parameters(), "sgd")
        start_epoch = restore_checkpoint(model, optimizer, opt.pretrained)
        train_loader, val_loader = build_loaders(opt)

        best_acc = 0.0
        writer = SummaryWriter(logdir=opt.log_path)
        try:
            for epoch in range(start_epoch, opt.n_epochs + 1):
                train_epoch(
                    (fold, epoch), train_loader, model, criterion, optimizer,
                    opt, None, writer,
                )
                best_acc = val_epoch(
                    (fold, epoch, best_acc), val_loader, model, criterion,
                    opt, writer, optimizer,
                )
        finally:
            writer.close()
        total_acc.update(best_acc)

    print(f"Total Acc: {total_acc.avg:.4f}")
    return total_acc.avg


if __name__ == "__main__":
    run(parse_opts())
