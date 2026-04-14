"""
Stage 1: Sentry Gate — DDP training with frozen DINOv3 backbone.
Binary classification (real/fake) + cluster probe auxiliary head.
Run: torchrun --nproc_per_node=4 train_stage1.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config_train import Stage1Config, MANIFEST_TRAIN, MANIFEST_VAL
from dataset import SentryGateDataset
from models import SentryGateModel
from utils import (set_seed, log, is_main_process, save_checkpoint, load_latest,
                   make_checkpoint_state, compute_metrics, gather_tensors,
                   CSVLogger, warmup_cosine_schedule, count_parameters)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0
    all_labels, all_probs = [], []

    bce = nn.BCEWithLogitsLoss(label_smoothing=cfg.label_smoothing if hasattr(nn.BCEWithLogitsLoss, 'label_smoothing') else 0)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    for step, batch in enumerate(loader):
        rgb = batch['rgb'].to(device, non_blocking=True)
        hh1 = batch['hh1'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        clusters = batch['cluster'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            binary_logits, cluster_logits = model(rgb, hh1)
            binary_logits = binary_logits.squeeze(-1)

            loss_binary = bce(binary_logits, labels)

            # Cluster loss only for fake samples with valid cluster
            mask = (labels == 1) & (clusters >= 0)
            if mask.any():
                loss_cluster = ce(cluster_logits[mask], clusters[mask])
            else:
                loss_cluster = torch.tensor(0.0, device=device)

            loss = loss_binary + cfg.lambda_cluster * loss_cluster

        scaler.scale(loss).backward()

        if cfg.gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        probs = torch.sigmoid(binary_logits).detach()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        if step % 50 == 0:
            log(f"  Epoch {epoch} step {step}/{len(loader)}: loss={loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0
    all_labels, all_probs = [], []
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        rgb = batch['rgb'].to(device, non_blocking=True)
        hh1 = batch['hh1'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            binary_logits, _ = model(rgb, hh1)
            binary_logits = binary_logits.squeeze(-1)
            loss = bce(binary_logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(binary_logits)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


def main():
    # DDP setup
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    cfg = Stage1Config()
    set_seed(cfg.seed + dist.get_rank())

    log(f"Stage 1: Sentry Gate Training — {dist.get_world_size()} GPUs")

    # Datasets
    train_ds = SentryGateDataset(MANIFEST_TRAIN, augment=True)
    val_ds = SentryGateDataset(MANIFEST_VAL, augment=False)

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                            num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = SentryGateModel(cfg.dinov3_model, cfg.hf_token, cfg.hidden_size,
                            cfg.num_clusters, freeze_backbone=cfg.freeze_backbone)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    log(f"Trainable params: {count_parameters(model):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = warmup_cosine_schedule(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # Resume
    start_epoch = load_latest(cfg.output_dir, model, optimizer, scheduler, device)

    # Logger
    csv_log = None
    if is_main_process():
        csv_log = CSVLogger(
            os.path.join(cfg.output_dir, 'training_log.csv'),
            ['epoch', 'train_loss', 'train_auc', 'train_acc', 'val_loss', 'val_auc', 'val_acc', 'lr']
        )

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, cfg.epochs):
        train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        if is_main_process():
            log(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['accuracy']:.4f}")

            is_best = val_metrics['auc'] > best_auc
            if is_best:
                best_auc = val_metrics['auc']
                patience_counter = 0
            else:
                patience_counter += 1

            state = make_checkpoint_state(model, optimizer, scheduler, epoch,
                                          {'train': train_metrics, 'val': val_metrics})
            save_checkpoint(state, cfg.output_dir, epoch, is_best=is_best)

            if csv_log:
                csv_log.log({
                    'epoch': epoch,
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'train_auc': f"{train_metrics['auc']:.4f}",
                    'train_acc': f"{train_metrics['accuracy']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_auc': f"{val_metrics['auc']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}",
                    'lr': f"{lr:.2e}",
                })

            if patience_counter >= cfg.patience:
                log(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    dist.destroy_process_group()
    log(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()