"""
Stage 3: Temporal Motion Analysis — CNN + BiLSTM on optical flow Haar HH wavelets.
DDP training across multiple GPUs with AMP, checkpointing, and full logging.

Usage:
    torchrun --nproc_per_node=4 train_stage3.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import numpy as np

from config_stage3 import Stage3Config, MANIFEST_TRAIN, MANIFEST_VAL
from dataset_stage3 import TemporalMotionDataset
from models_stage3 import TemporalMotionModel
from utils_stage3 import setup_logger, CSVTracker, CheckpointManager, compute_metrics, EarlyStopping


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, logger, epoch, rank):
    model.train()

    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([0.125]).to(device)
    )

    total_loss = 0.0
    all_labels, all_probs = [], []
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        flow_hh = batch['flow_hh'].to(device, non_blocking=True)           # (B, 63, 1, 128, 128)
        mask = batch['attention_mask'].to(device, non_blocking=True)        # (B, 63)
        labels = batch['label'].to(device, non_blocking=True)               # (B,)
        seq_len = batch['seq_len'].to(device, non_blocking=True)            # (B,)

        with autocast('cuda', dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(flow_hh, mask, seq_len).squeeze(-1)  # (B,)
            loss = bce_loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

        if rank == 0 and (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}"
            )

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([0.125]).to(device)
    )

    total_loss = 0.0
    all_labels, all_probs = [], []
    n_batches = 0

    for batch in loader:
        flow_hh = batch['flow_hh'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        seq_len = batch['seq_len'].to(device, non_blocking=True)

        with autocast('cuda', dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(flow_hh, mask, seq_len).squeeze(-1)
            loss = bce_loss_fn(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


def main():
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    cfg = Stage3Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = setup_logger('stage3', cfg.output_dir, filename='train.log')
    if rank != 0:
        logger.setLevel(100)

    logger.info("=" * 60)
    logger.info("STAGE 3: TEMPORAL MOTION ANALYSIS TRAINING")
    logger.info(f"World size: {world_size}, Device: {device}")
    logger.info(f"Config: {cfg}")
    logger.info("=" * 60)

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Datasets
    logger.info("Loading datasets...")
    train_ds = TemporalMotionDataset(
        MANIFEST_TRAIN,
        max_flow_frames=cfg.max_flow_frames,
        min_flow_frames=cfg.min_flow_frames,
        image_size=cfg.flow_image_size,
        augment=True,
    )
    val_ds = TemporalMotionDataset(
        MANIFEST_VAL,
        max_flow_frames=cfg.max_flow_frames,
        min_flow_frames=cfg.min_flow_frames,
        image_size=cfg.flow_image_size,
        augment=False,
    )
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # Model
    logger.info("Building TemporalMotionModel (CNN + BiLSTM)...")
    model = TemporalMotionModel(
        cnn_channels=cfg.cnn_channels,
        cnn_feature_dim=cfg.cnn_feature_dim,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        lstm_dropout=cfg.lstm_dropout,
        classifier_dropout=cfg.classifier_dropout,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # Optimizer — single LR, all params train from scratch
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.epochs, T_mult=1, eta_min=1e-7
    )

    scaler = GradScaler('cuda', enabled=cfg.use_amp)

    ckpt_mgr = CheckpointManager(os.path.join(cfg.output_dir, 'checkpoints'), keep_top_k=3)

    csv_fields = [
        'epoch', 'train_loss', 'val_loss',
        'train_auc', 'val_auc', 'train_acc', 'val_acc',
        'train_f1', 'val_f1', 'lr', 'timestamp',
    ]
    csv_tracker = CSVTracker(os.path.join(cfg.output_dir, 'metrics.csv'), csv_fields) if rank == 0 else None

    early_stop = EarlyStopping(patience=cfg.patience)

    # Resume from checkpoint
    start_epoch = ckpt_mgr.load_latest(model, optimizer, scheduler, scaler, device=device)
    start_epoch += 1
    if start_epoch > 0:
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0.0

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, logger, epoch, rank
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, cfg, device)

        epoch_time = time.time() - epoch_start

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f"Epoch {epoch}/{cfg.epochs-1} ({epoch_time:.0f}s)")
            logger.info(f"  Train — Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.4f} | "
                        f"Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
            logger.info(f"  Val   — Loss: {val_loss:.4f} | AUC: {val_metrics['auc']:.4f} | "
                        f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            logger.info(f"  LR: {current_lr:.8f}")

            csv_tracker.log({
                'epoch': epoch,
                'train_loss': f"{train_loss:.6f}",
                'val_loss': f"{val_loss:.6f}",
                'train_auc': f"{train_metrics['auc']:.6f}",
                'val_auc': f"{val_metrics['auc']:.6f}",
                'train_acc': f"{train_metrics['accuracy']:.6f}",
                'val_acc': f"{val_metrics['accuracy']:.6f}",
                'train_f1': f"{train_metrics['f1']:.6f}",
                'val_f1': f"{val_metrics['f1']:.6f}",
                'lr': f"{current_lr:.8f}",
            })

            if epoch % cfg.save_every == 0:
                ckpt_mgr.save(epoch, model, optimizer, scheduler, scaler, val_metrics['auc'])

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                logger.info(f"  *** New best val AUC: {best_val_auc:.4f} ***")

            if early_stop.should_stop(val_metrics['auc']):
                logger.info(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

        dist.barrier()

    if rank == 0:
        logger.info(f"Training complete. Best val AUC: {best_val_auc:.4f}")
        if csv_tracker:
            csv_tracker.close()

    cleanup_ddp()


if __name__ == '__main__':
    main()
