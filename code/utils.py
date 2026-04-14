"""
Training utilities: checkpointing, logging, metrics, reproducibility.
"""
import os
import json
import csv
import random
import time
import glob
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process():
    """Check if this is the main DDP process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log(msg, rank=None):
    """Print with timestamp, only on rank 0."""
    if rank is not None and rank != 0:
        return
    if not is_main_process():
        return
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# CHECKPOINTING
# ============================================================
def save_checkpoint(state, output_dir, epoch, is_best=False):
    """Save training checkpoint."""
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    path = os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch{epoch}.pt')
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(output_dir, 'checkpoints', 'best_model.pt')
        torch.save(state, best_path)
        log(f"  New best model saved (epoch {epoch})")

    return path


def load_latest(output_dir, model, optimizer=None, scheduler=None, device='cpu'):
    """Load latest checkpoint for resuming training. Returns epoch to resume from."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        return 0

    # Find latest checkpoint by epoch number
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch*.pt')))
    if not ckpts:
        return 0

    latest = ckpts[-1]
    log(f"Resuming from {latest}")
    state = torch.load(latest, map_location=device)

    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(state['model'])
    else:
        model.load_state_dict(state['model'])

    if optimizer and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    if scheduler and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])

    # Restore RNG states (fix: must be CPU ByteTensor)
    if 'rng_torch' in state:
        rng = state['rng_torch']
        if isinstance(rng, torch.Tensor):
            rng = rng.cpu().byte()
        torch.random.set_rng_state(rng)
    if 'rng_cuda' in state:
        cuda_states = state['rng_cuda']
        if isinstance(cuda_states, list):
            cuda_states = [s.cpu().byte() if isinstance(s, torch.Tensor) else s for s in cuda_states]
        torch.cuda.set_rng_state_all(cuda_states)
    if 'rng_numpy' in state:
        np.random.set_state(state['rng_numpy'])
    if 'rng_python' in state:
        random.setstate(state['rng_python'])

    return state.get('epoch', 0) + 1


def make_checkpoint_state(model, optimizer, scheduler, epoch, metrics):
    """Build checkpoint state dict with RNG states for exact resume."""
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    return {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'metrics': metrics,
        'rng_torch': torch.random.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all(),
        'rng_numpy': np.random.get_state(),
        'rng_python': random.getstate(),
    }


# ============================================================
# METRICS
# ============================================================
def compute_metrics(all_labels, all_probs):
    """Compute AUC, accuracy, and threshold-based metrics."""
    labels = np.array(all_labels)
    probs = np.array(all_probs)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)

    return {'auc': auc, 'accuracy': acc}


def gather_tensors(tensor, world_size):
    """Gather tensors from all DDP processes."""
    if not dist.is_initialized() or world_size <= 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


# ============================================================
# CSV LOGGING
# ============================================================
class CSVLogger:
    """Append-mode CSV logger for training metrics."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ============================================================
# LEARNING RATE WARMUP
# ============================================================
def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
    """Cosine annealing with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'], 0.5 * (1 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())