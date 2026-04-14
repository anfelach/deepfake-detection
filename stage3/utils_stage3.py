"""
Shared utilities for Stage 3: logging, CSV tracking, checkpointing, metrics, early stopping.
Standalone copy — does not depend on Stage 1/2 utils.py.
"""
import os
import sys
import csv
import logging
import shutil
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


# ============================================================
# LOGGER
# ============================================================
def setup_logger(name, log_dir, filename=None):
    """Create a logger that writes to both file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# CSV METRICS TRACKER
# ============================================================
class CSVTracker:
    """Append-only CSV for epoch-level metrics. Flush after every write."""

    def __init__(self, csv_path, fieldnames):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        write_header = not os.path.exists(csv_path)
        self._file = open(csv_path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row_dict):
        row_dict['timestamp'] = datetime.now().isoformat()
        self._writer.writerow(row_dict)
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
# CHECKPOINT MANAGER
# ============================================================
class CheckpointManager:
    """Save/load training checkpoints. Keeps top-K by val AUC."""

    def __init__(self, save_dir, keep_top_k=3):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)
        self.history = []

    def save(self, epoch, model, optimizer, scheduler, scaler, val_auc, extra=None):
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_auc': val_auc,
            'rng_torch': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_numpy': np.random.get_state(),
        }
        if extra:
            state.update(extra)

        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(state, latest_path)

        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:03d}.pt')
        shutil.copy2(latest_path, epoch_path)
        self.history.append((val_auc, epoch_path))

        if not self.history or val_auc >= max(h[0] for h in self.history):
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            shutil.copy2(latest_path, best_path)

        # Prune old checkpoints
        self.history.sort(key=lambda x: -x[0])
        keep_paths = {latest_path, os.path.join(self.save_dir, 'best_model.pt')}
        for auc, path in self.history[:self.keep_top_k]:
            keep_paths.add(path)
        for auc, path in self.history[self.keep_top_k:]:
            if path not in keep_paths and os.path.exists(path):
                os.remove(path)

        return epoch_path

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
        """Load latest checkpoint. Returns epoch number or -1 if none found."""
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        if not os.path.exists(latest_path):
            return -1

        state = torch.load(latest_path, map_location=device, weights_only=False)

        if hasattr(model, 'module'):
            model.module.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler and state.get('scheduler_state_dict'):
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if scaler and state.get('scaler_state_dict'):
            scaler.load_state_dict(state['scaler_state_dict'])

        if 'rng_torch' in state:
            torch.random.set_rng_state(state['rng_torch'])
        if 'rng_cuda' in state:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        if 'rng_numpy' in state:
            np.random.set_state(state['rng_numpy'])

        return state['epoch']


# ============================================================
# METRICS
# ============================================================
def compute_metrics(labels, probs):
    """Compute binary classification metrics."""
    preds = (np.array(probs) > 0.5).astype(int)
    labels = np.array(labels)

    metrics = {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
    }
    try:
        metrics['auc'] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics['auc'] = 0.0
    return metrics


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0

    def should_stop(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
