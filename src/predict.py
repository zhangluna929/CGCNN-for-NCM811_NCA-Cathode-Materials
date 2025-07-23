"""
CGCNN Prediction Script

Inference script for trained CGCNN models.

Author: LunaZhang  
Date: 2023
"""

from __future__ import print_function, division

import argparse
import os
import sys
import warnings
import time
from typing import Dict, List, Tuple, Optional, Mapping
import shutil  # needed for save_checkpoint legacy

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

# =========================
# Public API
# =========================

def predict(model_path: str,
            cif_dir: str,
            batch_size: int = 256,
            workers: int = 0,
            device: Optional[str] = None,
            print_freq: int = 10,
            n_dropout: int = 1) -> Mapping[str, object]:
    """Predict crystal properties for all CIFs in ``cif_dir`` using a trained CGCNN model.

    Parameters
    ----------
    model_path : str
        Path to ``.pth.tar`` checkpoint created by ``main.py``.
    cif_dir : str
        Directory that contains *.cif files (and optionally ``id_prop.csv`` – label not required).
    batch_size : int, default 256
    workers : int, default 0
        Number of worker processes for ``torch.utils.data.DataLoader``.
    device : str or None, default None
        ``'cuda'`` / ``'cpu'``. If *None* auto-detects GPU availability.
    print_freq : int, default 10
        How often to print batched progress (mini-benchmark & debugging).
    n_dropout : int, default 1
        >1 enables Monte-Carlo Dropout for crude uncertainty estimate (mean & std).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping ``cif_id`` → ``prediction`` (if ``n_dropout==1``) or
        ``(mean, std)`` tuple (if ``n_dropout>1``).
    """
    device_torch = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- load checkpoint & model args ----------
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device_torch)
    model_args = argparse.Namespace(**checkpoint["args"])

    # ---------- dataset & dataloader ----------
    dataset = CIFData(cif_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_pool,
                        pin_memory=(device_torch.type == "cuda"))

    # Build model (need atom/neighbor feature length from a sample)
    atom_fea_sample, nbr_fea_sample, *_ = dataset[0]
    # Robust feature-length detection that works for both Tensor and list
    if isinstance(atom_fea_sample, torch.Tensor):
        orig_atom_fea_len = atom_fea_sample.shape[-1]
    else:
        orig_atom_fea_len = len(atom_fea_sample[0])
    if isinstance(nbr_fea_sample, torch.Tensor):
        nbr_fea_len = nbr_fea_sample.shape[-1]
    else:
        nbr_fea_len = len(nbr_fea_sample[0])
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=(model_args.task == "classification"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device_torch)

    # Normalizer
    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint["normalizer"])

    # Loss criterion is not required for forward-only prediction

    # Prepare prediction containers
    predictions: Dict[str, object] = {}

    # Switch eval / dropout mode -----------------------------------
    if n_dropout > 1:
        model.train()  # enable dropout layers
    else:
        model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, _, batch_cif_ids) in enumerate(loader):
            if device_torch.type == "cuda":
                input_var = (Variable(inputs[0].cuda(non_blocking=True)),
                             Variable(inputs[1].cuda(non_blocking=True)),
                             inputs[2].cuda(non_blocking=True),
                             [idx.cuda(non_blocking=True) for idx in inputs[3]])
            else:
                input_var = (Variable(inputs[0]),
                             Variable(inputs[1]),
                             inputs[2],
                             inputs[3])

            # Monte-Carlo dropout passes
            preds_mc = []
            for _ in range(n_dropout):
                output = model(*input_var)
                if model_args.task == "regression":
                    preds_mc.append(normalizer.denorm(output.cpu()))
                else:
                    # For classification output raw logits → probability of class-1 if binary
                    if output.shape[1] == 1:
                        probs = torch.sigmoid(output.cpu())
                        preds_mc.append(probs)
                    else:
                        probs = torch.softmax(output.cpu(), dim=1)[:, 1:2]
                        preds_mc.append(probs)
            preds_mc_tensor = torch.stack(preds_mc)  # [n_dropout, batch, 1]

            # Aggregate and write
            mean_pred = preds_mc_tensor.mean(dim=0).squeeze().numpy()
            std_pred = preds_mc_tensor.std(dim=0).squeeze().numpy()

            for cif_id, mean_val, std_val in zip(batch_cif_ids, mean_pred, std_pred):
                cif_id = cif_id if isinstance(cif_id, str) else cif_id[0]
                if n_dropout > 1:
                    predictions[cif_id] = (float(mean_val), float(std_val))
                else:
                    predictions[cif_id] = float(mean_val)

            if (batch_idx % print_freq == 0) and (print_freq > 0):
                print(f"[Predict] Batch {batch_idx+1}/{len(loader)} processed.")

    return predictions

# =========================
# ----- legacy helpers ----
# (kept unchanged so that former code still works internally)
# =========================


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crystal gated neural networks')
    parser.add_argument('modelpath', help='path to the trained model.')
    parser.add_argument('cifpath', help='path to the directory of CIF files.')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--n-dropout', type=int, default=1,
                        help='number of Monte-Carlo dropout samples (default: 1)')

    args = parser.parse_args(sys.argv[1:])

    predictions = predict(args.modelpath, args.cifpath,
                          batch_size=args.batch_size,
                          workers=args.workers,
                          device=("cuda" if not args.disable_cuda and torch.cuda.is_available() else None),
                          print_freq=args.print_freq,
                          n_dropout=args.n_dropout)

    # The original code had a test=True flag in validate, which was not
    # directly exposed in the new predict function.
    # For now, we'll just print the results.
    # If the user wants to save to CSV, they'll need to adapt this.
    print("\nPredictions:")
    for cif_id, pred in predictions.items():
        if isinstance(pred, tuple):
            print(f"{cif_id}: (mean={pred[0]:.3f}, std={pred[1]:.3f})")
        else:
            print(f"{cif_id}: {pred:.3f}")
