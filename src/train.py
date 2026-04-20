"""Training & validation loop for the multimodal prostate MRI model.

Example
-------
.. code-block:: bash

    python -m src.train \\
        --data-root /path/to/UC2_data \\
        --labels-train /path/to/train.csv \\
        --labels-val /path/to/val.csv \\
        --out-dir ./runs/exp1 \\
        --batch-size 2 --epochs 200 --lr 3e-5
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (ProstateMultiModalDataset, align_sequences,
                   default_eval_transforms, default_train_transforms)
from .losses import focal_bce_loss, l1_norm_model
from .models import MultimodalProstateModel
from .utils import seed_everything


# ---------------------------------------------------------------------------
# Data setup helpers
# ---------------------------------------------------------------------------
def _sorted_glob(pattern):
    return sorted(glob.glob(pattern), key=lambda x: os.path.basename(x).split("_")[0])


def _load_label_dict(csv_path):
    df = pd.read_csv(csv_path)
    return pd.Series(df.label_ISUP.values, index=df.patient_id).to_dict()


def build_split(data_root, split, labels_csv):
    """Return (dataset-ready paths, label_dict) for one split."""
    t2w = _sorted_glob(os.path.join(data_root, "t2w", split, "*.nii.gz"))
    dwi = _sorted_glob(os.path.join(data_root, "dwi", split, "*.nii.gz"))
    adc = _sorted_glob(os.path.join(data_root, "adc", split, "*.nii.gz"))
    cln = _sorted_glob(os.path.join(data_root, "clinical_variables_v2", split, "*.npy"))

    t2w, dwi, adc, cln, _ = align_sequences(t2w, dwi, adc, cln, strict=True)
    label_dict = _load_label_dict(labels_csv)
    return (t2w, dwi, adc, cln), label_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "confusion_matrix": cm,
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, class_weights,
                    focal_gamma, l1_lambda, device):
    model.train()
    data_sum = 0.0
    count = 0
    y_true, y_pred, y_prob = [], [], []

    for t2w, dwi, adc, cln, label in tqdm(loader, desc="train", leave=False):
        t2w = t2w.to(device).float()
        dwi = dwi.to(device).float()
        adc = adc.to(device).float()
        cln = cln.to(device).float()
        label = label.to(device).float()

        out = model(t2w, dwi, adc, cln)
        per_sample = focal_bce_loss(out["binary_task"], label.unsqueeze(-1),
                                    class_weights, gamma=focal_gamma, reduction="none")
        data_loss = per_sample.mean()
        reg_loss = l1_lambda * l1_norm_model(model)
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        data_sum += per_sample.sum().item()
        count += label.size(0)

        probs = torch.sigmoid(out["binary_task"]).view(-1).detach().cpu().tolist()
        y_prob.extend(probs)
        y_pred.extend([1.0 if p >= 0.5 else 0.0 for p in probs])
        y_true.extend(label.detach().cpu().tolist())

    epoch_loss = data_sum / max(count, 1)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = epoch_loss
    return metrics


@torch.inference_mode()
def evaluate(model, loader, class_weights, focal_gamma, device):
    model.eval()
    ones = torch.tensor([1.0, 1.0], device=device, dtype=class_weights.dtype)

    sum_w, sum_u, count = 0.0, 0.0, 0
    y_true, y_pred, y_prob = [], [], []

    for t2w, dwi, adc, cln, label in tqdm(loader, desc="val", leave=False):
        t2w = t2w.to(device).float()
        dwi = dwi.to(device).float()
        adc = adc.to(device).float()
        cln = cln.to(device).float()
        label = label.to(device).float()

        out = model(t2w, dwi, adc, cln)
        fl_w = focal_bce_loss(out["binary_task"], label.unsqueeze(-1),
                              class_weights, gamma=focal_gamma, reduction="none")
        fl_u = focal_bce_loss(out["binary_task"], label.unsqueeze(-1),
                              ones, gamma=focal_gamma, reduction="none")

        sum_w += fl_w.sum().item()
        sum_u += fl_u.sum().item()
        count += label.size(0)

        probs = torch.sigmoid(out["binary_task"]).view(-1).cpu().tolist()
        y_prob.extend(probs)
        y_pred.extend([1.0 if p >= 0.5 else 0.0 for p in probs])
        y_true.extend(label.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = sum_u / max(count, 1)
    metrics["loss_weighted"] = sum_w / max(count, 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Root with {t2w,dwi,adc,clinical_variables_v2}/{train,val,test}/")
    parser.add_argument("--labels-train", required=True)
    parser.add_argument("--labels-val", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l1-lambda", type=float, default=1e-5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--w-neg", type=float, default=1.755,
                        help="Class weight for negatives (csPCa = 0).")
    parser.add_argument("--w-pos", type=float, default=0.699,
                        help="Class weight for positives (csPCa = 1).")
    parser.add_argument("--pretrained", default=None,
                        help="Optional path to a pretrained checkpoint to warm-start.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    # --- data ---
    (t2w_tr, dwi_tr, adc_tr, cln_tr), labels_tr = build_split(args.data_root, "train", args.labels_train)
    (t2w_va, dwi_va, adc_va, cln_va), labels_va = build_split(args.data_root, "val", args.labels_val)

    train_ds = ProstateMultiModalDataset(t2w_tr, dwi_tr, adc_tr, cln_tr,
                                         label_dict=labels_tr,
                                         transform=default_train_transforms())
    val_ds = ProstateMultiModalDataset(t2w_va, dwi_va, adc_va, cln_va,
                                       label_dict=labels_va,
                                       transform=default_eval_transforms())

    def worker_init_fn(worker_id):
        s = args.seed + worker_id
        np.random.seed(s)
        import random as _r
        _r.seed(s)
        torch.manual_seed(s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # --- model ---
    model = MultimodalProstateModel(in_channel=1,
                                    label_category_dict={"binary_task": 1},
                                    dim=3).to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    class_weights = torch.tensor([args.w_neg, args.w_pos],
                                 device=device, dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- loop ---
    best_val_auc = -float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, class_weights,
                             args.focal_gamma, args.l1_lambda, device)
        va = evaluate(model, val_loader, class_weights, args.focal_gamma, device)

        print(
            f"[epoch {epoch:4d}] "
            f"train loss {tr['loss']:.4f} AUC {tr['auc']:.4f} MCC {tr['mcc']:.4f} "
            f"BalAcc {tr['balanced_accuracy']:.4f} | "
            f"val loss {va['loss']:.4f} AUC {va['auc']:.4f} MCC {va['mcc']:.4f} "
            f"BalAcc {va['balanced_accuracy']:.4f} Spec {va['specificity']:.4f} "
            f"Recall {va['recall']:.4f}"
        )

        if va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_val_auc.pt"))


if __name__ == "__main__":
    main()
