from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import *
from dataset import (AppleImageDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM,
                     discover_class_names, collect_real_samples,
                     collect_synthetic_samples)
from model import build_model
from utils import set_seed, get_device, save_json

def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_y, all_pred = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _paths in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            preds = logits.argmax(dim=1)
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_correct += (preds == labels).sum().item()
            total_count += batch_n
            all_y.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1), all_y, all_pred

def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    class_names = discover_class_names(REAL_DIR)
    print("Classes:", class_names)
    if EXPECTED_CLASSES and set(class_names) != set(EXPECTED_CLASSES):
        print("NOTE: detected folders differ from EXPECTED_CLASSES; detected folders will be used.")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    real_samples = collect_real_samples(REAL_DIR, class_names)
    print(f"Real images found: {len(real_samples)}")

    # Split real images first. Synthetic images are added to TRAIN only, so synthetic
    # variants cannot leak into validation/test metrics.
    labels = [class_to_idx[name] for _, name in real_samples]
    train_real, holdout, y_train, y_holdout = train_test_split(
        real_samples, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_real, test_real, y_val, y_test = train_test_split(
        holdout, y_holdout, test_size=0.50, random_state=SEED, stratify=y_holdout
    )
    synthetic_samples = collect_synthetic_samples(SYNTHETIC_DIR, class_names)
    print(f"Synthetic images added to training only: {len(synthetic_samples)}")
    train_samples = train_real + synthetic_samples

    train_ds = AppleImageDataset(train_samples, class_to_idx, TRAIN_TRANSFORM)
    val_ds = AppleImageDataset(val_real, class_to_idx, EVAL_TRANSFORM)
    test_ds = AppleImageDataset(test_real, class_to_idx, EVAL_TRANSFORM)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_path = CHECKPOINT_DIR / "best_apple_efficientnet_b0.pth"
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, True
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, None, device, False
        )
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f}, acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "architecture": "EfficientNet-B0"
            }, best_path)

    elapsed = time.time() - start
    save_json(history, OUTPUT_DIR / "training_history.json")
    save_json({"classes": class_names, "class_to_idx": class_to_idx,
               "real_total": len(real_samples), "synthetic_train_only": len(synthetic_samples),
               "train_real": len(train_real), "validation_real": len(val_real),
               "test_real": len(test_real), "training_seconds": elapsed,
               "device": str(device)}, OUTPUT_DIR / "dataset_summary.json")

    # Plot training curves
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Apple EfficientNet-B0 Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Apple EfficientNet-B0 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=160)
    plt.close()

    # Final evaluation on held-out REAL images using best checkpoint
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device, False
    )
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    print("\nHeld-out REAL test accuracy:", round(test_acc, 4))
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    save_json({"test_accuracy": test_acc, "classification_report": report,
               "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
               "class_names": class_names}, OUTPUT_DIR / "test_metrics.json")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix — Apple Defects")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=35, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"\nSaved best model: {best_path}")
    print(f"Saved metrics and plots under: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
