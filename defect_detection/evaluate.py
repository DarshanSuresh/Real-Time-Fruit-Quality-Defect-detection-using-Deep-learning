"""Evaluate a trained checkpoint on a folder with one subfolder per class.
Use a separate untouched real-image test directory to get an unbiased estimate."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import AppleImageDataset, EVAL_TRANSFORM, image_files
from model import build_model
from utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder containing class subfolders")
    parser.add_argument("--model", default="outputs/checkpoints/best_apple_efficientnet_b0.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    data_dir, model_path = Path(args.data), Path(args.model)
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint["class_names"]
    mapping = {name: i for i, name in enumerate(classes)}
    samples = []
    for name in classes:
        samples += [(p, name) for p in image_files(data_dir / name)]
    if not samples:
        raise RuntimeError(f"No matching images found in {data_dir}")
    loader = DataLoader(AppleImageDataset(samples, mapping, EVAL_TRANSFORM),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            output = model(images.to(device)).argmax(1).cpu().tolist()
            truth.extend(labels.tolist())
            pred.extend(output)
    print(classification_report(truth, pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(truth, pred, labels=list(range(len(classes)))))

if __name__ == "__main__":
    main()
