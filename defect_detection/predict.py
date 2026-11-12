import argparse
from pathlib import Path
import torch
from PIL import Image
from model import build_model
from dataset import EVAL_TRANSFORM
from utils import get_device

def main():
    parser = argparse.ArgumentParser(description="Predict apple quality/defect class for one image.")
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument("--model", default="outputs/checkpoints/best_apple_efficientnet_b0.pth",
                        help="Checkpoint path")
    args = parser.parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.model)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train first.")
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = EVAL_TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0]
    order = torch.argsort(probs, descending=True)
    print(f"\nImage: {image_path}")
    print(f"Predicted class: {class_names[order[0].item()]}")
    print(f"Confidence: {probs[order[0]].item()*100:.2f}%")
    print("\nAll class probabilities:")
    for idx in order:
        print(f"  {class_names[idx.item()]:20s} {probs[idx].item()*100:6.2f}%")

if __name__ == "__main__":
    main()
