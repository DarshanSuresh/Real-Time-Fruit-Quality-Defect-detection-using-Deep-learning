from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)

class AppleImageDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples  # list of (path, class_name)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_name = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {path} ({exc})") from exc
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[class_name], str(path)

def discover_class_names(real_dir: Path):
    if not real_dir.exists():
        raise FileNotFoundError(
            f"Real image folder not found: {real_dir}. Set DATA_ROOT in config.py."
        )
    classes = sorted(p.name for p in real_dir.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found inside {real_dir}")
    return classes

def collect_real_samples(real_dir: Path, class_names):
    samples = []
    for class_name in class_names:
        files = image_files(real_dir / class_name)
        if not files:
            print(f"WARNING: no images found in {real_dir / class_name}")
        samples.extend((p, class_name) for p in files)
    if not samples:
        raise RuntimeError(f"No images found under {real_dir}")
    return samples

def collect_synthetic_samples(synthetic_dir: Path, class_names):
    samples = []
    for class_name in class_names:
        folder = synthetic_dir / class_name
        files = image_files(folder)
        if not folder.exists():
            print(f"WARNING: synthetic class folder missing: {folder}")
        samples.extend((p, class_name) for p in files)
    return samples
