# Apple Defect Detection — EfficientNet-B0

A single-fruit image-classification project for **apple** images. It predicts one of four folder labels:

- `bruise_defect`
- `fresh`
- `rot_defect`
- `scab_defect`

This is **image classification**, not object detection or defect segmentation. It labels the whole image with one class.

## 1. Expected dataset layout

The project expects the directory names below (including the spelling `synthethic` used in the supplied dataset):

```text
apple_images/
├── apple_real_images/
│   ├── bruise_defect/
│   ├── fresh/
│   ├── rot_defect/
│   └── scab_defect/
└── apple_synthethic_images/
    ├── bruise_defect/
    ├── fresh/
    ├── rot_defect/
    └── scab_defect/
```

Images may be JPG, JPEG, PNG, BMP, WEBP, TIF, or TIFF. Each class folder should contain images of that class.

### Important spelling note
Your tree listed `rot_deffect` under synthetic images, but the real folder is `rot_defect`. Rename the synthetic folder to **`rot_defect`** so the code can find it. The code deliberately uses the spelling `apple_synthethic_images` from your tree; you may rename it and update `config.py` if preferred.

## 2. Environment setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Apple Silicon Mac, PyTorch can use MPS if supported. For NVIDIA GPU, install the PyTorch build suitable for your CUDA version from the official PyTorch installation selector.

## 3. Point to the dataset

By default, `config.py` expects `data/apple_images` relative to this project. Either copy your `apple_images` folder there, or change:

```python
DATA_ROOT = Path("data/apple_images")
```

to the absolute path of your dataset, for example:

```python
DATA_ROOT = Path("/Users/yourname/datasets/apple_images")
```

## 4. Train

```bash
python train.py
```

The script:
1. Reads the real apple images and discovers class names.
2. Splits real images into 70% training, 15% validation, and 15% test (stratified where possible).
3. Adds synthetic images **only to the training set**.
4. Fine-tunes pretrained EfficientNet-B0.
5. Saves the best checkpoint based on validation loss.
6. Evaluates once on the held-out real test images.
7. Saves training curves, confusion matrix, dataset summary, and metrics in `outputs/`.

The split is done on individual images. If several images come from the same physical apple or are near-duplicates, group them before splitting to avoid leakage.

If a class has too few real images for a stratified 70/15/15 split, the split may fail. Collect more real images per class or adapt the split logic.

## 5. Predict one image

```bash
python predict.py "/path/to/apple_photo.jpg"
```

Or specify a different checkpoint:
```bash
python predict.py "/path/to/apple_photo.jpg" --model outputs/checkpoints/best_apple_efficientnet_b0.pth
```

## 6. Evaluate another labeled folder

Provide a folder with the same four class subfolders, ideally a separate real-image test set:

```bash
python evaluate.py --data "/path/to/held_out_apple_test"
```

## 7. Outputs

```text
outputs/
├── checkpoints/
│   └── best_apple_efficientnet_b0.pth
├── plots/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
├── dataset_summary.json
├── test_metrics.json
└── training_history.json
```

## 8. Experiment settings

Change `config.py` to adjust `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, and image size. Run separate experiments for batch sizes 8, 16, and 32 if your paper requires comparisons. Keep the same real-image test split across experiments.

## 9. Research integrity and limitations

- Do not report a result until the code has actually run on your dataset.
- Synthetic images are training-only here; validation and test metrics use real images.
- High confidence is not proof that a prediction is correct.
- This baseline predicts one label per image; it does not return bounding boxes, defect locations, or severity levels.
- To detect defect location and severity, you would need bounding-box/segmentation annotations and an object-detection or segmentation model.
