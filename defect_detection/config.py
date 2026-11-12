from pathlib import Path

# Set this to the folder containing apple_real_images and apple_synthethic_images.
# Example: DATA_ROOT = Path("/Users/you/datasets/apple_images")
DATA_ROOT = Path("/Users/darshansuresh/Darshan Suresh/Deep_Learning_Project/Dataset/apple_images")

REAL_DIR = DATA_ROOT / "apple_real_images"
SYNTHETIC_DIR = DATA_ROOT / "apple_synthethic_images"

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

# Classes are discovered from the real-image subfolders. These names are expected:
EXPECTED_CLASSES = ["bruise_defect", "fresh", "rot_defect", "scab_defect"]
