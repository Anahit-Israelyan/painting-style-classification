import json

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True)
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True)
    })

add_markdown("# 01 - EDA and Split\n\nThis notebook performs Exploratory Data Analysis and splits the data into train, val, and test sets. It has been made fully local-friendly and decoupled from Colab dependencies.")

add_markdown("## A. Project Setup")
add_code("""\
import os
import sys
from pathlib import Path

# Robust Project Root Detection
_CURRENT_DIR = Path(os.getcwd()).resolve()
PROJECT_ROOT = _CURRENT_DIR
while PROJECT_ROOT.name != 'image_classification_project' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if PROJECT_ROOT.name != 'image_classification_project':
    if (_CURRENT_DIR / 'image_classification_project').exists():
        PROJECT_ROOT = _CURRENT_DIR / 'image_classification_project'
    else:
        PROJECT_ROOT = _CURRENT_DIR

# Add src to python path for future imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define Directories
RAW_IMAGE_DIR = PROJECT_ROOT.parent / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Create output directories if they do not exist
for d in [PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set random seed
import random
import numpy as np
import torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

print(f"Project Root: {PROJECT_ROOT}")
print(f"Raw Image Dir: {RAW_IMAGE_DIR}")
""")

add_markdown("## B. Class Distribution")
add_code("""\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class_counts = []
if RAW_IMAGE_DIR.exists():
    for class_dir in RAW_IMAGE_DIR.iterdir():
        if class_dir.is_dir():
            count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS)
            class_counts.append({'class': class_dir.name, 'count': count})

df_counts = pd.DataFrame(class_counts).sort_values('class')
total_images = df_counts['count'].sum()
df_counts['percentage'] = (df_counts['count'] / total_images) * 100

print(f"Total number of images: {total_images}")
print(f"Number of classes: {len(df_counts)}")
if df_counts['count'].min() > 0:
    print(f"Class imbalance ratio (max/min): {df_counts['count'].max() / df_counts['count'].min():.2f}")

display(df_counts)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_counts, x='class', y='count')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_class_distribution.png')
plt.show()
""")

add_markdown("## C. Visual Inspection")
add_code("""\
from PIL import Image, ImageOps

plt.figure(figsize=(15, 10))
for i, row in enumerate(df_counts.itertuples()):
    class_name = row._1
    class_dir = RAW_IMAGE_DIR / class_name
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    
    if images:
        img_path = random.choice(images)
        try:
            # Handle EXIF orientation
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            
            plt.subplot(2, 4, i + 1)
            plt.imshow(img)
            plt.title(f"{class_name}\\nSize: {img.size}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_random_samples_per_class.png')
plt.show()
""")

add_markdown("## D. Integrity and Dimension Scan")
add_code("""\
import os

inventory = []
corrupted = []

for class_dir in RAW_IMAGE_DIR.iterdir():
    if not class_dir.is_dir():
        continue
        
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            inventory.append({
                'path': img_path.resolve().as_posix(),
                'rel_path': img_path.relative_to(PROJECT_ROOT.parent).as_posix(),
                'filename': img_path.name,
                'label': class_dir.name,
                'width': w,
                'height': h,
                'aspect_ratio': w / h,
                'suffix': img_path.suffix.lower()
            })
        except Exception as e:
            corrupted.append({
                'path': img_path.resolve().as_posix(),
                'rel_path': img_path.relative_to(PROJECT_ROOT.parent).as_posix(),
                'filename': img_path.name,
                'label': class_dir.name,
                'error': str(e)
            })

df_inventory = pd.DataFrame(inventory)
df_corrupted = pd.DataFrame(corrupted)

# Save corrupted using relative paths
if not df_corrupted.empty:
    df_corrupted_save = df_corrupted.drop(columns=['path'])
    df_corrupted_save.to_csv(SPLITS_DIR / 'corrupted_files.csv', index=False)
    print(f"Found {len(df_corrupted)} corrupted images. Saved to corrupted_files.csv")

# Save inventory using relative paths
if not df_inventory.empty:
    df_inventory_save = df_inventory.drop(columns=['path'])
    df_inventory_save.to_csv(SPLITS_DIR / 'image_inventory.csv', index=False)
    
    print("Dimension Summary:")
    display(df_inventory[['width', 'height', 'aspect_ratio']].describe())
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_inventory, x='width', y='height', hue='label', alpha=0.5)
    plt.title('Image Dimensions Scatter Plot')
    plt.savefig(FIGURES_DIR / '03_image_dimensions_scatter.png')
    plt.show()
    
    # Aspect ratio histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_inventory, x='aspect_ratio', bins=50)
    plt.title('Aspect Ratio Distribution')
    plt.savefig(FIGURES_DIR / '04_aspect_ratio_distribution.png')
    plt.show()
""")

add_markdown("""\
## E. EDA Conclusion
- **Dataset is imbalanced**: Accuracy alone is not enough. We must use Macro-F1 and balanced accuracy metrics. During training, CrossEntropyLoss with class weights should be used.
- **Images have mixed aspect ratios**: Padding to square before resizing is better than naive stretching to preserve the original painting composition and brushstrokes.
- **Resolution**: Use 128x128 if resources allow because painting style depends heavily on texture and composition.
""")

add_markdown("## F. Stratified Split")
add_code("""\
from sklearn.model_selection import train_test_split
import shutil

if not df_inventory.empty:
    X = df_inventory['rel_path'].values
    y = df_inventory['label'].values
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    manifest_rows = []
    
    # Clear and recreate folders
    for split_name in ['train', 'val', 'test']:
        split_dir = PROCESSED_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        
        X_split, y_split = splits[split_name]
        
        for rel_path_str, label in zip(X_split, y_split):
            src_path = PROJECT_ROOT.parent / rel_path_str
            
            class_dir = split_dir / label
            class_dir.mkdir(exist_ok=True)
            
            dst_path = class_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            
            manifest_rows.append({
                'source_path': rel_path_str, # relative to dataset root
                'processed_path': dst_path.relative_to(PROJECT_ROOT).as_posix(), # relative to project root
                'filename': src_path.name,
                'label': label,
                'split': split_name
            })
            
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(SPLITS_DIR / 'split_manifest.csv', index=False)
    
    # Crosstab
    df_crosstab = pd.crosstab(df_manifest['label'], df_manifest['split'], margins=True)
    df_crosstab.to_csv(SPLITS_DIR / 'split_crosstab.csv')
    print("Split Crosstab:")
    display(df_crosstab)
    
    # Class mappings
    classes = sorted(df_inventory['label'].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    
    import json
    with open(SPLITS_DIR / 'class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    with open(SPLITS_DIR / 'idx_to_class.json', 'w') as f:
        json.dump(idx_to_class, f, indent=2)
""")

add_markdown("""\
## G. Preprocessing Helper
The decision for future training transforms:
- **Training transform**: PadToSquare -> Resize(128,128) -> RandomHorizontalFlip -> RandomRotation(10) -> ColorJitter -> ToTensor
- **Validation/test transform**: PadToSquare -> Resize(128,128) -> ToTensor

This `PadToSquare` is defined here and will be moved to `src/data.py` for actual usage.
""")

add_code("""\
class PadToSquare:
    \"\"\"Pad an image to a square, preserving its aspect ratio.\"\"\"
    def __init__(self, fill=0):
        self.fill = fill
        
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

# Example Usage
if not df_inventory.empty:
    sample_path = PROJECT_ROOT.parent / df_inventory.iloc[0]['rel_path']
    try:
        sample_img = Image.open(sample_path).convert("RGB")
        sample_img = ImageOps.exif_transpose(sample_img)
        padder = PadToSquare()
        padded_img = padder(sample_img)
        
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(sample_img)
        plt.title(f"Original: {sample_img.size}")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(padded_img)
        plt.title(f"Padded: {padded_img.size}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Skipping preprocessing helper test: {e}")
""")

nb = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(r'c:\Users\Anahi\OneDrive\Desktop\bostongene_project\Task_IML_2_data\image_classification_project\notebooks\01_eda_and_split.ipynb', 'w') as fh:
    json.dump(nb, fh, indent=2)

print("Notebook generated successfully.")
