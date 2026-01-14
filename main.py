# PathMNIST Tissue Damage Detection CNN model in PyTorch + MedMNIST
# - downloads dataset
# - trains a ResNet18 (grayscale) to identify tissue damage
# - evaluates on test set
# - prints accuracy + confusion matrix + classification report
# - exports the trained model
# - (optional) exports a small "human benchmark" image set + labels CSV

# -----------------------------
# 0) Install (run once)
# -----------------------------
# pip install torch torchvision medmnist scikit-learn matplotlib tqdm

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import medmnist
from medmnist import INFO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import csv

# -----------------------------
# 1) Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------
    # 2) Dataset (PathMNIST - Tissue Damage Detection)
    # -----------------------------
    data_flag = "pathmnist"  # identifies tissue damage (abnormal) vs normal tissue
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    # PathMNIST is 28x28 RGB; we convert to grayscale and resize to 224x224 to use ResNet18 well
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    print("\n[STATUS] Loading PathMNIST dataset...")
    train_ds = DataClass(split="train", transform=train_tf, download=True)
    val_ds   = DataClass(split="val",   transform=eval_tf,  download=True)
    test_ds  = DataClass(split="test",  transform=eval_tf,  download=True)
    print(f"[STATUS] Dataset loaded - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Labels: 0/1 (check mapping in INFO)
    print("Label mapping:", info["label"])  # e.g., {0:'...'; 1:'...'}

    # -----------------------------
    # 3) Model (ResNet18 for grayscale, 9 classes)
    # -----------------------------
    print("\n[STATUS] Initializing ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Convert first conv from RGB(3) to Grayscale(1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # PathMNIST has 9 classes -> 9 outputs
    num_classes = len(info["label"])
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    print(f"[STATUS] Model initialized and moved to {device}")

    # -----------------------------
    # 4) Training setup
    # -----------------------------
    print("\n[STATUS] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("[STATUS] Training setup complete - Loss: CrossEntropyLoss, Optimizer: Adam (lr=1e-4)")

    def run_epoch(loader, training: bool):
        model.train() if training else model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, leave=False):
            # MedMNIST labels usually come as shape (B, 1)
            labels = labels.squeeze().long()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = model(images)
                loss = criterion(outputs, labels)

                if training:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    # -----------------------------
    # 5) Train with early stopping
    # -----------------------------
    epochs = 5  # Max epochs; early stopping will stop earlier if no improvement (patience=3)
    patience = 3
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    print(f"\n[STATUS] Starting training for up to {epochs} epochs (patience={patience})...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[STATUS] Epoch {epoch}/{epochs} - Training...")
        train_start = time.time()
        train_loss, train_acc = run_epoch(train_loader, training=True)
        train_time = time.time() - train_start
        print(f"[STATUS] Epoch {epoch}/{epochs} - Validating...")
        val_start = time.time()
        val_loss, val_acc     = run_epoch(val_loader,   training=False)
        val_time = time.time() - val_start
        epoch_time = time.time() - epoch_start

        print(f"[STATUS] Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s (train: {train_time:.1f}s, val: {val_time:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"[STATUS] ✓ New best validation accuracy: {best_val_acc:.4f} - Model saved!")
        else:
            patience_counter += 1
            print(f"[STATUS] No improvement (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print("[STATUS] Early stopping triggered.")
                break

    print("\n[STATUS] Loading best model weights...")
    # Load best weights
    model.load_state_dict(best_state)
    model.to(device)
    print(f"[STATUS] Best model loaded (val_acc: {best_val_acc:.4f})")

    # -----------------------------
    # 6) Test evaluation
    # -----------------------------
    print("\n[STATUS] Evaluating on test set...")
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[STATUS] Processing test batches"):
            labels = labels.squeeze().long()
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_true.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    test_acc = (all_preds == all_true).mean()
    print(f"\n[STATUS] Test evaluation complete!")
    print(f"[STATUS] TEST ACCURACY: {test_acc:.4f}")

    print("\n[STATUS] Classification report:")
    print(classification_report(all_true, all_preds, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    print("\n[STATUS] Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Tissue Damage Detection)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Tissue Damage"])
    plt.yticks([0, 1], ["Normal", "Tissue Damage"])
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 7) Export trained model
    # -----------------------------
    print("\n[STATUS] Exporting trained model...")
    model_save_path = "tissue_damage_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'resnet18',
        'num_classes': num_classes,
        'input_channels': 1,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'label_mapping': info["label"],
    }, model_save_path)
    print(f"[STATUS] ✓ Model exported successfully to: {model_save_path}")
    print(f"[STATUS] Model can be loaded with: torch.load('{model_save_path}')")

    # -----------------------------
    # 8) (Optional) Export images for your HUMAN benchmark
    # -----------------------------
    # This creates a folder of PNGs + a CSV with ground truth labels.
    # Use these images in Google Forms and compare human accuracy vs AI.

    EXPORT_HUMAN_SET = True
    human_set_size = 40  # good number for volunteers

    if EXPORT_HUMAN_SET:
        print(f"\n[STATUS] Exporting {human_set_size} images for human benchmark...")
        out_dir = "human_benchmark_images"
        os.makedirs(out_dir, exist_ok=True)

        # Use RAW images without augmentation for exporting (convert RGB to grayscale)
        export_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        export_ds = DataClass(split="test", transform=export_tf, download=True)

        # Randomly sample indices
        indices = np.random.choice(len(export_ds), size=human_set_size, replace=False)
        print(f"[STATUS] Selected {human_set_size} random test images...")

        # Save images + labels CSV
        csv_path = os.path.join(out_dir, "ground_truth_labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "true_label", "label_name"])
            for i, idx in enumerate(indices):
                img, label = export_ds[idx]  # img: tensor [1, 28, 28] (grayscale)
                label_int = int(label.item())

                # Save as PNG (grayscale)
                img_np = img.squeeze().numpy()
                filename = f"img_{i:03d}.png"
                filepath = os.path.join(out_dir, filename)
                plt.imsave(filepath, img_np, cmap="gray")

                label_name = info["label"].get(str(label_int), f"Class {label_int}")
                writer.writerow([filename, label_int, label_name])

        print(f"[STATUS] ✓ Exported {human_set_size} images to: {out_dir}")
        print(f"[STATUS] ✓ Ground truth CSV saved to: {csv_path}")

    # -----------------------------
    # 9) (Optional) Evaluate AI on the SAME exported indices
    # -----------------------------
    # If you want the AI accuracy on the exact same human set, you can do it here.

    EVAL_AI_ON_HUMAN_SET = True
    if EXPORT_HUMAN_SET and EVAL_AI_ON_HUMAN_SET:
        print("\n[STATUS] Evaluating AI on exported human-benchmark set...")
        # Build a loader using the same indices but with eval transforms
        human_subset = Subset(test_ds, indices)
        human_loader = DataLoader(human_subset, batch_size=32, shuffle=False)

        model.eval()
        preds_h = []
        true_h = []
        with torch.no_grad():
            for images, labels in human_loader:
                labels = labels.squeeze().long()
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                preds_h.append(preds)
                true_h.append(labels.numpy())

        preds_h = np.concatenate(preds_h)
        true_h  = np.concatenate(true_h)

        acc_h = (preds_h == true_h).mean()
        print(f"[STATUS] AI accuracy on exported human-benchmark set: {acc_h:.4f}")

    print("\n[STATUS] All tasks completed!")