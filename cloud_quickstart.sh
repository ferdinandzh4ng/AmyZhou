#!/bin/bash
# Quick setup script for cloud GPU instances
# Run this on your cloud VM after uploading your files

echo "=== Setting up GPU training environment ==="

# Update system
echo "Updating system packages..."
sudo apt-get update -qq

# Install Python and pip if needed
echo "Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-dev -qq

# Upgrade pip
pip3 install --upgrade pip -q

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# Install other dependencies
echo "Installing other packages..."
pip3 install -r requirements.txt -q

# Verify GPU
echo ""
echo "=== Verifying GPU setup ==="
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU device:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('Number of GPUs:', torch.cuda.device_count())
else:
    print('WARNING: CUDA not available! Check GPU drivers.')
"

echo ""
echo "=== Setup complete! ==="
echo "Run your training with: python3 main.py"

