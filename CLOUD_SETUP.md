# Cloud GPU Setup Guide

## Option 1: Google Cloud Platform (GCP)

### Step 1: Create a GPU Instance

1. **Go to Google Cloud Console**: https://console.cloud.google.com
2. **Enable Compute Engine API** (if not already enabled)
3. **Create VM Instance**:
   - Go to Compute Engine → VM instances → Create
   - **Name**: `gpu-training-vm`
   - **Region**: Choose one with GPU availability (e.g., `us-central1`, `us-west1`)
   - **Machine type**: 
     - For T4 GPU: `n1-standard-4` (4 vCPU, 15GB RAM) + 1x NVIDIA T4
     - For V100: `n1-standard-8` (8 vCPU, 30GB RAM) + 1x NVIDIA V100
   - **GPU**: Add GPU → NVIDIA T4 (cheapest) or V100 (faster)
   - **Boot disk**: 
     - OS: Ubuntu 22.04 LTS
     - Size: 50GB (minimum)
   - **Firewall**: Allow HTTP/HTTPS traffic (optional, for Jupyter)

### Step 2: Install Dependencies

SSH into your instance, then run:

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install CUDA drivers (if not pre-installed)
# For Ubuntu 22.04, NVIDIA drivers are usually pre-installed on GPU images

# Install Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Verify GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 3: Upload Your Code

**Option A: Using gcloud CLI (from your local machine)**
```bash
# Install gcloud CLI if you haven't
# Then copy files:
gcloud compute scp main.py requirements.txt gpu-training-vm:~/
```

**Option B: Using Git**
```bash
# On the VM, clone your repo or upload via GitHub
git clone <your-repo-url>
# or upload via Google Cloud Console file browser
```

**Option C: Using Cloud Storage**
```bash
# Upload to Cloud Storage, then download on VM
gsutil cp main.py gs://your-bucket/
# On VM:
gsutil cp gs://your-bucket/main.py ./
```

### Step 4: Run Training

```bash
# Run your training script
python3 main.py

# Or run in background (so you can disconnect)
nohup python3 main.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Step 5: Download Results

```bash
# Download trained model
gcloud compute scp gpu-training-vm:~/tissue_damage_model.pth ./

# Download logs
gcloud compute scp gpu-training-vm:~/training.log ./
```

### Step 6: Stop/Delete Instance (IMPORTANT - to save money!)

```bash
# Stop instance (can restart later)
gcloud compute instances stop gpu-training-vm

# Delete instance (permanent, but saves money)
gcloud compute instances delete gpu-training-vm
```

---

## Option 2: Google Colab (Easiest - Free GPU!)

1. **Go to**: https://colab.research.google.com
2. **Create new notebook**
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Upload your `main.py`** or copy-paste code into cells
5. **Install dependencies**:
   ```python
   !pip install torch torchvision medmnist scikit-learn matplotlib tqdm
   ```
6. **Run your code** - GPU is automatically detected!

**Note**: Colab free tier has usage limits (~12 hours/day), but perfect for testing.

---

## Option 3: AWS EC2 GPU Instance

1. **Launch EC2 Instance**:
   - AMI: Deep Learning AMI (Ubuntu) - has PyTorch pre-installed
   - Instance type: `g4dn.xlarge` (T4 GPU) or `p3.2xlarge` (V100)
   - Security group: Allow SSH (port 22)

2. **SSH and run**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   # PyTorch is usually pre-installed on Deep Learning AMI
   pip install medmnist scikit-learn matplotlib tqdm
   python3 main.py
   ```

---

## Cost Estimates

- **GCP T4 GPU**: ~$0.35-0.50/hour
- **GCP V100 GPU**: ~$2.50-3.00/hour
- **AWS g4dn.xlarge**: ~$0.50/hour
- **Google Colab**: FREE (with limits)

**Your training**: ~1-2 hours on GPU = **$0.50-1.00** total cost

---

## Tips to Save Money

1. **Use preemptible/spot instances** (60-80% cheaper)
2. **Stop instance immediately** after training
3. **Start with Colab** (free) to test
4. **Monitor usage** in cloud console
5. **Set up billing alerts** to avoid surprises

---

## Quick Start Script (GCP)

Save this as `setup_gpu.sh` and run on your VM:

```bash
#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install --upgrade pip
pip3 install torch torchvision medmnist scikit-learn matplotlib tqdm
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

