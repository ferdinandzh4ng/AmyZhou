# Google Colab Setup Guide

## Quick Start (3 Steps)

### Step 1: Enable GPU
1. Go to **Runtime → Change runtime type**
2. Select **GPU (T4)** from the Hardware accelerator dropdown
3. Click **Save**

### Step 2: Install Dependencies
In a new cell, run:
```python
!pip install torch torchvision medmnist scikit-learn matplotlib tqdm
```

### Step 3: Upload and Run
**Option A: Upload main.py file**
1. Click the folder icon in the left sidebar
2. Click the upload button (📤)
3. Select your `main.py` file
4. Run: `!python main.py`

**Option B: Copy-paste code directly**
Just copy your entire `main.py` code into a Colab cell and run it.

---

## Complete Step-by-Step

### 1. Open Google Colab
- Go to: https://colab.research.google.com
- Click **File → New notebook**

### 2. Enable GPU
```
Runtime → Change runtime type → GPU (T4) → Save
```

### 3. Install packages (first cell)
```python
!pip install torch torchvision medmnist scikit-learn matplotlib tqdm
```

### 4. Verify GPU (second cell)
```python
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

### 5. Upload main.py
**Method 1: File browser**
- Click folder icon (📁) in left sidebar
- Click upload button (📤)
- Select `main.py` from your computer

**Method 2: Code upload**
```python
from google.colab import files
uploaded = files.upload()  # Select main.py when prompted
```

### 6. Run training
```python
!python main.py
```

Or if you want to see output in real-time:
```python
import subprocess
result = subprocess.run(['python', 'main.py'], capture_output=False, text=True)
```

---

## Download Results

After training completes:

```python
from google.colab import files

# Download trained model
files.download('tissue_damage_model.pth')

# Download human benchmark images (if exported)
files.download('human_benchmark_images/ground_truth_labels.csv')
```

---

## Tips for Colab

1. **Session timeout**: Colab free tier disconnects after ~90 minutes of inactivity
   - Solution: Keep the tab active or upgrade to Colab Pro

2. **File persistence**: Files are deleted when runtime disconnects
   - Solution: Download important files or save to Google Drive

3. **Save to Drive** (optional):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Save to: /content/drive/MyDrive/your_folder/
   ```

4. **Monitor GPU usage**:
   ```python
   !nvidia-smi
   ```

---

## Expected Performance on Colab

- **CPU**: ~1.6 hours per epoch
- **GPU (T4)**: ~5-10 minutes per epoch
- **Total training**: ~30-60 minutes (vs 8+ hours on CPU)

---

## Troubleshooting

**"CUDA not available"**
- Make sure GPU is enabled: Runtime → Change runtime type → GPU

**"Module not found"**
- Run the pip install cell again

**Session disconnected**
- Re-run all cells from the top
- Consider saving checkpoints during training

**Out of memory**
- Reduce batch size in main.py (change `batch_size = 64` to `batch_size = 32`)

