# Tissue Damage Detection Web App

A simple web application for detecting tissue damage in medical images using a trained ResNet18 model.

## Features

- **Image Upload**: Drag-and-drop or click to upload tissue images
- **AI Analysis**: Automatic tissue damage assessment using deep learning
- **Confidence Checking**: Only returns results when confidence is above 50%
- **Input Validation**: Rejects invalid files (non-images, corrupted files, etc.)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model file `tissue_damage_model.pth` is in the project directory.

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5001
```

Note: The app uses port 5001 by default to avoid conflicts with macOS AirPlay Receiver. You can change this by setting the `PORT` environment variable.

## Usage

1. Click the upload area or drag and drop an image file
2. Click "Analyze Image" to process the image
3. View the results showing whether tissue damage was detected

## Technical Details

- **Model**: ResNet18 (grayscale, 9 classes)
- **Input**: Images are converted to grayscale and resized to 224x224
- **Confidence Threshold**: 50% minimum confidence required for predictions
- **Validation**: All uploaded files are validated to ensure they are valid images

## File Structure

```
.
├── app.py                    # Flask backend
├── main.py                   # Model training script
├── tissue_damage_model.pth   # Trained model
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Frontend HTML
└── uploads/                 # Temporary upload directory (auto-created)
```

