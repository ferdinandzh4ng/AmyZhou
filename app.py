import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
device = None
label_mapping = None
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to return a result

def load_model():
    """Load the trained model from disk"""
    global model, device, label_mapping
    
    model_path = "tissue_damage_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    # weights_only=False is safe here since this is our own trained model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get model configuration
    num_classes = checkpoint.get('num_classes', 9)
    input_channels = checkpoint.get('input_channels', 1)
    
    # Initialize model architecture
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get label mapping
    label_mapping = checkpoint.get('label_mapping', {})
    if isinstance(label_mapping, dict):
        # Convert string keys to int if needed
        label_mapping = {int(k): v for k, v in label_mapping.items()}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_mapping}")

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model inference"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms (same as eval_tf in main.py)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        # Apply transforms
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor, None
    except Exception as e:
        return None, str(e)

def is_valid_image(image_bytes):
    """Validate that the uploaded file is a valid image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        image_bytes = file.read()
        
        # Validate it's a real image (not gibberish)
        if not is_valid_image(image_bytes):
            return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400
        
        # Preprocess image
        tensor, error = preprocess_image(image_bytes)
        if error:
            return jsonify({'error': f'Error processing image: {error}'}), 400
        
        # Run inference
        with torch.no_grad():
            tensor = tensor.to(device)
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted.item()
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'error': f'Low confidence ({confidence:.2%}). Unable to make a reliable prediction. Please upload a clearer tissue image.',
                'confidence': confidence
            }), 400
        
        # Get class name
        class_name = label_mapping.get(predicted_class, f"Class {predicted_class}")
        
        # Determine if tissue damage is present
        # PathMNIST has 9 classes representing different tissue types
        # Class 0 is typically "adipose tissue" (normal), others may indicate various conditions
        # For a simple binary assessment, we consider class 0 as normal, others as potentially abnormal
        # This is a simplified interpretation - adjust based on your specific use case
        has_damage = predicted_class != 0
        
        # Get all class probabilities for transparency
        all_probs = probabilities[0].cpu().numpy()
        top_classes = []
        for i, prob in enumerate(all_probs):
            if prob > 0.05:  # Show classes with >5% probability
                top_classes.append({
                    'class': i,
                    'name': label_mapping.get(i, f"Class {i}"),
                    'probability': float(prob)
                })
        top_classes.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': float(confidence),
            'has_tissue_damage': has_damage,
            'top_classes': top_classes[:3],  # Return top 3 classes
            'message': f'Prediction: {class_name} (Confidence: {confidence:.2%})'
        })
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Initializing tissue damage detection web app...")
    load_model()
    print("Starting Flask server...")
    # Use port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    port = int(os.environ.get('PORT', 5001))
    # Disable debug mode in production
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"Server will be available at http://localhost:{port}")
    app.run(debug=debug, host='0.0.0.0', port=port)

