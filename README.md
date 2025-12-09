# EnhanceNet - GAN-Based Image Super-Resolution & Deblurring

The web app generates Super Resolution images from low-resolution inputs and fixes blurred images using advanced deep learning models. Traditional techniques for image upscaling result in distorted or reduced visual quality. This project combines multiple state-of-the-art models and enhancement techniques to deliver superior results.

## Features

- **🚀 Multiple Enhancement Modes**:
  - **Standard**: Fast processing with EDSR model for general upscaling
  - **High Quality**: MSRN model for superior detail preservation
  - **Deblur + Upscale**: Aggressive deblurring with unsharp masking and sharpening
  
- **📏 Flexible Upscaling**: Choose 2x, 3x, or 4x upscaling factors

- **🔧 Advanced Processing**:
  - Intelligent detail reconstruction
  - Unsharp masking for deblurring
  - Automatic sharpness and contrast enhancement
  - Side-by-side before/after comparison
  
- **🎨 User-Friendly Interface**: 
  - Easy-to-use Streamlit web interface
  - Real-time preview with original and enhanced images
  - Download enhanced images directly
  
- **📁 Multiple Formats**: Supports JPG, PNG, and JPEG formats

## Requirements

- Python 3.10 or higher (tested with Python 3.13)
- All dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd EnhanceNet-GAN-Based-Image-Super-Resolution
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Application (Recommended)

Run the Streamlit web app for an interactive interface:

```bash
streamlit run app.py
```

This will:
- Open a web interface in your browser (usually at `http://localhost:8501`)
- Allow you to upload images (JPG, PNG, JPEG)
- Click "Upscale Now" to enhance your image using the EDSR model
- View both the original and upscaled images side-by-side

### Option 2: Python Script

Use `inference.py` for programmatic access:

```python
from inference import predict
from PIL import Image

# Load your image
img = Image.open('your_image.jpg')

# Get the super-resolution result (2x upscaling)
upscaled_img = predict(img)

# Save the result
upscaled_img.save('upscaled_image.jpg')
```

### Option 3: Jupyter Notebook

For experimentation or training:

```bash
jupyter notebook srgan_training.ipynb
```

## Demo link
https://www.loom.com/share/7703a64ddf454b6ab4985724b1a84bb9

## Technical Details

### Models
- **EDSR (Enhanced Deep Super-Resolution)**: Fast and efficient model for general upscaling
- **MSRN (Multi-Scale Residual Network)**: Advanced model with better detail preservation

### Enhancement Techniques
- **Super-Resolution**: Deep learning-based upscaling (2x, 3x, or 4x)
- **Unsharp Masking**: Advanced deblurring technique for fixing blurry images
- **Adaptive Sharpening**: Intelligent edge enhancement
- **Contrast Optimization**: Automatic contrast adjustment for better clarity

### Technical Stack
- **Framework**: PyTorch with torchvision
- **Image Processing**: OpenCV, PIL (Pillow)
- **Web Interface**: Streamlit
- **Pre-trained weights**: From Hugging Face Hub

### Performance
- First run downloads model weights (~10-20MB depending on mode)
- Processing time: 2-10 seconds for 500x500 images (varies by hardware and mode)
- GPU acceleration supported (CUDA/Metal on Apple Silicon)
- Subsequent runs are faster with cached models

## Notes

- For GPU acceleration, ensure you have CUDA installed and a compatible PyTorch version
- On Apple Silicon Macs, PyTorch will automatically use Metal acceleration for faster processing
- Larger images will take longer to process but will yield better quality results

## Troubleshooting

If you encounter installation issues:
- Make sure you're using Python 3.10 or higher
- Try updating pip: `pip install --upgrade pip`
- For Apple Silicon Macs, the current setup should work out of the box

## License

This project uses open-source models and libraries. Please check individual component licenses for usage restrictions. 
