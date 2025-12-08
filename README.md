# EnhanceNet - GAN-Based Image Super-Resolution

The web app generates Super Resolution images from low-resolution inputs using deep learning. Various traditional techniques for image upscaling result in distorted or reduced visual quality images. Deep Learning provides better solutions to get optimized images. This project uses the EDSR (Enhanced Deep Super-Resolution) model, a state-of-the-art super-resolution network.

## Features

- **Web Interface**: Upload and enhance images through an easy-to-use Streamlit interface
- **High Quality**: Uses EDSR model for superior image quality
- **Fast Processing**: Optimized for quick inference
- **Multiple Formats**: Supports JPG, PNG, and JPEG formats

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

## Technical Details

- **Model**: EDSR (Enhanced Deep Super-Resolution)
- **Framework**: PyTorch
- **Upscaling Factor**: 2x (configurable)
- **Pre-trained weights**: From Hugging Face Hub
- On first run, the model will automatically download pre-trained weights (~10MB)
- Processing time depends on image size and your hardware

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
