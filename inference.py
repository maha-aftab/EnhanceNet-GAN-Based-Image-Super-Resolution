from PIL import Image, ImageEnhance
import torch
from super_image import EdsrModel, MsrnModel, ImageLoader
from torchvision.transforms import ToPILImage
import cv2
import numpy as np

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """Apply unsharp mask to enhance details and fix blur"""
    img_array = np.array(image)
    blurred = cv2.GaussianBlur(img_array, kernel_size, sigma)
    sharpened = float(amount + 1) * img_array - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)

def enhance_image(image, enhance_sharpness=1.5, enhance_contrast=1.1):
    """Apply additional enhancements to fix blurriness"""
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_sharpness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhance_contrast)
    return image

def predict(img, mode="Standard", scale=2):
    """
    Upscale and enhance image
    
    Args:
        img: PIL Image
        mode: "Standard", "High Quality", or "Deblur + Upscale"
        scale: Upscaling factor (2, 3, or 4)
    
    Returns:
        Enhanced PIL Image
    """
    # Load appropriate model
    if mode == "High Quality":
        model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=scale)
    else:
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    
    # Convert PIL Image to tensor
    inputs = ImageLoader.load_image(img)
    
    # Upscale the image
    with torch.no_grad():
        preds = model(inputs)
    
    # Convert tensor back to PIL Image
    to_pil = ToPILImage()
    result = to_pil(preds.squeeze(0))
    
    # Apply enhancements based on mode
    if mode == "Deblur + Upscale":
        result = unsharp_mask(result, kernel_size=(5, 5), sigma=1.0, amount=2.0)
        result = enhance_image(result, enhance_sharpness=1.8, enhance_contrast=1.15)
    elif mode == "High Quality":
        result = enhance_image(result, enhance_sharpness=1.3, enhance_contrast=1.05)
    else:
        result = enhance_image(result, enhance_sharpness=1.2, enhance_contrast=1.0)
    
    return result