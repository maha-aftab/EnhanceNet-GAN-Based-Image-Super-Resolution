from PIL import Image, ImageEnhance
import torch
from super_image import EdsrModel, MsrnModel, ImageLoader
from torchvision.transforms import ToPILImage
import cv2
import numpy as np

def reduce_pixelation(image):
    """Remove pixel artifacts and smooth the image while preserving edges"""
    img_array = np.array(image)
    smoothed = cv2.bilateralFilter(img_array, d=5, sigmaColor=50, sigmaSpace=50)
    smoothed = cv2.medianBlur(smoothed, 3)
    return Image.fromarray(smoothed)

def advanced_deblur(image, strength='medium'):
    """
    Advanced deblurring using multiple techniques
    strength: 'light', 'medium', 'strong'
    """
    img_array = np.array(image).astype(np.float32)
    
    # Bilateral filtering to preserve edges
    denoised = cv2.bilateralFilter(img_array.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    denoised = denoised.astype(np.float32)
    
    # Adaptive sharpening
    if strength == 'light':
        kernel_size, sigma, amount = (3, 3), 0.8, 0.6
    elif strength == 'medium':
        kernel_size, sigma, amount = (5, 5), 1.0, 1.0
    else:
        kernel_size, sigma, amount = (5, 5), 1.2, 1.3
    
    blurred = cv2.GaussianBlur(denoised, kernel_size, sigma)
    sharpened = denoised + amount * (denoised - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Edge enhancement
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY) if len(sharpened.shape) == 3 else sharpened
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    if len(sharpened.shape) == 3:
        laplacian_3channel = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        edge_weight = 0.3 if strength == 'light' else (0.5 if strength == 'medium' else 0.7)
        enhanced = cv2.addWeighted(sharpened, 1.0, laplacian_3channel, edge_weight, 0)
    else:
        enhanced = sharpened
    
    return Image.fromarray(enhanced)

def enhance_image(image, enhance_sharpness=1.3, enhance_contrast=1.05, remove_artifacts=False):
    """Apply gentle enhancements"""
    if remove_artifacts:
        image = reduce_pixelation(image)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_sharpness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhance_contrast)
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.05)
    
    return image

def predict(img, mode="Standard", scale=2):
    """
    Upscale and enhance image with advanced processing
    
    Args:
        img: PIL Image
        mode: "Standard", "High Quality", "Deblur + Upscale", or "Fix Pixelation"
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
    
    # Apply mode-specific enhancements
    if mode == "Deblur + Upscale":
        # Advanced deblurring with medium strength
        result = advanced_deblur(result, strength='medium')
        result = reduce_pixelation(result)
        result = enhance_image(result, enhance_sharpness=1.15, enhance_contrast=1.08, remove_artifacts=False)
        
    elif mode == "Fix Pixelation":
        # Focus on removing pixel artifacts
        result = reduce_pixelation(result)
        result = advanced_deblur(result, strength='light')
        result = enhance_image(result, enhance_sharpness=1.1, enhance_contrast=1.03, remove_artifacts=False)
        
    elif mode == "High Quality":
        # Balanced enhancement with artifact removal
        result = reduce_pixelation(result)
        result = enhance_image(result, enhance_sharpness=1.2, enhance_contrast=1.05, remove_artifacts=False)
        
    else:  # Standard
        # Minimal processing
        result = enhance_image(result, enhance_sharpness=1.1, enhance_contrast=1.02, remove_artifacts=False)
    
    return result