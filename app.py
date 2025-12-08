import streamlit as st 
from PIL import Image, ImageEnhance, ImageFilter
import torch
from super_image import EdsrModel, MsrnModel, ImageLoader
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
import io
import traceback

# Load models once at startup
@st.cache_resource
def load_edsr_model(scale=2):
    with st.spinner('📥 Downloading EDSR model (first time only)...'):
        return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)

@st.cache_resource
def load_msrn_model(scale=2):
    with st.spinner('📥 Downloading MSRN model (first time only)...'):
        return MsrnModel.from_pretrained('eugenesiow/msrn', scale=scale)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Apply unsharp mask to enhance details and fix blur"""
    img_array = np.array(image)
    blurred = cv2.GaussianBlur(img_array, kernel_size, sigma)
    sharpened = float(amount + 1) * img_array - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.abs(img_array - blurred) < threshold
        np.copyto(sharpened, img_array, where=low_contrast_mask)
    
    return Image.fromarray(sharpened)

def enhance_image(image, enhance_sharpness=1.5, enhance_contrast=1.1):
    """Apply additional enhancements to fix blurriness"""
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_sharpness)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhance_contrast)
    
    return image

def predict(img, mode="Standard", scale=2):
    """
    Upscale and enhance image
    mode: "Standard", "High Quality", "Deblur + Upscale"
    """
    if mode == "High Quality":
        # Use MSRN model for better detail preservation
        model = load_msrn_model(scale=scale)
    else:
        # Use EDSR model (faster)
        model = load_edsr_model(scale=scale)
    
    # Convert PIL Image to tensor
    inputs = ImageLoader.load_image(img)
    
    # Upscale the image
    with torch.no_grad():
        preds = model(inputs)
    
    # Convert tensor back to PIL Image
    to_pil = ToPILImage()
    result = to_pil(preds.squeeze(0))
    
    # Apply deblurring and enhancement based on mode
    if mode == "Deblur + Upscale":
        # Apply unsharp mask for deblurring
        result = unsharp_mask(result, kernel_size=(5, 5), sigma=1.0, amount=2.0)
        # Additional enhancement
        result = enhance_image(result, enhance_sharpness=1.8, enhance_contrast=1.15)
    elif mode == "High Quality":
        # Light enhancement for high quality mode
        result = enhance_image(result, enhance_sharpness=1.3, enhance_contrast=1.05)
    else:
        # Standard mode - minimal enhancement
        result = enhance_image(result, enhance_sharpness=1.2, enhance_contrast=1.0)
    
    return result


st.title("🎨 Super Resolution & Image Enhancement")
st.subheader("Upload an image to upscale and fix blurriness")

# Info box
with st.expander("ℹ️ How it works"):
    st.write("""
    This app uses advanced deep learning models to:
    - **Upscale images** while preserving and enhancing details
    - **Fix blurred images** through intelligent reconstruction
    - **Sharpen and enhance** overall image quality
    
    **Models used:**
    - **EDSR** (Enhanced Deep Super-Resolution): Fast and efficient
    - **MSRN** (Multi-Scale Residual Network): Better detail preservation
    - **Unsharp Masking**: Advanced deblurring technique
    
    First run may take longer as models download (~10-20MB total)
    """)

# Settings sidebar
st.sidebar.header("⚙️ Enhancement Settings")

enhancement_mode = st.sidebar.selectbox(
    "Enhancement Mode",
    ["Standard", "High Quality", "Deblur + Upscale"],
    help="Choose the enhancement mode based on your needs"
)

scale_factor = st.sidebar.selectbox(
    "Upscale Factor",
    [2, 3, 4],
    index=0,
    help="How much to upscale the image (2x = double resolution)"
)

# Mode descriptions
if enhancement_mode == "Standard":
    st.sidebar.info("⚡ **Standard**: Fast processing with good quality. Best for general upscaling.")
elif enhancement_mode == "High Quality":
    st.sidebar.info("✨ **High Quality**: Uses MSRN model for superior detail preservation. Slightly slower.")
else:
    st.sidebar.info("🔧 **Deblur + Upscale**: Aggressive deblurring and sharpening. Best for fixing blurry images.")

st.info("💡 Tip: For best results, use images smaller than 800x800 pixels. Larger images will take longer to process.")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Original ({image.size[0]}x{image.size[1]})', use_column_width=True)
        st.write(f"**Original size:** {image.size[0]} x {image.size[1]} pixels")
    
    if st.button('🚀 Enhance Image', type="primary"):
        with st.spinner(f'🔄 Processing with {enhancement_mode} mode... This may take a moment on first run.'):
            try:
                pred = predict(image, mode=enhancement_mode, scale=scale_factor)
                
                with col2:
                    st.image(pred, caption=f'Enhanced ({pred.size[0]}x{pred.size[1]})', use_column_width=True)
                    st.write(f"**Enhanced size:** {pred.size[0]} x {pred.size[1]} pixels")
                
                st.success(f'✅ Enhancement complete! Image upscaled {scale_factor}x and enhanced.')
                
                # Download button
                buf = io.BytesIO()
                pred.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f'❌ Error during enhancement: {str(e)}')
                st.write("Please try with a smaller image, different mode, or lower scale factor.")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())        