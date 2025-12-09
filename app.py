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

def reduce_pixelation(image):
    """Remove pixel artifacts and smooth the image while preserving edges"""
    img_array = np.array(image)
    
    # Apply bilateral filter - preserves edges while smoothing flat areas
    # This removes pixel artifacts without losing detail
    smoothed = cv2.bilateralFilter(img_array, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Apply median filter to remove salt-and-pepper noise/pixel artifacts
    smoothed = cv2.medianBlur(smoothed, 3)
    
    return Image.fromarray(smoothed)

def advanced_deblur(image, strength='medium'):
    """
    Advanced deblurring using multiple techniques
    strength: 'light', 'medium', 'strong'
    """
    img_array = np.array(image).astype(np.float32)
    
    # Step 1: Bilateral filtering to preserve edges while denoising
    denoised = cv2.bilateralFilter(img_array.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    denoised = denoised.astype(np.float32)
    
    # Step 2: Adaptive sharpening based on strength
    if strength == 'light':
        kernel_size, sigma, amount = (3, 3), 0.8, 0.6
    elif strength == 'medium':
        kernel_size, sigma, amount = (5, 5), 1.0, 1.0
    else:  # strong
        kernel_size, sigma, amount = (5, 5), 1.2, 1.3
    
    # Create unsharp mask
    blurred = cv2.GaussianBlur(denoised, kernel_size, sigma)
    sharpened = denoised + amount * (denoised - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Step 3: Edge enhancement using Laplacian
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY) if len(sharpened.shape) == 3 else sharpened
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Blend original with edge-enhanced version
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
        # First remove pixel artifacts
        image = reduce_pixelation(image)
    
    # Gentle sharpness enhancement
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(enhance_sharpness)
    
    # Gentle contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(enhance_contrast)
    
    # Slight color enhancement
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.05)
    
    return image

def predict(img, mode="Standard", scale=2):
    """
    Upscale and enhance image with advanced processing
    mode: "Standard", "High Quality", "Deblur + Upscale", "Fix Pixelation"
    """
    # Choose model based on mode
    if mode == "High Quality":
        model = load_msrn_model(scale=scale)
    else:
        model = load_edsr_model(scale=scale)
    
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
        # Remove any pixel artifacts from upscaling
        result = reduce_pixelation(result)
        # Gentle final enhancement
        result = enhance_image(result, enhance_sharpness=1.15, enhance_contrast=1.08, remove_artifacts=False)
        
    elif mode == "Fix Pixelation":
        # Focus on removing pixel artifacts first
        result = reduce_pixelation(result)
        # Light deblurring
        result = advanced_deblur(result, strength='light')
        # Very gentle enhancement
        result = enhance_image(result, enhance_sharpness=1.1, enhance_contrast=1.03, remove_artifacts=False)
        
    elif mode == "High Quality":
        # Balanced enhancement with artifact removal
        result = reduce_pixelation(result)
        result = enhance_image(result, enhance_sharpness=1.2, enhance_contrast=1.05, remove_artifacts=False)
        
    else:  # Standard
        # Minimal processing - just gentle enhancement
        result = enhance_image(result, enhance_sharpness=1.1, enhance_contrast=1.02, remove_artifacts=False)
    
    return result


st.title("🎨 Super Resolution & Image Enhancement")
st.subheader("Upload an image to upscale and fix blurriness")

# Info box
with st.expander("ℹ️ How it works - Advanced Techniques"):
    st.write("""
    This app combines multiple AI and image processing techniques:
    
    **🧠 Deep Learning Models:**
    - **EDSR** (Enhanced Deep Super-Resolution): Fast and efficient upscaling
    - **MSRN** (Multi-Scale Residual Network): Superior detail preservation
    
    **🔬 Advanced Processing:**
    - **Bilateral Filtering**: Edge-preserving smoothing to remove pixel artifacts
    - **Adaptive Deblurring**: Multi-stage deblurring with edge enhancement
    - **Median Filtering**: Removes salt-and-pepper noise and pixel artifacts
    - **Laplacian Edge Enhancement**: Brings out fine details
    - **Adaptive Sharpening**: Smart sharpening without creating artifacts
    
    **📊 What's Fixed:**
    - ✅ Blurry images (out of focus, motion blur)
    - ✅ Pixelation and compression artifacts
    - ✅ Low resolution and lack of detail
    - ✅ Poor contrast and dull colors
    
    First run downloads models (~10-20MB). Processing takes 3-15 seconds depending on image size and mode.
    """)

# Settings sidebar
st.sidebar.header("⚙️ Enhancement Settings")

enhancement_mode = st.sidebar.selectbox(
    "Enhancement Mode",
    ["Standard", "High Quality", "Deblur + Upscale", "Fix Pixelation"],
    help="Choose the enhancement mode based on your needs"
)

scale_factor = st.sidebar.selectbox(
    "Upscale Factor",
    [2, 3, 4],
    index=0,
    help="How much to upscale the image (2x = double resolution)"
)

# Mode descriptions with better guidance
if enhancement_mode == "Standard":
    st.sidebar.info("⚡ **Standard**: Fast and gentle processing. Best for already good quality images that just need to be bigger.")
    
elif enhancement_mode == "High Quality":
    st.sidebar.info("✨ **High Quality**: MSRN model + artifact removal. Best for photos, portraits, and images where quality matters most.")
    
elif enhancement_mode == "Deblur + Upscale":
    st.sidebar.info("🔧 **Deblur + Upscale**: Advanced deblurring with bilateral filtering and edge enhancement. Best for out-of-focus or motion-blurred photos.")
    
else:  # Fix Pixelation
    st.sidebar.info("🎯 **Fix Pixelation**: Removes visible pixels and artifacts while smoothing. Best for pixelated images, compressed JPEGs, or low-quality screenshots.")

st.info("💡 Tip: For best results, use images smaller than 800x800 pixels. Larger images will take longer to process.")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display original image
    st.markdown("### 📸 Original Image")
    st.image(image, caption=f'Original: {image.size[0]} x {image.size[1]} pixels', use_column_width=True)
    st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
    
    if st.button('🚀 Enhance Image', type="primary"):
        with st.spinner(f'🔄 Processing with {enhancement_mode} mode... This may take a moment on first run.'):
            try:
                pred = predict(image, mode=enhancement_mode, scale=scale_factor)
                
                st.success(f'✅ Enhancement complete! Image upscaled {scale_factor}x and enhanced.')
                
                # Display enhanced image at TRUE pixel dimensions (no scaling)
                st.markdown("### ✨ Enhanced Image (True Pixel Dimensions)")
                st.image(pred, caption=f'Enhanced: {pred.size[0]} x {pred.size[1]} pixels')
                st.write(f"**Size:** {pred.size[0]} x {pred.size[1]} pixels")
                st.info(f"📏 This image is displayed at its actual {pred.size[0]}x{pred.size[1]} pixel dimensions. Scroll to see the full resolution!")
                
                # Show comparison info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Width", f"{image.size[0]}px")
                with col2:
                    st.metric("Enhanced Width", f"{pred.size[0]}px", f"+{pred.size[0] - image.size[0]}px")
                with col3:
                    st.metric("Scale Factor", f"{scale_factor}x")
                
                # Download button
                buf = io.BytesIO()
                pred.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Option to see side-by-side comparison
                with st.expander("👁️ View Side-by-Side Comparison"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(image, caption='Original', use_column_width=True)
                    with col_b:
                        st.image(pred, caption='Enhanced', use_column_width=True)
                
            except Exception as e:
                st.error(f'❌ Error during enhancement: {str(e)}')
                st.write("Please try with a smaller image, different mode, or lower scale factor.")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())        