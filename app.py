import streamlit as st 
from PIL import Image
import torch
from super_image import EdsrModel, ImageLoader
from torchvision.transforms import ToPILImage

# Load model once at startup
@st.cache_resource
def load_model():
    with st.spinner('📥 Downloading model (first time only)...'):
        return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

def predict(img):
    model = load_model()
    # Convert PIL Image to tensor
    inputs = ImageLoader.load_image(img)
    # Upscale the image
    with torch.no_grad():
        preds = model(inputs)
    # Convert tensor back to PIL Image
    to_pil = ToPILImage()
    return to_pil(preds.squeeze(0))


st.title("Super Resolution GAN ")
st.subheader("Upload an image which you want to upscale")

# Info box
with st.expander("ℹ️ How it works"):
    st.write("""
    - This app uses the **EDSR (Enhanced Deep Super-Resolution)** model
    - Upscales images by **2x** while preserving quality
    - First run may take longer as the model downloads (~10MB)
    - Processing time depends on image size
    """)
    
st.info("💡 Tip: For best results, use images smaller than 1000x1000 pixels")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    if st.button('Upscale Now'):
        with st.spinner('🔄 Loading model and upscaling image... This may take a moment on first run.'):
            try:
                pred = predict(image)
                st.success('✅ Upscaling complete!')
                st.image(pred, caption='Upscaled Image', use_column_width=True)
            except Exception as e:
                st.error(f'❌ Error during upscaling: {str(e)}')
                st.write("Please try with a smaller image or a different format.")        