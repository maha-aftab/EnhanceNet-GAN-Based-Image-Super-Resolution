from PIL import Image
import torch
from super_image import EdsrModel, ImageLoader
from torchvision.transforms import ToPILImage

def predict(img):
    # Load the pre-trained model
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    # Convert PIL Image to tensor
    inputs = ImageLoader.load_image(img)
    # Upscale the image
    with torch.no_grad():
        preds = model(inputs)
    # Convert tensor back to PIL Image
    to_pil = ToPILImage()
    return to_pil(preds.squeeze(0))