import cv2 
import numpy as np 
from PIL import Image 
from simple_lama_inpainting import SimpleLama
import torch

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

print(f"Running on device: {device}")
class ImageInpainter: 
    def __init__(self):
        print("Initializing the model...")
        self.model = SimpleLama()
    
    def remove_object(self, image_path, mask_path):
        ''' Processes the image and removes the masked area. '''
        #load images 
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") #convert to grayscale.L-mode

        print(f"Processing inpainting for {image_path}..")
        result = self.model(image, mask)

        return result 


if __name__ == "__main__":
    inpainter = ImageInpainter()
    print("Inpainter module ready!")






