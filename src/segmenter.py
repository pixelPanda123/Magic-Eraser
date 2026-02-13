import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

class ObjectSegmenter:
    def __init__(self, model_type="vit_b", checkpoint_path="models/sam_vit_b_01ec64.pth"):
        """
        Initializes SAM.
        model_type: 'vit_b' 
        """
        print(f"Loading SAM ({model_type})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model registry and weights
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        # Initialize the predictor
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image_array):
        """
        Pre-computes the image embeddings. 
        This makes subsequent 'clicks' on the same image nearly instant.
        """
        # image_array should be RGB
        self.predictor.set_image(image_array)

    def get_mask_at_point(self, point_coords):
        """
        Generates a mask based on a single click.
        point_coords: List or array [[x, y]]
        """
        input_point = np.array(point_coords)
        input_label = np.array([1])  # 1 means 'foreground' point

        # Predict returns masks, scores, and low-res logits
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True, # Returns 3 possible masks (e.g., shirt, person, whole)
        )
        
        # We take the mask with the highest confidence score
        best_mask = masks[np.argmax(scores)]
        return best_mask

if __name__ == '__main__':
    segmenter = ObjectSegmenter() 
    print(f"Model successfully loaded on: {segmenter.device}")

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    segmenter.set_image(dummy_img)
    print("Test encoding successful!")


