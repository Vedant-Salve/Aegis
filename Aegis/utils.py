import cv2
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from PIL import Image

def show_comparison(original_path, cloaked_image):
        """Displays side-by-side comparison in the notebook."""
        orig_bgr = cv2.imread(original_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        clok_rgb = cv2.cvtColor(cloaked_image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(orig_rgb)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        ax[1].imshow(clok_rgb)
        ax[1].set_title("Generated Image")
        ax[1].axis('off')
        plt.show()

def save_image(image,folder_path="GeneratedImages/"):

    filename = f"Aegis{int(1e7*(time.time()-int(time.time())))}.png"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    full_path = os.path.join(folder_path, filename)
    

    if isinstance(image, Image.Image):
        image = np.array(image)
    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    elif hasattr(image, 'cpu'):
        image = image.squeeze(0).cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
        
    else:
        raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL Image or Numpy Array.")

    success = cv2.imwrite(full_path, image)
    
    if success:
        print(f"Saved: {full_path}")
        return full_path
    else:
        print(f"Failed to write image to {full_path}")
        return None