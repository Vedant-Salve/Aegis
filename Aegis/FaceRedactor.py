import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import time

class FaceRedactor:
    def __init__(self):
        """Initialize the MTCNN detector once to save overhead."""
        print("Initializing MTCNN Detector...")
        self.detector = MTCNN()

    def load_image(self, image_path):
        """Loads an image and converts it to RGB for MTCNN/Matplotlib."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        return img

    def get_detections(self, image):
        """Performs face detection and returns bounding boxes."""
        # MTCNN needs RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_img)
        return [d['box'] for d in detections]

    def plot_preview(self, image, boxes):
        """Displays the image with ID labels using Matplotlib."""
        display_img = image.copy()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"ID: {i}"
            cv2.putText(display_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(display_img)
        plt.title(f"Detected {len(boxes)} faces")
        plt.axis('off')
        plt.show()

    def apply_redaction(self, image, boxes, indices_to_blur):
        """Applies blur with padding to avoid hard square edges."""
        final_img = image.copy()
        img_h, img_w = final_img.shape[:2]  # Get image dimensions
        
        for idx in indices_to_blur:
            if 0 <= idx < len(boxes):
                x, y, w, h = boxes[idx]
                
                #  Add Padding
                # We expand the box by ~20% on all sides
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                
                # Calculate new expanded coordinates, keeping them within image bounds
                x_new = max(0, x - pad_w)
                y_new = max(0, y - pad_h)
                w_new = min(img_w - x_new, w + 2 * pad_w)
                h_new = min(img_h - y_new, h + 2 * pad_h)
                
                # Extract the larger ROI
                roi = final_img[y_new:y_new+h_new, x_new:x_new+w_new]
                
                if roi.size > 0:
                    # Adaptive kernel relative to the NEW size
                    ksize = (int(w_new) | 1, int(h_new) | 1)
                    ksize = (max(51, ksize[0]), max(51, ksize[1]))
                    
                    blurred_roi = cv2.GaussianBlur(roi, ksize, 75)

                    # blurred_roi = cv2.GaussianBlur(blurred_roi, ksize, 75)
                    
                    # Create mask for the larger ROI
                    mask = np.zeros((h_new, w_new), dtype=np.uint8)
                    
                    # Draw ellipse: Keep it slightly smaller than the full padded box
                    # so the fade-out happens entirely inside the ROI
                    center = (w_new // 2, h_new // 2)
                    axes = (int(w * 0.5), int(h * 0.5)) # Use ORIGINAL w/h for ellipse size
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                    
                    # Heavy feathering for the soft edge
                    feather_ksize = (int(w_new//3) | 1, int(h_new//3) | 1) 
                    mask = cv2.GaussianBlur(mask, feather_ksize, 0)
                    
                    # Alpha Blending
                    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    alpha = mask_3d.astype(float) / 255.0
                    
                    roi_float = roi.astype(float)
                    blurred_float = blurred_roi.astype(float)
                    
                    blended_roi = (blurred_float * alpha) + (roi_float * (1.0 - alpha))
                    
                    # Place the larger blended ROI back into the image
                    final_img[y_new:y_new+h_new, x_new:x_new+w_new] = blended_roi.astype(np.uint8)
                    
        return final_img




    