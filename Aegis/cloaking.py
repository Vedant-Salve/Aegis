import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import random
from pytorch_msssim import ssim 
import time

class AegisCloakingEngine:
    def __init__(self, target_pool_path=None):
        """        
        Args:
            target_pool_path: Path to a directory containing images of other people. 
                              Used to find highly dissimilar targets.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Aegis Cloaking Engine Initialized on {self.device} ---")

        # 1. Feature Extractor
        resnet = models.resnet50(pretrained=True).to(self.device)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).eval()
        
        # Freeze gradients for the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Preprocessing (Decoupled for Resolution Preservation)
        # Converts PIL image to [0, 1] tensor without resizing
        self.to_tensor = transforms.ToTensor()
        
        # Normalization specific to the ResNet50 model expectations
        self.normalize_for_model = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        self.target_pool_path = target_pool_path

    def load_image_as_tensor(self, path):
        """Loads image in its original resolution as a [0, 1] tensor."""
        img = Image.open(path).convert('RGB')
        tensor_img = self.to_tensor(img).unsqueeze(0).to(self.device)
        return tensor_img

    def get_features(self, img_tensor):
        """Helper to resize, normalize, and extract features for any sized tensor."""
        # Interpolate dynamically resizes the tensor while keeping the operation differentiable
        resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        normalized = self.normalize_for_model(resized)
        return self.feature_extractor(normalized)

    def select_optimal_target(self, user_features, top_k=10):
        """
        Custom Target Selection:
        Calculates L2 distance for all images, takes the top K most dissimilar,
        and randomly selects one from that pool.
        """
        if not self.target_pool_path or not os.path.exists(self.target_pool_path):
            print("No target pool provided. Using random noise target (High Privacy).")
            return None

        print("Scanning target pool for highly dissimilar identities...")
        distances = []
        valid_exts = ('.jpg', '.jpeg', '.png')
        
        for filename in os.listdir(self.target_pool_path):
            if not filename.lower().endswith(valid_exts):
                continue
                
            path = os.path.join(self.target_pool_path, filename)
            try:
                # Load candidate target and extract features
                t_tensor = self.load_image_as_tensor(path)
                with torch.no_grad():
                    t_features = self.get_features(t_tensor)
                
                # Calculate L2 distance
                dist = torch.dist(user_features, t_features).item()
                distances.append((dist, path))
            except Exception as e:
                continue
        
        if not distances:
            return None

        # Sort by distance descending (most dissimilar at index 0)
        distances.sort(key=lambda x: x[0], reverse=True)
        
        # Determine how many candidates make up the "top pool"
        pool_size = min(top_k, len(distances))
        top_candidates = distances[:pool_size]
        
        # Randomly select one target from the top dissimilar pool
        chosen_dist, best_target_path = random.choice(top_candidates)
        
        print(f"Target Selected: {os.path.basename(best_target_path)} (Distance: {chosen_dist:.4f})")
        return best_target_path

    # --- Tanh Space Transformations  ---
    def to_tanh_space(self, x, eps=1e-6):
        """Maps an image from [0, 1] to unbounded tanh space."""
        x = torch.clamp(x, eps, 1.0 - eps)
        x = (x - 0.5) * 2.0
        return torch.atanh(x)

    def from_tanh_space(self, w):
        """Maps unbounded tanh space back to [0, 1] image space."""
        return (torch.tanh(w) + 1.0) / 2.0

    def generate_cloak(self, input_path, manual_target_path=None, iterations=1000, lr=0.5, rho=0.007, penalty_lambda=15000.0):
        """
        Step 2: Computing Per-image Cloaks (Full Resolution + Paper Specs)
        """
        # A. Load Original Image (Full Resolution)
        original_tensor = self.load_image_as_tensor(input_path) 
        
        # B. Extract User Features (Dynamically Resized inside get_features)
        with torch.no_grad():
            user_features = self.get_features(original_tensor)

        # C. Determine Target Features
        target_path = manual_target_path
        if not target_path:
            target_path = self.select_optimal_target(user_features)
            
        if target_path:
            target_tensor = self.load_image_as_tensor(target_path)
            with torch.no_grad():
                target_features = self.get_features(target_tensor)
        else:
            target_features = torch.randn_like(user_features)

        # D. Initialize Delta in Tanh Space 
        # Convert original image to tanh space to ensure pixel bounds
        w_orig = self.to_tanh_space(original_tensor).detach()
        
        # We optimize delta in tanh space, at full original resolution!
        delta_w = torch.zeros_like(w_orig, requires_grad=True).to(self.device)
        
        # E. Setup Optimizer (Paper uses lr=0.5 for Adam) [cite: 315]
        optimizer = optim.Adam([delta_w], lr=lr) 

        print(f"Starting Cloaking Process ({iterations} iters, DSSIM budget {rho})...")
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 1. Apply delta and map back to [0, 1] image space 
            cloaked_input = self.from_tanh_space(w_orig + delta_w)
            
            # 2. Extract Features 
            # (Interpolates down to 224x224 smoothly so gradients flow back to full-res delta)
            current_features = self.get_features(cloaked_input)
            
            # 3. Feature Loss: Minimize L2 distance to Target features 
            feature_loss = torch.dist(current_features, target_features, p=2)
            
            # 4. DSSIM Penalty Method 
            current_ssim = ssim(cloaked_input, original_tensor, data_range=1.0, size_average=True)
            current_dssim = (1.0 - current_ssim) / 2.0
            
            # Penalty activates only if DSSIM exceeds our budget rho 
            dssim_penalty = torch.clamp(current_dssim - rho, min=0.0)
            
            # Total Loss combining feature displacement and visual fidelity constraint 
            loss = feature_loss + (penalty_lambda * dssim_penalty)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Step {i}/{iterations} | Loss: {loss.item():.4f} | DSSIM: {current_dssim.item():.5f}")

        # F. Generate Final High-Res Image
        with torch.no_grad():
            final_tensor = self.from_tanh_space(w_orig + delta_w)
            final_tensor = torch.clamp(final_tensor.squeeze(0), 0, 1)
            
            final_image = final_tensor.permute(1, 2, 0).cpu().numpy()
            final_image = (final_image * 255).astype(np.uint8)
            final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            
            return final_bgr
