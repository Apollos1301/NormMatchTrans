import os
import glob
import random
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from utils.config import cfg

class SyntheticHomographyDataset(Dataset):
    """
    Step 1: Synthetic Homography dataloader using PascalVOC.
    Generates an image pair (original and warped) via a random homography.
    It computes dense ground-truth coordinate matches between the two views.
    """
    def __init__(self, step_size=8, image_size=384, max_points=1024, train=True):
        super().__init__()
        self.step_size = step_size
        self.image_size = image_size
        self.max_points = max_points
        
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        
        # We fetch all the PascalVOC images to use purely as textures/backgrounds.
        voc_img_dir = os.path.join(cfg.VOC2011.ROOT_DIR, "JPEGImages")
        self.image_paths = sorted(glob.glob(os.path.join(voc_img_dir, "*.jpg")))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {voc_img_dir}")
            
    def __len__(self):
        return len(self.image_paths)

    def generate_random_homography(self):
        """
        Creates a random homography that includes translation, rotation, 
        scaling, and perspective shifts.
        """
        # Base corners of the image
        pts1 = np.float32([
            [0, 0], 
            [self.image_size, 0], 
            [0, self.image_size], 
            [self.image_size, self.image_size]
        ])

        # Randomize the target corners to create a strong perspective warp
        # Increase the boundary perturbation to up to 40% to mimic HPatches extreme viewpoints
        difficulty = random.uniform(0.25, 0.70)
        margin = int(self.image_size * difficulty)
        
        # Add a global random translation to mimic large panning movements
        tx = random.randint(-margin, margin)
        ty = random.randint(-margin, margin)
        
        pts2 = np.float32([
            [0 + tx + random.randint(-margin, margin), 0 + ty + random.randint(-margin, margin)],
            [self.image_size + tx + random.randint(-margin, margin), 0 + ty + random.randint(-margin, margin)],
            [0 + tx + random.randint(-margin, margin), self.image_size + ty + random.randint(-margin, margin)],
            [self.image_size + tx + random.randint(-margin, margin), self.image_size + ty + random.randint(-margin, margin)]
        ])
        
        # Calculate Homography matrix Mapping Image 1 -> Image 2
        H = cv2.getPerspectiveTransform(pts1, pts2)
        return H

    def __getitem__(self, idx):
        # 1. Load an image and resize it exactly to the NMT resolution (384x384)
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')
        
        # We will load a larger image to avoid reflection artifacts at the borders
        pad = int(self.image_size * 0.25)
        large_size = self.image_size + 2 * pad
        img_pil = img_pil.resize((large_size, large_size), Image.BICUBIC)
        img_large = np.array(img_pil)
        
        # Extract the center crop as the baseline image
        img1 = img_large[pad:pad+self.image_size, pad:pad+self.image_size]
        
        while True:
            # 2. Generate the Random Homography Matrix (mapping img1 to img2)
            H = self.generate_random_homography()
            
            # Generate the dense pixel coordinate grid for Image 1
            y_coords, x_coords = np.mgrid[0:self.image_size:self.step_size, 0:self.image_size:self.step_size]
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            
            # Convert to homogeneous coordinates: [x, y, 1]^T
            num_points = len(x_coords)
            coords1_homo = np.ones((3, num_points), dtype=np.float32)
            coords1_homo[0, :] = x_coords
            coords1_homo[1, :] = y_coords
            
            # Project Image 1 coordinates mathematically into Image 2
            coords2_homo = H @ coords1_homo
            z_eps = 1e-8
            x2_projected = coords2_homo[0, :] / (coords2_homo[2, :] + z_eps)
            y2_projected = coords2_homo[1, :] / (coords2_homo[2, :] + z_eps)
            
            # Filter points that naturally fell outside the boundaries of Image 2 during the warp
            valid_mask = (
                (x2_projected >= 0) & (x2_projected <= self.image_size - 1) &
                (y2_projected >= 0) & (y2_projected <= self.image_size - 1)
            )
            
            # Ensure we have a minimum number of valid matching points before proceeding
            # Otherwise, the collate graph size might drop to 0 and crash DDP tensor concatenations.
            if np.sum(valid_mask) >= 32:
                break
        
        # 3. Warp the large image using a translated homography, then crop to avoid border reflections
        T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float32)
        H_large = T @ H @ np.linalg.inv(T)
        
        img2_large = cv2.warpPerspective(img_large, H_large, (large_size, large_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        img2 = img2_large[pad:pad+self.image_size, pad:pad+self.image_size]
        
        # Apply independent photometric augmentations
        img1_pil = Image.fromarray(img1)
        img2_pil = Image.fromarray(img2)
        
        img1_pil = self.color_jitter(img1_pil)
        img2_pil = self.color_jitter(img2_pil)
        
        # Optional random Gaussian Blur for HPatches-like difficulty
        if random.random() < 0.5:
            img1_pil = img1_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))
        if random.random() < 0.5:
            img2_pil = img2_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)
        
        # 4. Generate the dense pixel coordinate grid for Image 1 (Already done in while loop for validation)
        
        valid_x1, valid_y1 = x_coords[valid_mask], y_coords[valid_mask]
        valid_x2, valid_y2 = x2_projected[valid_mask], y2_projected[valid_mask]
        
        # Limit to max_points and add variance to train robustness across different graph sizes (like HPatches 400-1000)
        lower_bound = min(400, self.max_points)
        target_points = random.randint(lower_bound, self.max_points)
        if len(valid_x1) > target_points:
            indices = np.random.choice(len(valid_x1), target_points, replace=False)
            valid_x1, valid_y1 = valid_x1[indices], valid_y1[indices]
            valid_x2, valid_y2 = valid_x2[indices], valid_y2[indices]
            
        valid_coords1 = np.stack([valid_x1, valid_y1], axis=1)
        valid_coords2 = np.stack([valid_x2, valid_y2], axis=1)
        
        # Return cleanly aligned data (NumPy formats) ready for the next step 
        return {
            'image1': img1,
            'image2': img2,
            'coords1': valid_coords1,
            'coords2': valid_coords2
        }
