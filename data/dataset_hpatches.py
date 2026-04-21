import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HPatchesDataset(Dataset):
    """
    HPatches dataset loader for evaluating dense keypoint matching.
    """
    def __init__(
        self, 
        root_dir='/data/cat/ws/tosa098h-abtin_NormMatchTrans/data/downloaded/hpatches/hpatches-sequences-release', 
        step_size=8, 
        transform=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.step_size = step_size
        self.transform = transform
        
        self.pairs = []
        
        # Find all sequence folders
        seq_folders = sorted(glob.glob(os.path.join(self.root_dir, '*')))
        
        for folder in seq_folders:
            seq_name = os.path.basename(folder)
            
            # Filter only viewpoint (v_) and illumination (i_) folders
            if not (seq_name.startswith('v_') or seq_name.startswith('i_')):
                continue
                
            seq_type = 'viewpoint' if seq_name.startswith('v_') else 'illumination'
            
            # The reference image is always 1.ppm
            img1_path = os.path.join(folder, '1.ppm')
            
            # Target images are 2.ppm through 6.ppm
            for tgt_idx in range(2, 7):
                img2_path = os.path.join(folder, f'{tgt_idx}.ppm')
                h_path = os.path.join(folder, f'H_1_{tgt_idx}')
                
                # Check if required files exist before appending
                if os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(h_path):
                    self.pairs.append({
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'h_path': h_path,
                        'seq_type': seq_type,
                        'seq_name': seq_name
                    })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load reference and target images
        img1 = Image.open(pair['img1_path']).convert('RGB')
        img2 = Image.open(pair['img2_path']).convert('RGB')
        
        # Get image dimensions to generate grids and filter points
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        if self.transform is not None:
            img1_tensor = self.transform(img1)
            img2_tensor = self.transform(img2)
        else:
            img1_tensor = img1
            img2_tensor = img2
            
        # Load 3x3 Homography matrix mapping Image 1 -> Image 2
        H = np.loadtxt(pair['h_path'])
        
        # Generate dense pixel coordinates for Image 1
        y_coords, x_coords = np.mgrid[0:h1:self.step_size, 0:w1:self.step_size]
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        
        # Convert to homogeneous coordinates: [x, y, 1]^T
        num_points = len(x_coords)
        coords1_homo = np.ones((3, num_points), dtype=np.float32)
        coords1_homo[0, :] = x_coords
        coords1_homo[1, :] = y_coords
        
        # Project points: [x', y', z']^T = H * [x, y, 1]^T
        coords2_homo = H @ coords1_homo
        
        # Convert back to Cartesian via division by Z
        z_eps = 1e-8
        x2_projected = coords2_homo[0, :] / (coords2_homo[2, :] + z_eps)
        y2_projected = coords2_homo[1, :] / (coords2_homo[2, :] + z_eps)
        
        # Filter points that fall outside the boundaries of Image 2
        valid_mask = (
            (x2_projected >= 0) & (x2_projected < w2) &
            (y2_projected >= 0) & (y2_projected < h2)
        )
        
        # Gather valid coordinates
        valid_x1 = x_coords[valid_mask]
        valid_y1 = y_coords[valid_mask]
        valid_x2 = x2_projected[valid_mask]
        valid_y2 = y2_projected[valid_mask]
        
        # Return tensors for valid matches
        valid_coords1 = torch.tensor(np.stack([valid_x1, valid_y1], axis=1), dtype=torch.float32)
        valid_coords2 = torch.tensor(np.stack([valid_x2, valid_y2], axis=1), dtype=torch.float32)
        
        return {
            'image1': img1_tensor,
            'image2': img2_tensor,
            'coords1': valid_coords1,
            'coords2': valid_coords2,
            'seq_type': pair['seq_type'],       # 'viewpoint' or 'illumination'
            'seq_name': pair['seq_name'],       # e.g., 'v_abs'
            'image1_size': (w1, h1),
            'image2_size': (w2, h2)
        }
