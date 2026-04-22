import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch_geometric.data import Data, Batch

# Adjust path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_hpatches import HPatchesDataset
from model import NMT
from utils.build_graphs import build_graphs
from utils.config import cfg
from utils.utils import update_params_from_cmdline
import sys

# Load the config properly with its defaults resolving
config_path = '/home/tosa098h/infera-abtin/NormMatchTrans/experiments/voc_synthetic.json'
cfg = update_params_from_cmdline(cmd_line=['', config_path], default_params=cfg, verbose=False)

from eval import sinkhorn_logspace

def load_nmt_model(weights_path, device='cuda'):
    """
    Initializes the NMT model and loads the trained weights.
    """
    print(f"Loading NMT model from {weights_path}...")
    model = NMT()
    
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Check if the model was trained with DataParallel/DistributedDataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model

def prepare_hpatches_sample(sample, device='cuda', image_size=384):
    """
    Takes a raw sample from HPatchesDataset, resizes images, scales points,
    normalizes data, and builds the graphs required by NMT.
    """
    img1 = sample['image1']
    img2 = sample['image2']
    coords1 = sample['coords1'] # tensor N x 2
    coords2 = sample['coords2'] # tensor N x 2
    
    # 1. Resize images to 384x384
    img1_res = img1.resize((image_size, image_size), Image.BICUBIC)
    img2_res = img2.resize((image_size, image_size), Image.BICUBIC)
    
    # 2. Normalize images using PyTorch transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORM_MEANS, std=cfg.NORM_STD)
    ])
    img1_tensor = transform(img1_res).unsqueeze(0).to(device)
    img2_tensor = transform(img2_res).unsqueeze(0).to(device)
    images = [img1_tensor, img2_tensor]
    
    # 3. Scale the coordinates
    w1, h1 = sample['image1_size']
    w2, h2 = sample['image2_size']
    
    scale1 = torch.tensor([image_size / w1, image_size / h1], dtype=torch.float32)
    scale2 = torch.tensor([image_size / w2, image_size / h2], dtype=torch.float32)
    
    scaled_coords1 = coords1 * scale1
    scaled_coords2 = coords2 * scale2
    
    n_points1 = scaled_coords1.shape[0]
    n_points2 = scaled_coords2.shape[0]
    
    # If there are no valid points (rare but possible), return None
    if n_points1 == 0:
        return None
        
    # 4. Build Graphs for Image 1 and Image 2
    edge_idx1, edge_feat1 = build_graphs(scaled_coords1.numpy(), n_points1)
    edge_idx2, edge_feat2 = build_graphs(scaled_coords2.numpy(), n_points2)
    
    # Node features (pos) are points divided by image_size
    pos1 = scaled_coords1 / float(image_size)
    pos2 = scaled_coords2 / float(image_size)
    
    graph1 = Data(
        x=pos1, pos=pos1, 
        edge_index=torch.tensor(edge_idx1, dtype=torch.long), 
        edge_attr=torch.tensor(edge_feat1, dtype=torch.float32)
    )
    graph2 = Data(
        x=pos2, pos=pos2, 
        edge_index=torch.tensor(edge_idx2, dtype=torch.long), 
        edge_attr=torch.tensor(edge_feat2, dtype=torch.float32)
    )
    graph1.num_nodes = n_points1
    graph2.num_nodes = n_points2
    
    graphs = [Batch.from_data_list([graph1]).to(device), Batch.from_data_list([graph2]).to(device)]
    
    # 5. Format remaining inputs for the model
    points = [scaled_coords1.unsqueeze(0).to(device), scaled_coords2.unsqueeze(0).to(device)]
    n_points = [torch.tensor([n_points1]).to(device), torch.tensor([n_points2]).to(device)]
    n_points_sample = n_points[0]
    
    # Ground-truth permutation matrix is the identity because of our dataloader's 1-to-1 strict mapping
    gt_perm_mat = torch.eye(n_points1).unsqueeze(0).to(device)
    perm_mats = [gt_perm_mat]
    
    return {
        'images': images,
        'points': points,
        'graphs': graphs,
        'n_points': n_points,
        'n_points_sample': n_points_sample,
        'perm_mats': perm_mats,
        'seq_type': sample['seq_type'],
        'seq_name': sample['seq_name'],
        'scaled_coords1': scaled_coords1,
        'scaled_coords2': scaled_coords2
    }

def run_hpatches_inference(weights_path, step_size=32, device='cuda', max_samples=None):
    # 1. Load the model
    model = load_nmt_model(weights_path, device)
    
    # 2. Init dataset
    print(f"Initializing HPatchesDataset with step_size={step_size}...")
    dataset = HPatchesDataset(step_size=step_size)
    print(f"Total HPatches pairs: {len(dataset)}")
    
    limit = max_samples if max_samples is not None else len(dataset)
    
    results = [] # store results for Step 3 calculation later
    
    with torch.no_grad():
        for idx in tqdm(range(limit), desc="Evaluating HPatches"):
            sample = dataset[idx]
            
            inputs = prepare_hpatches_sample(sample, device=device)
            if inputs is None:
                continue
                
            # 3. Forward Pass
            # NMT signature returns: similarity_scores, hs_dec_output, ht_dec_output, layer_loss
            similarity_scores, _, _, _ = model(
                images=inputs['images'],
                points=inputs['points'],
                graphs=inputs['graphs'],
                n_points=inputs['n_points'],
                n_points_sample=inputs['n_points_sample'],
                perm_mats=inputs['perm_mats'],
                eval_pred_points=0,
                in_training=False
            )
            
            # 4. Sinkhorn Assignment
            sinkhorn = sinkhorn_logspace(similarity_scores)
            
            # For each point in Image 1, we find the index of the predicted matched point in Image 2
            predictions = torch.argmax(sinkhorn, dim=-1).squeeze(0) # [N1]
            
            # Store data for Step 3 metrics
            results.append({
                'seq_type': inputs['seq_type'],
                'seq_name': inputs['seq_name'],
                'pred_indices': predictions.cpu(),
                'scaled_coords1': inputs['scaled_coords1'],
                'scaled_coords2': inputs['scaled_coords2'],
                'n_points': inputs['n_points'][0].item()
            })
            
    print(f"Finished inference on {len(results)} valid pairs.")
    return results

if __name__ == "__main__":
    from utils.mma_metrics import compute_mma
    
    # Execution
    # Update to point dynamically to however you want to fetch weights, or keep it pointing to the synthetic model explicitly
    # spair_weights = '/home/tosa098h/infera-abtin/NormMatchTrans/results/voc_synthetic/nmt_hardest/params/0008/params.pt'
    spair_weights = '/data/cat/ws/tosa098h-abtin_NormMatchTrans/saved_models/results/voc_synthetic/nmt_hardest/params/0005/params.pt'
    
    # Provide the option to run on the full dataset or a subset
    print("Starting full HPatches evaluation (this may take a while)...")
    
    # Use a solid step size for testing, or scale down if you want denser points
    # e.g., step_size=8 is standard, but keeping 32 for quicker testing if needed
    results = run_hpatches_inference(spair_weights, step_size=32, max_samples=None)
    
    if len(results) > 0:
        # Calculate & Output Mean Matching Accuracy (MMA)
        compute_mma(results, thresholds=[1.0, 3.0, 5.0])
    else:
        print("No valid pairs found.")

