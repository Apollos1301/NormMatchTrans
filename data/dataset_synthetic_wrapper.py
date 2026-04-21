import torch
import numpy as np
from torchvision import transforms
from torch_geometric.data import Data, Batch

from data.dataset_synthetic import SyntheticHomographyDataset
from utils.build_graphs import build_graphs
from utils.config import cfg

class SyntheticCollateWrapper:
    """
    Step 2: Dataloader Wrapper wrapping SyntheticHomographyDataset into NMT format.
    Constructs Delaunay graphs, standardizes tensors, and packages data dictionaries
    identical to how `data_loader_multigraph.py` behaves.
    """
    def __init__(self, step_size=32, image_size=384, max_points=1024, train=True):
        self.dataset = SyntheticHomographyDataset(
            step_size=step_size, 
            image_size=image_size, 
            max_points=max_points, 
            train=train
        )
        self.classes = ['synthetic_warping']
        self.cls = 'synthetic_warping'
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORM_MEANS, std=cfg.NORM_STD)
        ])

    def set_num_graphs(self, num_graphs_in_matching_instance):
        pass # Dummy method to avoid crash in train_eval where this function is called

    def set_cls(self, cls):
        self.cls = cls

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We fetch the raw NumPy dict from step 1
        sample = self.dataset[idx]
        return sample

def synthetic_collate_fn(data_list):
    """
    Collates a list of dictionaries emitted by SyntheticHomographyDataset
    and structures them into the exact batch dictionary NMT expects.
    """
    batch_images = []
    batch_Ps = []
    batch_ns = []
    batch_edges = []
    batch_gt_perm_mats = []
    
    # 1. We process each item in the batch sequentially
    for item in data_list:
        img1 = item['image1']
        img2 = item['image2']
        coords1 = item['coords1']  # (N, 2)
        coords2 = item['coords2']  # (N, 2)
        
        N = coords1.shape[0]
        
        # 2. Convert and transform images
        # The NMT dataloader expects a list of 2 tensors per item
        t_img1 = transforms.ToTensor()(img1)
        t_img2 = transforms.ToTensor()(img2)
        
        t_img1 = transforms.Normalize(mean=cfg.NORM_MEANS, std=cfg.NORM_STD)(t_img1)
        t_img2 = transforms.Normalize(mean=cfg.NORM_MEANS, std=cfg.NORM_STD)(t_img2)
        
        # 3. Create Points and N Counts
        p1 = torch.tensor(coords1, dtype=torch.float32)
        p2 = torch.tensor(coords2, dtype=torch.float32)
        
        # 4. Construct Delaunay Graphs
        edge_idx1, edge_feat1 = build_graphs(coords1, N)
        edge_idx2, edge_feat2 = build_graphs(coords2, N)
        
        # Positional nodes (divided by image size to map 0 to 1)
        pos1 = p1 / 384.0
        pos2 = p2 / 384.0
        
        graph1 = Data(
            x=pos1, pos=pos1,
            edge_index=torch.tensor(edge_idx1, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat1, dtype=torch.float32)
        )
        graph1.num_nodes = N
        
        graph2 = Data(
            x=pos2, pos=pos2,
            edge_index=torch.tensor(edge_idx2, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat2, dtype=torch.float32)
        )
        graph2.num_nodes = N
        
        # 5. Ground Truth Permutation Matrix (Strict 1-to-1 diagonal matching)
        perm_mat = torch.eye(N, dtype=torch.float32)
        
        # Append structures for the whole batch
        batch_images.append([t_img1, t_img2])
        batch_Ps.append([p1, p2])
        batch_ns.append(torch.tensor([N, N]))
        batch_edges.append([graph1, graph2])
        batch_gt_perm_mats.append(perm_mat)

    # 6. Finally, convert lists to Batch objects and Padded Tensors
    # Group images
    img1_stack = torch.stack([x[0] for x in batch_images], dim=0) # [B, 3, 384, 384]
    img2_stack = torch.stack([x[1] for x in batch_images], dim=0)
    
    # Pad Point sets because N varies depending on what warped off-screen
    max_N = max([p[0].shape[0] for p in batch_Ps])
    
    padded_P1 = []
    padded_P2 = []
    padded_perms = []
    
    for i in range(len(data_list)):
        p1, p2 = batch_Ps[i]
        n_curr = p1.shape[0]
        
        pad_size = max_N - n_curr
        
        pad_P1 = torch.nn.functional.pad(p1, (0, 0, 0, pad_size), "constant", 0)
        pad_P2 = torch.nn.functional.pad(p2, (0, 0, 0, pad_size), "constant", 0)
        
        padded_P1.append(pad_P1)
        padded_P2.append(pad_P2)
        
        # Pad permutation matrices with zeroes for padding points
        perm = batch_gt_perm_mats[i]
        pad_perm = torch.nn.functional.pad(perm, (0, pad_size, 0, pad_size), "constant", 0)
        padded_perms.append(pad_perm)
        

    # Create PyG Batches
    graphs1 = Batch.from_data_list([e[0] for e in batch_edges])
    graphs2 = Batch.from_data_list([e[1] for e in batch_edges])
    
    # NMT expects: {"images": [B1, B2], "Ps": [P1, P2], "ns": ns, "edges": [G1, G2], "gt_perm_mat": [Perm]}
    ret_dict = {
        "images": [img1_stack, img2_stack],
        "Ps": [torch.stack(padded_P1, dim=0), torch.stack(padded_P2, dim=0)],
        "ns": [torch.stack([x[0] for x in batch_ns]), torch.stack([x[1] for x in batch_ns])],
        "edges": [graphs1, graphs2],
        "gt_perm_mat": [torch.stack(padded_perms, dim=0)]
    }
    
    return ret_dict
