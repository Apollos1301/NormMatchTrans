import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from data.willow_obj import WillowObject
from data.SPair71k import SPair71k
from utils.build_graphs import build_graphs
import albumentations as A
import os
from PIL import Image


from utils.config import cfg
from torch_geometric.data import Data, Batch

datasets = {"PascalVOC": PascalVOC,
            "WillowObject": WillowObject,
            "SPair71k": SPair71k}

class GMDataset(Dataset):
    def __init__(self, train_or_test, name, length, **args):
        self.added_length = 0
        self.name = name
        self.train = True if train_or_test == "train" else False
        self.ds = datasets[name](**args)
        self.true_epochs = length is None
        print("TRUE EPOCHS: ", self.true_epochs)
        print(length)
        self.length = (
            self.ds.total_size if self.true_epochs else length
        )  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        # self.length = length
        if self.true_epochs:
            print(f"Initializing {self.ds.sets}-set with all {self.length} examples.")
        else:
            print(f"Initializing {self.ds.sets}-set. Randomly sampling {self.length} examples.")
        # length here represents the iterations between two checkpoints
        # if length is None the length is set to the size of the ds
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None
        #TODO: Hard-coded to 2 graphs  
        self.num_graphs_in_matching_instance = 2
        self.aug_erasing = A.Compose([A.CoarseDropout(num_holes_range=(1, 3),
                                        hole_height_range=(20, 40),
                                        hole_width_range=(20, 40),
                                        p=cfg.TRAIN.random_erasing_prob)],
                                     keypoint_params=A.KeypointParams(format="xy", remove_invisible=True, label_fields=['class_labels']))
        self.aug_pipeline = A.Compose([#A.HueSaturationValue(p=0.5),
                                       #A.RandomGamma(p=0.5),
                                       #A.RGBShift(p=0.5),
                                       #A.CLAHE(p=0.5),
                                       #A.Blur(p=0.5),
                                       A.RandomBrightnessContrast(p=0.1)],
                                      keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
        self.added_data = []
        self.folder_path = './data/downloaded/PascalVOC/VOC2011/JPEGImages'
        self.filenames = os.listdir(self.folder_path)
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)

    def set_cls(self, cls):
        if cls == "none":
            cls = None
        self.cls = cls
        if self.true_epochs:  # Update length of dataset for dataloader according to class
            self.length = self.ds.total_size if cls is None else self.ds.size_by_cls[cls]

    def set_num_graphs(self, num_graphs_in_matching_instance):
        self.num_graphs_in_matching_instance = num_graphs_in_matching_instance
        
    def inject_new_data(self, new_data):
        self.added_data.append[new_data]
        self.added_length += 1

    def __len__(self):
        return self.length + self.added_length

    def __getitem__(self, idx):
        random_mixUP_img_idx = random.randint(0, len(self.filenames)-1)
        random_mixUP_img_path = os.path.join(self.folder_path, self.filenames[random_mixUP_img_idx])
        with Image.open(str(random_mixUP_img_path)) as img:
            random_mixUP_img = img.resize(self.obj_size, resample=Image.BICUBIC)
        random_mixUP_img = np.array(random_mixUP_img)
        
        
        random_cutMix_img_idx = random.randint(0, len(self.filenames)-1)
        random_cutMix_img_path = os.path.join(self.folder_path, self.filenames[random_cutMix_img_idx])
        with Image.open(str(random_cutMix_img_path)) as img:
            random_cutMix_img = img.resize(self.obj_size, resample=Image.BICUBIC)
        random_cutMix_img = np.array(random_cutMix_img)
            
        sampling_strategy = cfg.train_sampling if self.ds.sets == "train" else cfg.eval_sampling
        if self.num_graphs_in_matching_instance is None:
            raise ValueError("Num_graphs has to be set to an integer value.")

        idx = idx if self.true_epochs else None
        anno_list, perm_mat_list = self.ds.get_k_samples(idx, k=self.num_graphs_in_matching_instance, cls=self.cls, mode=sampling_strategy)
        # print(anno_list)
        # print(perm_mat_list)
        
        """
        Implement Random Swap here
        """
        for perm_mat in perm_mat_list:
            if (
                not perm_mat.size
                or (perm_mat.size < 2 * 2 and sampling_strategy == "intersection")
                and not self.true_epochs
            ):
                # 'and not self.true_epochs' because we assume all data is valid when sampling a true epoch
                next_idx = None if idx is None else idx + 1
                return self.__getitem__(next_idx)

        points_gt = [np.array([(kp["x"], kp["y"]) for kp in anno_dict["keypoints"]]) for anno_dict in anno_list]
        n_points_gt = [len(p_gt) for p_gt in points_gt]
        # print(points_gt)
        # annotation_dict = [np.array(anno_dict["keypoints"]) for anno_dict in anno_list]
        turn_off = False
        imgs = [anno["image"] for anno in anno_list]
        if self.train:
            transformed_class_labels1 = []
            transformed_class_labels2 = []
            h, w = imgs[0].size[:2]
            points_gt[0] = np.clip(points_gt[0], [0, 0], [w-1, h-1])
            points_gt[1] = np.clip(points_gt[1], [0, 0], [w-1, h-1])
            kp_labels1 = np.arange(n_points_gt[0])
            kp_labels2 = np.arange(n_points_gt[1])
            
            augmented = self.aug_erasing(image=np.array(imgs[0]), keypoints=points_gt[0], class_labels=kp_labels1)
            transformed_class_labels1 = augmented['class_labels']
            
            
            removed_kp_rows = np.setdiff1d(kp_labels1, transformed_class_labels1)
            if len(removed_kp_rows) < len(points_gt[0]):
                imgs[0] = augmented["image"]
                # # removed_kp_columns = np.setdiff1d(kp_labels2, transformed_class_labels2)
                sub_matrix1 = perm_mat_list[0][removed_kp_rows, :]
                # # sub_matrix2 = perm_mat_list[0][:, removed_kp_columns]
                _, column_indices = np.where(sub_matrix1 == 1)
                # # row_indices, _ = np.where(sub_matrix2 == 1)
                
                # perm_mat_list[0][removed_kp_rows, :] = 0
                # # # perm_mat_list[0][:, removed_kp_columns] = 0
                # has_one = np.any(perm_mat_list[0] == 1, axis=1)
                # perm_mat_list[0] = perm_mat_list[0][has_one]#np.vstack([perm_mat_list[0][has_one], perm_mat_list[0][~has_one]])
                perm_mat_list[0] = np.delete(perm_mat_list[0], removed_kp_rows, axis=0)  # Delete rows
                perm_mat_list[0] = np.delete(perm_mat_list[0], column_indices, axis=1)   # Delete columns
                
                
                points_gt[0] = np.delete(points_gt[0], removed_kp_rows, axis=0)
                points_gt[1] = np.delete(points_gt[1], column_indices, axis=0)
            
            
            img_cutMix, cutMix_rm_kp = cutmix_with_keypoints_indices(imgs[0], random_cutMix_img, points_gt[0], cutmix_prob=cfg.TRAIN.cutmix_prob, beta=cfg.TRAIN.cutmix_beta)
            
            if len(cutMix_rm_kp) < len(points_gt[0]):
                imgs[0] = img_cutMix
                sub_matrix1 = perm_mat_list[0][cutMix_rm_kp, :]
                _, column_indices = np.where(sub_matrix1 == 1)
                perm_mat_list[0] = np.delete(perm_mat_list[0], cutMix_rm_kp, axis=0)  # Delete rows
                perm_mat_list[0] = np.delete(perm_mat_list[0], column_indices, axis=1)   # Delete columns
                
                points_gt[0] = np.delete(points_gt[0], cutMix_rm_kp, axis=0)
                points_gt[1] = np.delete(points_gt[1], column_indices, axis=0)
            
            
            
            n_points_gt = [len(p_gt) for p_gt in points_gt]
            
            #MixUp
            alpha = cfg.TRAIN.mixup_alpha
            mixup_prob = cfg.TRAIN.mixup_prob
            if np.random.rand() < mixup_prob:
                lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
                imgs[0] = lam*imgs[0] + (1.0 - lam)*random_mixUP_img
                imgs[0] = imgs[0].astype(np.float32)
            if np.random.rand() < mixup_prob:
                lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
                imgs[1] = lam*imgs[1] + (1.0 - lam)*random_mixUP_img
                imgs[1] = imgs[1].astype(np.float32)
        
        graph_list = []
        for p_gt, n_p_gt in zip(points_gt, n_points_gt):
            edge_indices, edge_features = build_graphs(p_gt, n_p_gt)
            #print(self.obj_size)
            # Add dummy node features so the __slices__ of them is saved when creating a batch
            pos = torch.tensor(p_gt).to(torch.float32) / self.obj_size[0]
            assert (pos > -1e-5).all(), p_gt
            graph = Data(
                edge_attr=torch.tensor(edge_features).to(torch.float32),
                edge_index=torch.tensor(edge_indices, dtype=torch.long),
                x=pos,
                pos=pos,
            )
            graph.num_nodes = n_p_gt
            graph_list.append(graph)

        # current_class = anno_list[0]["cls"]
        
        ret_dict = {
            "Ps": [torch.Tensor(x) for x in points_gt],
            "ns": [torch.tensor(x) for x in n_points_gt],
            "gt_perm_mat": perm_mat_list,
            "edges": graph_list
        }

        
        if imgs[0] is not None:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
            imgs = [trans(img) for img in imgs]
            ret_dict["images"] = imgs
        elif "feat" in anno_list[0]["keypoints"][0]:
            feat_list = [np.stack([kp["feat"] for kp in anno_dict["keypoints"]], axis=-1) for anno_dict in anno_list]
            ret_dict["features"] = [torch.Tensor(x) for x in feat_list]

        return ret_dict
    
    def inject_error_data(self, ret_dict, error_indices):
        pass

import numpy as np

def cutmix_with_keypoints_indices(image, sampled_image, keypoints, cutmix_prob=0.5, beta=1.0, max_cut_ratio=0.3):
    """
    Applies CutMix augmentation on an input image and returns the indices of keypoints removed.
    The size of the cut region is limited by max_cut_ratio.
    
    Parameters:
        image (np.ndarray): The input image (H x W x C).
        sampled_image (np.ndarray): The second image sampled from the dataset (H x W x C).
        keypoints (np.ndarray): 2D numpy array of keypoint coordinates in [x, y] format (shape: N x 2).
        cutmix_prob (float): The probability of applying CutMix.
        beta (float): Hyperparameter for the Beta distribution to sample lambda.
        max_cut_ratio (float): Maximum fraction of image width/height for the cut region.
        
    Returns:
        new_image (np.ndarray): The augmented image.
        removed_indices (np.ndarray): Indices of keypoints that lie inside the replaced region.
    """
    # If CutMix is not applied, return the original image and an empty array for indices.
    if np.random.rand() > cutmix_prob:
        return image.copy(), np.empty((0,), dtype=int)
    
    if image is None:
        return image.copy(), np.empty((0,), dtype=int)
    
    if not hasattr(image, 'shape'):
        return image.copy(), np.empty((0,), dtype=int)
    # Sample lambda from a Beta distribution.
    lam = np.random.beta(beta, beta)
    
    # Get dimensions of the image
    H, W, _ = image.shape

    # Compute ideal cut ratio based on lambda but then limit it with max_cut_ratio.
    cut_ratio = np.sqrt(1. - lam)
    cut_ratio = np.minimum(cut_ratio, max_cut_ratio)
    
    # Determine the size of the patch to be cut out.
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Randomly choose the center of the patch.
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Compute the coordinates of the patch and ensure they are within image bounds.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Create a copy of the original image.
    new_image = image.copy()
    
    # Replace the selected region in the original image with the corresponding region from the sampled image.
    new_image[bby1:bby2, bbx1:bbx2, :] = sampled_image[bby1:bby2, bbx1:bbx2, :]
    
    # Identify the indices of keypoints that lie inside the replaced patch.
    inside_patch = (
        (keypoints[:, 0] >= bbx1) & (keypoints[:, 0] < bbx2) &
        (keypoints[:, 1] >= bby1) & (keypoints[:, 1] < bby2)
    )
    removed_indices = np.where(inside_patch)[0]
    
    return new_image, removed_indices


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            raise ValueError("Cannot handle type {}".format(type(inp[0])))
        return ret

    ret = stack(data)
    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, data_sampler, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=data_sampler,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
    )
