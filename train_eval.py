import torch
import torch.optim as optim
import torch.nn.functional as F
# import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from pathlib import Path
import os
import pandas as pd
import matplotlib


from sklearn.metrics import f1_score
from data.data_loader_multigraph import GMDataset, get_dataloader
import eval
from matchAR import Net, SimpleNet, EncoderNet, ResMatcherNet, MatchARNet
from utils.config import cfg
from utils.utils import update_params_from_cmdline, compute_grad_norm



def calculate_correct_and_valid(prediction_tensor, y_values_matching):
    valid_mask = (prediction_tensor != -1) & (y_values_matching != -1)
    batch_correct = (prediction_tensor[valid_mask] == y_values_matching[valid_mask]).sum().item()
    batch_total_valid = valid_mask.sum().item()
    return batch_correct, batch_total_valid


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


lr_schedules = {
    #TODO: CHANGE BACK TO 10
    "long_halving": (30, (2, 4, 6, 9, 10, 13, 16, 18, 20, 23, 26, 29), 0.5),
    # "long_halving": (30, (3, 6, 12, 26), 0.25),
    # "long_halving": (50, (40,), 0.1),
    "short_halving": (2, (1,), 0.5),
    "long_nodrop": (10, (10,), 1.0),
    "minirun": (1, (10,), 1.0),
}

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets['train'], datasets['val']

def swap_src_tgt_order(data_list, i):
    # edge features
    if data_list[0].__class__.__name__ == 'DataBatch':
        tmp = data_list[1]
        data_list[1] = data_list[0]
        data_list[0] = tmp
    else:
        tmp = data_list[1][i].clone()
        data_list[1][i] = data_list[0][i]
        data_list[0][i] = tmp
    return data_list

def swap_permutation_matrix(perm_mat_list, i):
    transposed_slice = torch.transpose(perm_mat_list[0][i, :, :], 1, 0)
    output_tensor = perm_mat_list[0].clone()
    output_tensor[i, :, :] = transposed_slice

    return [output_tensor]

def mask_loss(perm_mat_list, sampled_points):
    perm_mat_mask = []
    B, N_s, N_t  = perm_mat_list[0].size()

    for k, i in enumerate(sampled_points):
        matched_ndx = torch.nonzero(perm_mat_list[0][k,:,:] == 1).squeeze()[:i].to(sampled_points.device)
        mask = torch.ones(N_s, N_t).to(sampled_points.device)
        mask[:i,:] = torch.zeros(i, N_t)
        mask[:,matched_ndx[:,1]] = torch.zeros(N_s, matched_ndx[:,1].size(0)).to(sampled_points.device)
        perm_mat_mask.append(mask)
    perm_mat_mask = torch.stack(perm_mat_mask, dim=0).to(sampled_points.device)
    return perm_mat_mask
        

def sample_errors(perm_mats, pred_mats, threshold=0.2):
    """
    Samples error cases where the prediction deviates significantly from the ground truth.
    
    Args:
        perm_mats: Ground truth permutation matrices.
        pred_mats: Predicted permutation matrices.
        threshold: Error threshold to define an incorrect prediction.
    
    Returns:
        error_samples: A list of indices where errors occurred.
    """
    error_samples = []
    for i in range(perm_mats.shape[0]):  # Iterate over the batch
        error = torch.sum(perm_mats[i] != pred_mats[i]) / perm_mats[i].numel()  # Error rate
        if error > threshold:
            error_samples.append(i)
    return error_samples

def create_pred_mask(batch_size, n_points_list, target_length=40):
    """
    Creates a mask for each sample in the batch, setting the first and last elements to False
    and a middle segment of length specified by `n_points_list` to True.

    Parameters:
    - batch_size: Number of samples in the batch.
    - n_points_list: List of integers, where each integer specifies the number of points 
                     to set as True in the middle of the mask for each batch sample.
    - target_length: The length of each sequence.

    Returns:
    - A binary mask of shape [batch_size, target_length] with the specified pattern for each sample.
    """
    # Initialize the mask with all False
    mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
    
    for i in range(batch_size):
        # Define the start and end indices for the True segment for each batch sample
        n_points = n_points_list[i]
        start_idx = n_points
        end_idx = min(start_idx + n_points, target_length)
        
        # Set the middle section of n_points to True for this batch sample
        mask[i, start_idx-1:end_idx-1] = True

    return mask

def calculate_f1_score(prediction_tensor, y_values_matching):
    
    # Mask to filter out invalid predictions/labels
    valid_mask = (prediction_tensor != -1) & (y_values_matching != -1)
    valid_preds = prediction_tensor[valid_mask]
    valid_labels = y_values_matching[valid_mask]
    

    valid_preds = valid_preds.cpu().numpy()
    valid_labels = valid_labels.cpu().numpy()
    
    f1_score_ = f1_score(valid_labels, valid_preds, average='micro')

    return f1_score_


def split_tensor(tensor_1, tensor_2):
    result = []
    start_index = 0

    for length in tensor_2:
        end_index = start_index + length
        result.append(tensor_1[start_index:end_index])
        start_index = end_index

    return result

def train_eval_model(model, criterion, optimizer, dataloader, max_norm, num_epochs, local_rank, resume=False, start_epoch=0):
    since = time.time()
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)
    

    device = next(model.parameters()).device
    if local_rank == 0:
        print("Start training...")
        print("{} model on device: {}".format(cfg.MODEL_ARCH , device))

    checkpoint_path = Path(cfg.model_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path))

        optim_path = os.path.join(cfg.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # Evaluation only
    if cfg.evaluate_only:
        assert resume
        print(f"Evaluating without training...")
        accs, f1_scores, err_dict = eval.eval_model(model, dataloader["test"], eval_epoch=2)
        acc_dict = {
            "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
        }
        f1_dict = {
            "f1_{}".format(cls): single_f1_score
            for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
        }
        acc_dict.update(f1_dict)
        acc_dict["matching_accuracy"] = torch.mean(accs)
        acc_dict["f1_score"] = torch.mean(f1_scores)

        time_elapsed = time.time() - since
        print(
            "Evaluation complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
            )
        )
        err_dist_df = pd.DataFrame(err_dict)
        return model, acc_dict

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_decay
    )
    
    for epoch in range(start_epoch, num_epochs):
        if local_rank == 0:
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
        model.train()  # Set model to training mode

        if local_rank == 0:
            print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss_2 = 0
        epoch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        epoch_acc = 0.0
        running_f1 = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        iter_num = 0

        # print(len(dataloader["train"]))
        # Iterate over data.
        epoch_correct = 0
        epoch_total_valid = 0
        modeL_parameter_list = list(model.parameters())
        # print(modeL_parameter_list[-1:])
        for inputs in dataloader["train"]:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.cuda() for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            # print("**************************************     training     **************************************")
            # # print(data_list)
            # # print("----------------------------------------------------------------")
            # print(len(points_gt_list), points_gt_list[0].size())
            # print(points_gt_list)
            # print("----------------------------------------------------------------")
            # print(n_points_gt_list)
            # print("----------------------------------------------------------------")
            # br
            # print(edges_list)
            # br
            # print("----------------------------------------------------------------")
            # print(perm_mat_list)
            # print("----------------------------------------------------------------")
            
            # # randomly swap source and target images
            if cfg.TRAIN.random_swap:
                for i in range(data_list[0].shape[0]):
                    # with 0.5 probability
                    swap_flag = torch.bernoulli(torch.Tensor([0.5]))
                    swap_flag = int(swap_flag.item())

                    if swap_flag:
                        # swap edge list
                        # swap everything else
                        perm_mat_list = swap_permutation_matrix(perm_mat_list, i)
                        data_list = swap_src_tgt_order(data_list, i)
                        points_gt_list = swap_src_tgt_order(points_gt_list, i)
                        n_points_gt_list = swap_src_tgt_order(n_points_gt_list, i)
                        edges_list = swap_src_tgt_order(edges_list, i)
            n_points_gt_sample = n_points_gt_list[0].to('cpu').apply_(lambda x: torch.randint(low=0, high=x, size=(1,)).item()).to(device)
            
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                
                target_points, model_output = model(data_list, points_gt_list, edges_list, n_points_gt_list, n_points_gt_sample, perm_mat_list)
                
                batch_size = model_output.size()[0]
                num_points1 = model_output.size()[1]
                total_loss = 0
                total_cosine_similarities = []
                #****************************************************************
                for b in range(batch_size):
                    batch_loss = 0
                    batch_cosine_similarities = []
                    for i in range(num_points1):
                        # Compute cosine similarity of model_output[b, i] with all points in target_points[b]
                        cosine_similarities = F.cosine_similarity(model_output[b, i].unsqueeze(0), target_points[b])
                        batch_cosine_similarities.append(cosine_similarities)
                    total_cosine_similarities.append(torch.stack(batch_cosine_similarities))
                total_cosine_similarities = torch.stack(total_cosine_similarities).to(model_output.device)
                similarity_scores = torch.atanh(total_cosine_similarities)
                has_one = perm_mat_list[0].sum(dim=2) != 0
                expanded_mask = has_one.unsqueeze(-1).expand_as(perm_mat_list[0])
                similarity_scores = similarity_scores.masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                y_values = perm_mat_list[0].masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                
                loss = criterion(similarity_scores, y_values)
                
                # loss /= y_values.shape[0]

                # backward + optimize
                loss.backward()
                #****************************************************************
                #print(total_probabs)
                # print(perm_mat_list[0])
                ################################################################	
                # for b in range(batch_size):
                #     batch_loss = 0
                #     for i in range(num_points1):
                #         # Compute cosine similarity of model_output[b, i] with all points in target_points[b]
                #         cosine_similarities = F.cosine_similarity(model_output[b, i].unsqueeze(0), target_points[b])
                #         # print(cosine_similarities)
                #         # Apply target for this specific model_output[b, i] row
                #         losses = torch.where(target_similarity[b, i] == 1, 1 - cosine_similarities, torch.clamp(cosine_similarities, min=0))
                        
                #         # print(losses.shape, losses)
                #         # print(losses)
                #         # Accumulate the mean loss for this model_output[b, i] with all points in target_points[b]
                #         batch_loss += losses.mean()
                #         # print(batch_loss)
                #     # Average loss across all points in model_output for the batch and accumulate
                #     total_loss += batch_loss / num_points1

                # # Average loss across the entire batch
                # loss = total_loss #/ batch_size
                # # print(loss.item())
                # loss.backward()
                ################################################################	
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            with torch.no_grad():
                matchings = []
                B, N_s, N_t = perm_mat_list[0].size()
                # print(perm_mat_list[0].size(), perm_mat_list[0])
                n_points_sample = torch.zeros(B, dtype=torch.int).to(device)
                # perm_mat_dec_list = [torch.zeros(B, N_s, N_t, dtype=torch.int).to(device)]
                # cost_mask = torch.ones(B, N_s, N_t, dtype=torch.int).to(device)
                # batch_idx = torch.arange(cfg.BATCH_SIZE)

                # set matching score for padded to zero
                # for batch in batch_idx:
                #     n_point = n_points_gt_list[0][batch]
                #     cost_mask[batch, n_point:, :] = -1
                #     cost_mask[batch, :, n_point:] = -1 # ?
                eval_pred_points = 0
                j_pred = 0
                predictions_list = []
                for i in range(B):
                    predictions_list.append([])
                for np in range(N_t):
                    
                    target_points, model_output = model(data_list, points_gt_list, edges_list, n_points_gt_list,  perm_mat_list, n_points_sample, eval_pred_points, in_training= False)
                    batch_size = model_output.size()[0]
                    num_points1 = model_output.size()[1]
                    for b in range(batch_size):
                        cosine_similarities = F.cosine_similarity(model_output[b, eval_pred_points].unsqueeze(0), target_points[b])
                        cosine_similarities = torch.atanh(cosine_similarities)
                        cosine_scores = F.softmax(cosine_similarities, dim=-1)
                        cosine_matchings = torch.argmax(cosine_scores, dim=-1)
                        if eval_pred_points < n_points_gt_list[0][b]:
                            predictions_list[b].append(cosine_matchings.item())
                        else:
                            predictions_list[b].append(-1)
                    
                    eval_pred_points +=1
                prediction_tensor = torch.tensor(predictions_list).to(perm_mat_list[0].device)
                y_values_matching = torch.argmax(perm_mat_list[0], dim=-1)
                batch_correct, batch_total_valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                f1_score_ = calculate_f1_score(prediction_tensor, y_values_matching)
                # print(prediction_tensor)
                # print(y_values_matching)
                # print("------------------")
                # print(batch_correct, batch_total_valid)
                epoch_correct += batch_correct
                epoch_total_valid += batch_total_valid
                
                
                
                
                
            bs = perm_mat_list[0].size(0)
            epoch_loss += loss.item() * bs
            epoch_f1 += f1_score_ * bs
            
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0
        
        
        epoch_loss = epoch_loss / dataset_size
        
        epoch_f1 = epoch_f1 / dataset_size
        if local_rank == 0:
            print(f'epoch loss: {epoch_loss}, epoch accuracy: {epoch_acc}, epoch f1_score: {epoch_f1}')
            
            
        scheduler.step()

    # time_elapsed = time.time() - since
    # print(
    #     "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
    #         time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
    #     )
    # )

    return model, acc_dict


if __name__ == "__main__":
    # print('Using config file from: ', os.sys.argv[1])
    cfg = update_params_from_cmdline(default_params=cfg)
    
    #windows
    dist.init_process_group(backend='gloo', init_method='env://')
    
    #linux
    # dist.init_process_group(backend='nccl', init_method='env://')
    
    import json
    import os

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)
    
    
    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="matchAR",
    
    # # track hyperparameters and run metadata
    # config={
    # "learning_rate": cfg.TRAIN.LR,
    # "architecture": cfg.MODEL_ARCH,
    # "dataset": cfg.DATASET_NAME,
    # "epochs": lr_schedules[cfg.TRAIN.lr_schedule][0],
    # "batch_size": cfg.BATCH_SIZE,
    # "cfg_full": cfg
    # }
    # )

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }
    
    sampler = {
    "train": DistributedSampler(image_dataset["train"]),
    "test": DistributedSampler(image_dataset["test"])
    }
    
    dataloader = {x: get_dataloader(image_dataset[x],sampler[x], fix_seed=(x == "test")) for x in ("train", "test")}

    # torch.cuda.set_device(0)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.MODEL_ARCH == 'tf':
        model = Net()
    elif cfg.MODEL_ARCH == 'mlp':
        model = SimpleNet()
    elif cfg.MODEL_ARCH == 'enc':
        model = EncoderNet()
    elif cfg.MODEL_ARCH == 'res':
        model = ResMatcherNet()
    elif cfg.MODEL_ARCH == 'ar':
        model = MatchARNet()
    
    
    local_rank = int(os.environ['LOCAL_RANK']) 
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()

    # print(model)
    backbone_params = list(model.module.node_layers.parameters()) + list(model.module.edge_layers.parameters())
    # backbone_params += list(model.final_layers.parameters())
    
    

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = optim.RAdam(opt_params) #, weight_decay=1e-5
    # optimizer = optim.Adam(opt_params, weight_decay=1e-5)

    if not Path(cfg.model_dir).exists():
        Path(cfg.model_dir).mkdir(parents=True)

    num_epochs, _, __ = lr_schedules[cfg.TRAIN.lr_schedule]
    model, accs = train_eval_model(model, 
                                   criterion, 
                                   optimizer,
                                   dataloader,
                                   cfg.TRAIN.clip_norm, 
                                   num_epochs=num_epochs,
                                   local_rank=local_rank,
                                   resume=cfg.warmstart_path is not None, 
                                   start_epoch=0,
                                   )
    
    dist.destroy_process_group()