import math
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
# import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DistributedSampler
import time
from pathlib import Path
import os
from datetime import timedelta

from data.data_loader_multigraph import GMDataset, get_dataloader
import eval
from model import NMT
from utils.config import cfg
from utils.utils import update_params_from_cmdline
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score

class InfoNCE_Loss(torch.nn.Module):
    def __init__(self, temperature):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = torch.tensor(temperature, dtype=torch.float32)
        #self.temperature = torch.nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
    def forward(self, similarity_tensor, pos_indices, source_Points, target_Points, similarity_tensor_2, pos_indices_2):
        source_sim_numer = torch.bmm(source_Points, source_Points.transpose(1, 2))
        source_sim_normed1 = torch.norm(source_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(2)
        source_sim_normed2 = torch.norm(source_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
        source_sim_denominator = torch.bmm(source_sim_normed1, source_sim_normed2)
        source_cosine_sim_ = source_sim_numer / source_sim_denominator
        
        
        target_sim_numer = torch.bmm(target_Points, target_Points.transpose(1, 2))
        target_sim_normed1 = torch.norm(target_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(2)
        target_sim_normed2 = torch.norm(target_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
        target_sim_denominator = torch.bmm(target_sim_normed1, target_sim_normed2)
        target_cosine_sim_ = target_sim_numer / target_sim_denominator
        
        ident_mat = torch.eye(source_cosine_sim_.shape[1]).to(device)
        source_cosine_sim = source_cosine_sim_ - 2 * ident_mat
        target_cosine_sim = target_cosine_sim_ - 2 * ident_mat

        source_prot_score_max, _ = torch.max(source_cosine_sim, dim=-1)
        source_prot_score_mean = torch.mean(source_prot_score_max, dim=-1)
        source_prot_score_mean = torch.mean(source_prot_score_mean)
        
        target_prot_score_max, _ = torch.max(target_cosine_sim, dim=-1)
        target_prot_score_mean = torch.mean(target_prot_score_max, dim=-1)
        target_prot_score_mean = torch.mean(target_prot_score_mean)
        
        sim_score = similarity_tensor #torch.atanh(similarity_tensor)
        logits = sim_score / self.temperature
        loss_1 = F.cross_entropy(logits, pos_indices)
        
        logits_2 = similarity_tensor_2 / self.temperature
        loss_2 = F.cross_entropy(logits_2, pos_indices_2)
       
        loss = loss_1 + loss_2 #(loss_1 + loss_2) / 2
        return loss + source_prot_score_mean + target_prot_score_mean #  sq_forb_norm 

lr_schedules = {
    #TODO: CHANGE BACK TO 10
    "long_halving1": (32, (3, 8, 13, 20), 0.3),
    "long_halving2": (32, (10, 15, 30), 0.1),
    "long_halving3": (32, (2, 3, 5), 0.1),
    "long_halving4": (32, (2, 4, 6), 0.1),
    "long_halving5": (32, (2, 5, 7), 0.1),
    "short_halving": (2, (1,), 0.5),
    "long_nodrop": (10, (10,), 1.0),
    "minirun": (1, (10,), 1.0),
}

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



def train_eval_model(model, criterion, optimizer, dataloader, max_norm, num_epochs, local_rank, output_rank, world_size, resume=False, start_epoch=0):
    
    
    
    
    since = time.time()
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)
    all_error_dict = {}

    device = next(model.parameters()).device
    if local_rank == output_rank:
        print("Start training...")
        print("NMT model on device: {}".format(device))

    checkpoint_path = Path(cfg.model_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path, map_location=f'cuda:{local_rank}'))

        optim_path = os.path.join(cfg.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path, map_location=f'cuda:{local_rank}'))

    # Evaluation only
    if cfg.evaluate_only:
        # assert resume
        if local_rank == output_rank:
            print(f"Evaluating without training...")
            evaluation_epoch = 5
            accs, f1_scores, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank, eval_epoch=evaluation_epoch)
            all_error_dict[evaluation_epoch] = error_dict
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
        
        return model, all_error_dict

    _, lr_milestones, lr_decay = lr_schedules[cfg.TRAIN.lr_schedule]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_decay
    )
    torch.autograd.set_detect_anomaly(True)
    result_dict = {}
    
    iter_num = 0
    
    for epoch in range(start_epoch, num_epochs):
        if local_rank == output_rank:
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
        model.train()  # Set model to training mode

        if local_rank == output_rank:
            print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss_2 = 0
        epoch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        epoch_acc = 0.0
        running_f1 = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        

        tp = 0
        fp = 0
        fn = 0
        
        epoch_correct = 0
        epoch_total_valid = 0
        
        for inputs in dataloader["train"]:
            # all_classes = [_ for _ in inputs["cls"]]
            # print(all_classes)
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.cuda() for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            
            
            n_points_gt_sample = n_points_gt_list[0] #n_points_gt_list[0].to('cpu').apply_(lambda x: torch.randint(low=1, high=x, size=(1,)).item()).to(device)
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                similarity_scores, s_points, t_points, layer_loss = model(data_list, points_gt_list, edges_list, n_points_gt_list, n_points_gt_sample, perm_mat_list)
                eval_similarity_scores = similarity_scores.clone().detach()
                batch_size = similarity_scores.shape[0]
                
                
                for idx, e in enumerate(n_points_gt_sample):
                    perm_mat_list[0][idx, e:, :] = 0
                
                has_one = perm_mat_list[0].sum(dim=2) != 0
                expanded_mask = has_one.unsqueeze(-1).expand_as(perm_mat_list[0])
                
                similarity_scores_2 = similarity_scores.clone().transpose(-2, -1)
                perm_mat_list_2 = perm_mat_list[0].clone().transpose(-2, -1)
                
                similarity_scores = similarity_scores.masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                y_values = perm_mat_list[0].masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                y_values_ = torch.argmax(y_values, dim=1)
                
                similarity_scores_2 = similarity_scores_2.masked_select(expanded_mask).view(-1, perm_mat_list_2.size(2))
                y_values_2 = perm_mat_list_2.masked_select(expanded_mask).view(-1, perm_mat_list_2.size(2))
                y_values_2 = torch.argmax(y_values_2, dim=1)
                
                loss = criterion(similarity_scores, y_values_, s_points, t_points, similarity_scores_2, y_values_2) #, prototype_score
                loss = loss + layer_loss
                loss.backward()
                
                if max_norm > 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            torch.nn.utils.clip_grad_norm_(param, max_norm)
                        
            
                optimizer.step()

                
                model.module.enforce_constraints()
                
                
            with torch.no_grad():
                matchings = []
                B, N_s, N_t = perm_mat_list[0].size()
                
                
                eval_pred_points = 0
                j_pred = 0
                predictions_list = []
                for i in range(B):
                    predictions_list.append([])
                
                # similarity_scores, _, _, _ = model(data_list, points_gt_list, edges_list, n_points_gt_list,  n_points_gt_sample, perm_mat_list, eval_pred_points=eval_pred_points, in_training= True)
                
                batch_size = eval_similarity_scores.shape[0]
                keypoint_preds = F.softmax(eval_similarity_scores, dim=-1)
                keypoint_preds = torch.argmax(keypoint_preds, dim=-1)
                for np in range(N_t):
                    for b in range(batch_size):
                        if eval_pred_points < n_points_gt_sample[b]:
                            predictions_list[b].append(keypoint_preds[b][eval_pred_points].item())
                        else:
                            predictions_list[b].append(-1)
                    eval_pred_points +=1
                prediction_tensor = torch.tensor(predictions_list).to(perm_mat_list[0].device)
                y_values_matching = torch.argmax(perm_mat_list[0], dim=-1)
                
                error_list = (prediction_tensor != y_values_matching).int()
            
                for idx, e in enumerate(n_points_gt_sample):
                    if e.item() not in result_dict:
                        result_dict[e.item()] = [1, error_list[idx,:e.item()]]
                    result_dict[e.item()][0] += 1
                    result_dict[e.item()][1] += error_list[idx,:e.item()]
                
                
                has_one = perm_mat_list[0].sum(dim=2) != 0
                expanded_mask = has_one.unsqueeze(-1).expand_as(perm_mat_list[0])
                y_values = perm_mat_list[0].masked_select(expanded_mask).view(-1, perm_mat_list[0].size(2))
                
                
                
                batch_correct, batch_total_valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                _tp, _fp, _fn = calculate_f1_score(prediction_tensor, y_values_matching)
                

                # Accumulate batch statistics
                epoch_correct += batch_correct
                epoch_total_valid += batch_total_valid
                tp += _tp
                fp += _fp
                fn += _fn
                
                
                
                
                
            bs = perm_mat_list[0].size(0)
            epoch_loss += loss.item() * bs
        
        # Calculate final metrics
           
        precision_global = tp / (tp + fp + 1e-8)
        recall_global = tp / (tp + fn + 1e-8)
        
        # Global F1 score
        epoch_f1 = 2 * (precision_global * recall_global) / (precision_global + recall_global + 1e-8)
        
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0
        
        
        epoch_loss = epoch_loss / dataset_size
        
        if (epoch+1) % cfg.STATISTIC_STEP == 0:
            if local_rank == output_rank:
                accs, f1_scores, error_dict = eval.eval_model(model, dataloader["test"], local_rank, output_rank)
                all_error_dict[epoch+1] = error_dict
                # wandb.log({"ep_loss": epoch_loss, "ep_acc": epoch_acc, "ep_f1": epoch_f1, "mean test_acc": torch.mean(accs), "mean test_f1": torch.mean(f1_scores)})
        
        if local_rank == output_rank:
            # wandb.log({"ep_loss": epoch_loss, "ep_acc": epoch_acc, "ep_f1": epoch_f1})
            print(f'epoch loss: {epoch_loss}, epoch accuracy: {epoch_acc}, epoch f1_score: {epoch_f1}')
        
        if cfg.save_checkpoint and local_rank == output_rank:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))
        scheduler.step()
        
    
    
    return model, all_error_dict


if __name__ == "__main__":
    # print('Using config file from: ', os.sys.argv[1])
    cfg = update_params_from_cmdline(default_params=cfg)
    
    #windows
    # dist.init_process_group(backend='gloo', init_method='env://')
    
    #linux
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=60))
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ['LOCAL_RANK']) 
    output_rank = 0
    
    import json
    import os

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)
    
    # if local_rank == output_rank:
    #     wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="NMT",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": cfg.TRAIN.LR,
    #     "architecture": "NMT",
    #     "dataset": cfg.DATASET_NAME,
    #     "epochs": lr_schedules[cfg.TRAIN.lr_schedule][0],
    #     "batch_size": cfg.BATCH_SIZE,
    #     "cfg_full": cfg
    #     }
    #     )

    torch.manual_seed(cfg.RANDOM_SEED)
    
    #Edit
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    
    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES * world_size} # 
    image_dataset = {
        x: GMDataset(x, cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(384, 384)) for x in ("train", "test")
    }
    
    sampler = {
    "train": DistributedSampler(image_dataset["train"]),
    "test": DistributedSampler(image_dataset["test"])
    }
    
    dataloader = {x: get_dataloader(image_dataset[x],sampler[x], fix_seed=(x == "test")) for x in ("train", "test")}

    model = NMT()
        
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = InfoNCE_Loss(temperature=cfg.TRAIN.temperature)
    backbone_params = model.module.backbone_params

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.03),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = optim.Adam(opt_params, weight_decay=cfg.TRAIN.weight_decay)
    

    if not Path(cfg.model_dir).exists():
        Path(cfg.model_dir).mkdir(parents=True)

    num_epochs, _, __ = lr_schedules[cfg.TRAIN.lr_schedule]
    model, all_error_dict = train_eval_model(model, 
                                   criterion, 
                                   optimizer,
                                   dataloader,
                                   cfg.TRAIN.clip_norm, 
                                   num_epochs=num_epochs,
                                   local_rank=local_rank,
                                   output_rank = output_rank,
                                   world_size = world_size,
                                   resume=cfg.warmstart_path is not None, 
                                   start_epoch=0,
                                   )
    
    if local_rank == output_rank:
        if all_error_dict is not None:
            output_folder = "errors"
            os.makedirs(output_folder, exist_ok=True)
            for epoch, class_dict in all_error_dict.items():
                save_dict = {}
                for class_, e_dict in class_dict.items():
                    e_dict_ = sorted(e_dict.items())
                    e_len, e_idx = e_dict_[-1]
                    result_tensor = torch.zeros(e_len, dtype=torch.float).to(device)
                    e_num = 0
                    for errors in e_dict_:
                        e_len, e_idx = errors
                        
                        e_ten = e_idx[1]
                        t1_resized = torch.cat((e_ten, torch.zeros(result_tensor.size(0) - e_ten.size(0), dtype=result_tensor.dtype).to(device))).to(device)
                        
                        result_tensor += t1_resized
                        e_num += e_idx[0]
                    # e_num = e_idx[0]
                    # e_tensor = e_idx[1]
                    
                    e_avg = (result_tensor/e_num).cpu().detach().tolist()
                    
                    save_dict[class_] = e_avg
                    
                file_name = f"{output_folder}/epoch_{epoch}_save_dict.json"
                with open(file_name, "w") as json_file:
                    json.dump(save_dict, json_file)
            
                
    dist.destroy_process_group()
