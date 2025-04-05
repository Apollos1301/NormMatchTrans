import time
from pathlib import Path
import torch

from utils.config import cfg
from utils.evaluation_metric import calculate_correct_and_valid, calculate_f1_score, get_pos_neg


def sinkhorn_logspace(
    similarity: torch.Tensor,
    epsilon: float = 0.035,
    max_iter: int = 27
) -> torch.Tensor:
    """
    Log-space Sinkhorn to convert a batch of similarity matrices into 
    doubly-stochastic matrices.
    
    Args:
        similarity: [batch_size, n, m] matrix of similarities
        epsilon:    Entropic regularization (larger => smoother distribution)
        max_iter:   Number of Sinkhorn iterations in log domain
    
    Returns:
        [batch_size, n, m] doubly-stochastic matrix
    """
    log_Q = similarity / epsilon

    for _ in range(max_iter):
        log_sum_rows = torch.logsumexp(log_Q, dim=2, keepdim=True)
        log_Q = log_Q - log_sum_rows

        log_sum_cols = torch.logsumexp(log_Q, dim=1, keepdim=True)
        # broadcast subtraction
        log_Q = log_Q - log_sum_cols

    Q = torch.exp(log_Q)
    return Q

def eval_model(model, dataloader, local_rank, output_rank, eval_epoch=None, verbose=True):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.model_dir) / "params" / "{:04}".format(eval_epoch) / "params.pt")
        if local_rank == output_rank:
            print("Loading model parameters from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    ds.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)
    error_dist_dict = {}
    

    for i, cls in enumerate(classes):
        if local_rank == output_rank:
            if verbose:
                print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        
        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)

        
        result_dict = {}
        tp = 0
        fp = 0
        fn = 0
        epoch_f1 = 0.0
        epoch_correct = 0
        epoch_total_valid = 0
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]

            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)
            num_nodes_s = points_gt[0].size(1)
            num_nodes_t = points_gt[1].size(1)

            iter_num = iter_num + 1

            visualize = k == 0 and cfg.visualize
            visualization_params = {**cfg.visualization_params, **dict(string_info=cls, true_matchings=perm_mat_list)}

            matched_instances_at_step = []
            with torch.set_grad_enabled(False):
                
                matchings = []
                B, N_s, N_t = perm_mat_list[0].size()
                n_points_sample = n_points_gt[0]
                
                eval_pred_points = 0
                j_pred = 0
                predictions_list = []
                # keypoint_order = []
                for _ in range(B):
                    predictions_list.append([])
                    
                
                
                    
                similarity_scores, _, _, _ = model(data_list, points_gt, edges, n_points_gt, n_points_sample, perm_mat_list, eval_pred_points, in_training= False)
                
                batch_size = similarity_scores.shape[0]
                
                sinkhorn = sinkhorn_logspace(similarity_scores)
                
                sinkhorn_max = torch.argmax(sinkhorn, dim=-1)
                for np in range(N_t):
                    for b in range(batch_size):
                        if eval_pred_points < n_points_gt[0][b]:
                            predictions_list[b].append(sinkhorn_max[b, eval_pred_points].item())
                        else:
                            predictions_list[b].append(-1)
                    eval_pred_points +=1
                
                
                
                prediction_tensor = torch.tensor(predictions_list).to(perm_mat_list[0].device)
                y_values_matching = torch.argmax(perm_mat_list[0], dim=-1)
                batch_correct, batch_total_valid = calculate_correct_and_valid(prediction_tensor, y_values_matching)
                
                
                
                error_list = (prediction_tensor != y_values_matching).int()
            
                for idx, e in enumerate(n_points_gt[0]):
                    if e.item() not in result_dict:
                        result_dict[e.item()] = [1, error_list[idx,:e.item()]]
                    result_dict[e.item()][0] += 1
                    result_dict[e.item()][1] += error_list[idx,:e.item()]
                # Iterate through the batch
                # _tp, _fp, _fn = calculate_f1_score(prediction_tensor, y_values_matching)

                
                epoch_correct += batch_correct
                epoch_total_valid += batch_total_valid
                # tp += _tp
                # fp += _fp
                # fn += _fn
            
            if iter_num % 40 == 0 and verbose: #cfg.STATISTIC_STEP
                running_speed = 40 * batch_num / (time.time() - running_since) #cfg.STATISTIC_STEP
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()
        
        
        dataset_size = len(dataloader.dataset)
        
        if epoch_total_valid > 0:
            epoch_acc = epoch_correct / epoch_total_valid
        else:
            epoch_acc = 0.0

        # precision_global = tp / (tp + fp + 1e-8)
        # recall_global = tp / (tp + fn + 1e-8)
        
        # Global F1 score
        # epoch_f1 = 2 * (precision_global * recall_global) / (precision_global + recall_global + 1e-8)
        
        accs[i] = epoch_acc
        # f1_scores[i] = epoch_f1
        if verbose:
            print("Class {} acc = {:.4f}".format(cls, accs[i]))
            
        error_dist_dict[cls] = result_dict
        
    # print(error_dist_dict)
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc in zip(classes, accs):
        print("{} = {:.4f}".format(cls, single_acc))
    print("average = {:.4f}".format(torch.mean(accs)))


    return accs, error_dist_dict
