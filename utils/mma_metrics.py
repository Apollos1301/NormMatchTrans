import torch
import numpy as np

def compute_mma(results_list, thresholds=[1.0, 3.0, 5.0]):
    """
    Computes Mean Matching Accuracy (MMA) given a list of results.
    
    results_list format: list of dictionaries, where each dict has:
       - 'seq_type': 'viewpoint' or 'illumination'
       - 'pred_indices': The predicted match index in image 2 for each point in image 1.
       - 'scaled_coords1': Tensor of valid coordinates in image 1.
       - 'scaled_coords2': Tensor of the *actual* matched coordinates in image 2.
    """
    viewpoint_results = {t: [] for t in thresholds}
    illum_results = {t: [] for t in thresholds}
    
    for res in results_list:
        seq_type = res['seq_type']
        
        preds = res['pred_indices']      # shape [N]
        coords2 = res['scaled_coords2']  # shape [N, 2]
        
        # Predict coordinates by looking up the predicted matches in the graph
        # Note: coords2 is already ordered such that the ith point in coords1 
        # corresponds to the ith point in coords2.
        # Thus, if pred_indices[i] = j, the model predicts coords1[i] maps to coords2[j].
        pred_coords2 = coords2[preds]
        
        # Euclidean pixel distance between prediction and optimal match
        pixel_errors = torch.norm(pred_coords2 - coords2, dim=1) # shape [N]
        
        # Calculate percentage of points below each threshold for this image pair
        for t in thresholds:
            correct_matches = (pixel_errors <= t).sum().item()
            total_points = len(pixel_errors)
            
            if total_points > 0:
                accuracy = correct_matches / total_points
                
                if seq_type == 'viewpoint':
                    viewpoint_results[t].append(accuracy)
                elif seq_type == 'illumination':
                    illum_results[t].append(accuracy)
                    
    # Aggregate to compute final Mean Matching Accuracy
    print("\n" + "="*50)
    print("Mean Matching Accuracy (MMA) Results:")
    print("="*50)
    
    overall_mma = {t: [] for t in thresholds}
    
    for t in thresholds:
        v_acc = np.mean(viewpoint_results[t]) if viewpoint_results[t] else 0.0
        i_acc = np.mean(illum_results[t]) if illum_results[t] else 0.0
        
        # Overall is the mean of all pairs (not the mean of the two means)
        all_accs = viewpoint_results[t] + illum_results[t]
        o_acc = np.mean(all_accs) if all_accs else 0.0
        
        overall_mma[t] = o_acc
        
        print(f"Threshold: {t} pixels")
        print(f"  -> Viewpoint (v_):    {v_acc*100:.2f}%")
        print(f"  -> Illumination (i_): {i_acc*100:.2f}%")
        print(f"  -> Overall:           {o_acc*100:.2f}%")
        print("-" * 50)
        
    return overall_mma
