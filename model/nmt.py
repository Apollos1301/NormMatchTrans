import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment

import utils.backbone
from model.sconv_archs import SConv
from model.positionalEmbedding import Pointwise2DPositionalEncoding
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.evaluation_metric import make_perm_mat_pred
from utils.visualization import easy_visualize
from model.nGPT_decoder import NGPT_DECODER
from model.nGPT_encoder import NGPT_ENCODER


def normalize_over_channels(x, eps=1e-9):
    channel_norms = torch.norm(x, dim=1, keepdim=True) + eps
    return x / channel_norms



def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def make_queries(h_s, h_t):
    n_s = h_s.size(dim=1)
    n_t = h_t.size(dim=1)

    queries = []

    for i in range(0,n_s):
        for j in range(0, n_t):
            query = torch.cat((h_s[:,i,:], h_t[:,j,:]), dim=-1)
            queries.append(query)
    queries = torch.stack(queries, dim=1)
    
    return queries

def pad_input(input_tensor, target_length=40):
    """
    Pads the input tensor to match the target length in the data point dimension.
    
    Parameters:
    - input_tensor: Input tensor of shape [batch_size, current_length, feature_dim].
    - target_length: The desired length for padding.

    Returns:
    - Padded tensor of shape [batch_size, target_length, feature_dim].
    """
    current_length = input_tensor.size(1)
    if current_length >= target_length:
        return input_tensor  # No padding needed if current length meets or exceeds target length

    # Padding to the right along the data point dimension (dim=1)
    padding_size = target_length - current_length
    padded_tensor = F.pad(input_tensor, (0, 0, 0, padding_size), "constant", 0)
    return padded_tensor

def create_source_masks(source_points, n_points, max_length=40):
    """
    Create masks for the source points tensor used in the TransformerDecoder.

    Parameters:
    - source_points: Tensor of shape [batch_size, seq_len, feature_dim].
    - n_points: List or tensor indicating the original length of each sequence before padding.
    - max_length: The maximum sequence length (after padding).

    Returns:
    - source_points_mask: Upper triangular mask of shape [seq_len, seq_len] (or None if not needed).
    - source_key_padding_mask: Padding mask of shape [batch_size, max_length].
    """
    batch_size, seq_len, _ = source_points.size()

    # Create target mask (if needed for causal attention)
    source_points_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(source_points.device)

    # Create key padding mask for source points
    source_key_padding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=source_points.device)
    for i, length in enumerate(n_points):
        source_key_padding_mask[i, length:] = True  # Mark padding positions with True

    return source_points_mask, source_key_padding_mask

def cosine_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Places vectors onto the unit-hypersphere

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

class Scale(nn.Module):
    """
    A module that manages learnable scaling parameters to ensure different learning rates
    from the rest of the parameters in the model (see pages 5 and 19)
    
    Args:
        dim (int): Dimension of the scaling parameter
        scale (float): Initial scale value
        init (float): Initial value for the scaling parameter
        device (str, optional): Device to store the parameter on
    """
    def __init__(self, dim: int, heads: int = 1, scale: float = 1.0, init: float = 1.0, device=None):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
                      if device is None else device)
        self.init = init
        self.scale = scale
        self.s = nn.Parameter(torch.ones(heads, dim, device=self.device) * scale)
            # heads == 1 gives us a single regular vector
            # heads > 1 gets used in attention mechanism for different scaling vector for each head
    
    def forward(self):
        """Compute the effective scaling factor."""
        return self.s * (self.init / self.scale) # shape (heads, dim)

class ModelConfig:
    """
    Design your N-GPT here
    """
    dim: int = 128
    device: str = None
        # defaults to best available GPU/CPU
    num_layers: int = 6
    num_heads: int = 4 # number of heads in the multi-head attention mechanism
    mlp_hidden_mult: float = 4

class NMT(utils.backbone.SwinV2): #Gmt_base #Vit_base
    def __init__(self):
        super(NMT, self).__init__()
        self.model_name = 'Transformer'
        self.psi = SConv(input_features=cfg.SPLINE_CNN.input_features, output_features=cfg.Matching_TF.d_model)
        # self.mlp = MLPQuery(cfg.Matching_TF.d_model, 1024, cfg.Matching_TF.d_model, batch_norm=cfg.Matching_TF.batch_norm)
        
        self.vit_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features//2, cfg.Matching_TF.d_model)

        # self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # self.scaled_mlp = MLP_scaled(cfg.Matching_TF.d_model*2, cfg.Matching_TF.d_model//2, cfg.Matching_TF.d_model)      
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()

        # self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model= cfg.Matching_TF.d_model, 
        #                                                    nhead= cfg.Matching_TF.n_head, 
        #                                                    batch_first=True)
        # self.tf_decoder_layer = nn.TransformerDecoderLayer(d_model= cfg.Matching_TF.d_model,
        #                                                    nhead=cfg.Matching_TF.n_head,
        #                                                    batch_first=True)
        # self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=cfg.Matching_TF.n_encoder)
        # self.tf_decoder = nn.TransformerDecoder(self.tf_decoder_layer, num_layers=cfg.Matching_TF.n_decoder)
        
        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        
        self.n_gpt_decoder_2 = NGPT_DECODER(nGPT_decoder_config)
        # self.n_gpt_decoder_2.load_state_dict(self.n_gpt_decoder.state_dict())
        
        # self.anchorMLP = MLP_scaled(cfg.Matching_TF.d_model*2, cfg.Matching_TF.d_model//3, 1)
        
        # nGPT_encoder_config = ModelConfig()
        # nGPT_encoder_config.dim = cfg.Matching_TF.d_model
        # nGPT_encoder_config.num_layers = cfg.Matching_TF.n_encoder
        # nGPT_encoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        # nGPT_encoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        # self.n_gpt_encoder = NGPT_ENCODER(nGPT_encoder_config)
        
        # self.prot_MLP = MLP_prototype(cfg.Matching_TF.d_model)
        
        self.w_cosine = PairwiseWeightedCosineSimilarity(cfg.Matching_TF.d_model)
        
        self.global_state_dim = 1024
        
    
    def normalize_linear(self, module):
        """
        Helper method to normalize Linear layer weights where one dimension matches model dim
        """
        # Find the dimension that matches cfg.dim
        dim_to_normalize = None
        for dim, size in enumerate(module.weight.shape):
            if size == cfg.Matching_TF.d_model:
                dim_to_normalize = dim
                break
        
        if dim_to_normalize is not None:
            # Normalize the weights
            module.weight.data = cosine_norm(module.weight.data, dim=dim_to_normalize)
    
    def enforce_constraints(self):
        """
        Enforces constraints after each optimization step:
        2. Cosine normalization on Linear layer weights where one dimension matches model dim
        """
        # self.vit.enforce_constraints()
        self.n_gpt_decoder.enforce_constraints()
        self.n_gpt_decoder_2.enforce_constraints()
        
    
    
    
    def update_order(self, source_nodes, input_order):
        B, _, _ = source_nodes.shape
        for b in range(B):
            source_nodes[b, :, :] = source_nodes[b, input_order[b], :]
        return source_nodes
    

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        n_points_sample, 
        perm_mats,
        eval_pred_points=None,
        in_training=True,
        input_order=None,
        matched_points_mask=None,
        matched_padding_mask_hs=None,
        matched_padding_mask_ht=None,
    ):
        batch_size = graphs[0].num_graphs
        global_list = []
        orig_graph_list = []
        node_feat_list = []
        # for visualisation purposes only
        graph_list = []
        global_feat = 0
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            #SwinV2
            swin_nodes, swin_edges, glob_token= self.swin(image)
            
            swin_nodes = normalize_over_channels(swin_nodes)
            swin_edges = normalize_over_channels(swin_edges)
            
            swin_U = concat_features(feature_align(swin_nodes, p, n_p, (384, 384)), n_p)
            swin_F = concat_features(feature_align(swin_edges, p, n_p, (384, 384)), n_p)
            
            node_features = torch.cat((swin_U, swin_F), dim=-1)
            
            #VIT
            #vit_nodes, vit_edges, glob_token = self.vit(image)
            
            #vit_nodes = normalize_over_channels(vit_nodes)
            #vit_edges = normalize_over_channels(vit_edges)
            
            #vit_U = concat_features(feature_align(vit_nodes, p, n_p, (224, 224)), n_p)
            #vit_F = concat_features(feature_align(vit_edges, p, n_p, (224, 224)), n_p)
            
            #node_features = torch.cat((vit_U, vit_F), dim=-1)
            
            
            #GMT
            # gmt_nodes, gmt_edges, glob_token = self.gmt(image, p, n_p)
            
            # gmt_nodes = normalize_over_channels(gmt_nodes)
            # gmt_edges = normalize_over_channels(gmt_edges)
            
            # gmt_U = concat_features(gmt_nodes, n_p)
            # gmt_F = concat_features(gmt_edges, n_p)
            
            # node_features = torch.cat((gmt_U, gmt_F), dim=-1)
            
            
            
            graph.x = node_features
            # for visualisation purposes only
            graph_list.append(graph.to_data_list())

            # node + edge features from vgg
            vit_features = self.vit_to_node_dim(node_features)
            # splineCNN spatial features 
            h = self.psi(graph)
            h_res = h + vit_features
                            
            (h_res, mask) = to_dense_batch(h_res, graph.batch, fill_value=0)

            if cfg.Matching_TF.pos_encoding:
                h_res = h_res + self.pos_encoding(p)
                
            #global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            global_feature = self.glob_to_node_dim(glob_token)
            #global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            
            # global_feature = self.linear_cls(global_feature)
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)

            orig_graph_list.append((h_res,mask))

        h_s, s_mask = orig_graph_list[0]
        h_t, t_mask = orig_graph_list[1]

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        
        
        
        batch_size, seq_len, _ = h_s.shape
        padding_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool).to(h_s.device)
        if in_training is True:
            for idx, e in enumerate(n_points_sample):
                if cfg.Matching_TF.global_feat:
                    s_mask[idx, e+1:] = False
                    h_s[idx, e+1:, :] = 0
                    
                    t_mask[idx, e+1:] = False
                    h_t[idx, e+1:, :] = 0
                    
                    padding_mask[idx, :, e+1:] = 1
                    padding_mask[idx, e+1:, :] = 1
                else:
                    s_mask[idx, e:] = False
                    h_s[idx, e:, :] = 0
                    
                    t_mask[idx, e:] = False
                    h_t[idx, e:, :] = 0
          
        
            
        
        source_points_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool).to(h_t.device)
        hs_dec_output = self.n_gpt_decoder(h_s, source_points_mask, padding_mask, h_t)
        
        ht_decoder_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool).to(h_t.device)
        ht_dec_output = self.n_gpt_decoder_2(h_t, ht_decoder_mask, padding_mask, h_s)
            
        # paired_global_feat = torch.cat([hs_dec_output[ :, 0, :], ht_dec_output[ :, 0, :]], dim=-2).to(hs_dec_output.device)
        # paired_global_feat = self.scaled_mlp(paired_global_feat)
        
        #Encoder-Decoder
        # hs_dec_output = hs_dec_output[:, 1:, :]
        # target_points = cosine_norm(target_points)
        # ht_dec_output = target_points[:, 1:, :]
        
        #Decoder-Decoder
        hs_dec_output = hs_dec_output[:, 1:, :]
        ht_dec_output = ht_dec_output[:, 1:, :]
        
        prototype_score = []
        
        
        # h_t_norm = cosine_norm(h_t)
        # print(hs_dec_output.size(), ht_dec_output.size())
        
        sim_score = self.w_cosine(hs_dec_output, ht_dec_output) #self.w_cosine(hs_dec_output, h_t_norm)
        
        sim_score_all = torch.tensor([]) #self.w_cosine(hs_dec_output_, ht_dec_output_)
        
        
        return sim_score, sim_score_all, prototype_score, hs_dec_output, ht_dec_output #gloabl_feat_score  #target_points, hs_dec_output
        


class MLP_prototype(nn.Module):
    def __init__(self, model_dim):
        super(MLP_prototype, self).__init__()
        self.layer1 = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x, dropout=None):
        # Feedforward
        x = self.layer1(x)
        if dropout is not None:
            x = torch.nn.functional.dropout(x, p=dropout)
        
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, l2_scaling):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.l2_scaling = l2_scaling
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
        output = self.out(x)
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return output 

class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, l2_scaling):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.l2_scaling = l2_scaling
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
        output = self.out(x)
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return output

class MLPQuery(nn.Module):
    def __init__(self, node_dim, hidden_size, hidden_out, batch_norm):
        super(MLPQuery, self).__init__()
        self.lin1 = nn.Linear(2 * node_dim, hidden_size)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_out)

    def forward(self, x):
            x = self.lin1(x)
            if self.batch_norm:
                x = self.bn(torch.transpose(x, 1, 2))
                x = torch.transpose(x, 1, 2)
            x = nn.functional.relu(x)
            out = self.lin2(x)
            return out



class PairwiseWeightedCosineSimilarity(nn.Module):
    def __init__(self, node_feature_dim):
        super(PairwiseWeightedCosineSimilarity, self).__init__()
        # Initialize weights with ones for each feature dimension
        self.w = nn.Parameter(torch.ones(1, 1, node_feature_dim))
    
    def forward(self, x, y):
        # x and y have shape [batch_size, nodes, node_feature]
        
        # Apply weights
        x_weighted = x #* self.w  # Shape: [batch_size, nodes_x, node_feature]
        y_weighted = y #* self.w  # Shape: [batch_size, nodes_y, node_feature]
        
        y_weighted_transposed = y_weighted.transpose(-2, -1)  # Shape: [batch_size, node_feature, nodes_y]
        numerator = torch.bmm(x_weighted, y_weighted_transposed)  # Shape: [batch_size, nodes_x, nodes_y]
        
        x_norm = torch.norm(x_weighted, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_x]
        y_norm = torch.norm(y_weighted, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_y]
        #epsilon = 1e-8  # To prevent division by zero
        #x_norm = x_norm + epsilon
        #y_norm = y_norm + epsilon
        
        denominator = torch.bmm(x_norm.unsqueeze(2), y_norm.unsqueeze(1))  # Shape: [batch_size, nodes_x, nodes_y]
        
        # Compute cosine similarity matrix
        cosine_similarity = numerator / denominator  # Shape: [batch_size, nodes_x, nodes_y]
        #cosine_similarity = torch.clamp(cosine_similarity, -1 + epsilon, 1 - epsilon)
        
        return cosine_similarity
        
class MLP_scaled(nn.Module):
    """
    Multilayer Perceptron (MLP) module with optional gating and dropout.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        device (str or torch.device): Device to run the module on.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device = None
    ):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        # the up, down, and gate projections
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_dim, bias=False, device=self.device)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        x_ = self.lin1(x)
        x_ = self.relu(x_)
        x_ = self.lin2(x_)
        return x_
