import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

import utils.backbone
from model.sconv_archs import SConv
from model.positionalEmbedding import Pointwise2DPositionalEncoding
from utils.config import cfg
from utils.feature_align import feature_align
from model.nGPT_decoder import NGPT_DECODER
from model.nGPT_encoder import NGPT_ENCODER


def normalize_over_channels(x, eps=1e-9):
    channel_norms = torch.norm(x, dim=1, keepdim=True) + eps
    return x / channel_norms



def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


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
        
        self.vit_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features//2, cfg.Matching_TF.d_model)

        
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
        
        
        # nGPT_encoder_config = ModelConfig()
        # nGPT_encoder_config.dim = cfg.Matching_TF.d_model
        # nGPT_encoder_config.num_layers = cfg.Matching_TF.n_encoder
        # nGPT_encoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        # nGPT_encoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        # self.n_gpt_encoder = NGPT_ENCODER(nGPT_encoder_config)
        
        
        self.w_cosine = PairwiseCosineSimilarity(cfg.Matching_TF.d_model)
        
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
                
            global_feature = self.glob_to_node_dim(glob_token)
            #global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            
            #global_feature = self.linear_cls(global_feature)
            
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
                h_s[idx, e+1:, :] = 0
                h_t[idx, e+1:, :] = 0
                
                padding_mask[idx, :, e+1:] = 1
                padding_mask[idx, e+1:, :] = 1
                
          
        
            
        hs_dec_output = self.n_gpt_decoder(h_s, padding_mask, h_t)
        
        ht_dec_output = self.n_gpt_decoder_2(h_t, padding_mask, h_s)
            
        # paired_global_feat = torch.cat([hs_dec_output[ :, 0, :], ht_dec_output[ :, 0, :]], dim=-2).to(hs_dec_output.device)
        # paired_global_feat = self.scaled_mlp(paired_global_feat)
        
        #Encoder-Decoder
        # hs_dec_output = hs_dec_output[:, 1:, :]
        # target_points = cosine_norm(target_points)
        # ht_dec_output = target_points[:, 1:, :]
        
        #Decoder-Decoder
        hs_dec_output = hs_dec_output[:, 1:, :]
        ht_dec_output = ht_dec_output[:, 1:, :]
        
        sim_score = self.w_cosine(hs_dec_output, ht_dec_output) 
        
        return sim_score, hs_dec_output, ht_dec_output 
        

class PairwiseCosineSimilarity(nn.Module):
    def __init__(self, node_feature_dim):
        super(PairwiseCosineSimilarity, self).__init__()
    
    def forward(self, x, y):
        
        y_transposed = y.transpose(-2, -1)  # Shape: [batch_size, node_feature, nodes_y]
        numerator = torch.bmm(x, y_transposed)  # Shape: [batch_size, nodes_x, nodes_y]
        
        x_norm = torch.norm(x, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_x]
        y_norm = torch.norm(y, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_y]
        
        denominator = torch.bmm(x_norm.unsqueeze(2), y_norm.unsqueeze(1))  # Shape: [batch_size, nodes_x, nodes_y]
        
        # Compute cosine similarity matrix
        cosine_similarity = numerator / denominator  # Shape: [batch_size, nodes_x, nodes_y]
        
        return cosine_similarity
