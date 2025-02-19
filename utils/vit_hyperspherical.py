from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

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

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 device=None):
        super(Attention, self).__init__()
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop_ratio)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 

        # Define linear projections for queries, keys, and values
        self.Wq = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)
        self.Wk = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)
        self.Wv = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)

        # the scaling factor to apply to the normalized queries & keys (see page 4)
        self.s_qk = Scale(self.head_dim, heads=num_heads, scale = 1. / math.sqrt(dim), device=self.device)

        # the scaling factor to apply to the attention logits to restore a variance of 1 (see page 4)
        self.scale = self.head_dim ** 0.5

        # Output projection that mixes all the attention heads back together
        self.Wo = nn.Linear(num_heads * self.head_dim, dim, bias=False, device=self.device)
        # this flag designates Wo to have a different parameter initialization as defined below in Model
        self.Wo.GPT_scale_init = 1

    def forward(self, x):
        batch_size, num_patches, _ = x.shape
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            # shape: (batch_size, num_patches, dim) -> (batch_size, num_patches, num_heads * head_dim)
        # Reshape projections to separate heads
        q = q.view(batch_size, num_patches, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_patches, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_patches, self.num_heads, self.head_dim)

        # normalizing & scaling our queries  & keys (see page 4)
        s_qk = self.s_qk() # (num_heads, head_dim)
        q = cosine_norm(q) * s_qk # then scale each head
        k = cosine_norm(k) * s_qk # no shape change

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, num_patches, head_dim)
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2) 
        
        # Compute attention logits (compare queries & keys)
        logits = (q @ k.transpose(-2, -1)) * self.scale # (batch_size, num_heads, num_patches, num_patches)
        
        # Compute attention scores (grab the relevant values that correspond to the attention logits)
        scores =  F.softmax(logits, dim=-1) @ v # (batch_size, n_heads, num_patches, head_dim)
        # Combine heads
        scores = scores.transpose(1, 2).contiguous().view(batch_size, num_patches, -1) 
            # (batch_size, num_patches, n_heads * head_dim)
        out = self.Wo(scores)
        return out
        


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., device = None):
        super().__init__()
        
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        # the up, down, and gate projections
        self.Wup = nn.Linear(in_features, hidden_features, bias=False, device=self.device)
        self.Wgate = nn.Linear(in_features, hidden_features, bias=False, device=self.device)
        self.Wdown = nn.Linear(hidden_features, out_features, bias=False, device=self.device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1

        # the learnable scaling factors
        self.s_u = Scale(hidden_features, device=device)
        self.s_v = Scale(hidden_features, device=device)

        # the varaince-controlling scaling term, needed to benefit from SiLU (see appendix A.1)
        self.scale = math.sqrt(in_features)
        

    def forward(self, x):
        u = self.Wup(x) # (batch_size, seq_len, hidden_dim)
        v = self.Wgate(x)
        # scale them
        u = u * self.s_u()
        v = v * self.s_v() * self.scale 
        # now perform the nonlinearity gate
        hidden = u * F.silu(v) # (batch_size, seq_len, hidden_dim)
        out = self.Wdown(hidden)
        return  out# (batch_size, seq_len, output_dim)


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 device=None):
        super(Block, self).__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        ### attention connection
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # eigen learning rate vector
        self.alpha_A = Scale(dim, init = 0.05, scale = 1. / math.sqrt(dim), device=self.device) #init= 0.05
            # not sure what scale to use with a_A and a_M. At one point i had it as 1./math.sqrt(dim)
            # but now i can't find the reference to that in the paper

        ### feedforward connection
        # ensures mlp_hidden_mult maintains the same parameter count as if we were using a not-gated MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop_ratio)
        # eigen learning rate vector
        self.alpha_M = Scale(dim, init = 0.05, scale = 1. / math.sqrt(dim), device=self.device) #init= 0.05

    def forward(self, h):
        h_A = cosine_norm(self.attn(h))
        h = cosine_norm(h + self.alpha_A() * (h_A - h))
        
        h_M = cosine_norm(self.mlp(h))
        h = cosine_norm(h + self.alpha_M() * (h_M - h))
        return h


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=1024, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.img_size = img_size
        self.depth = depth
        self.patch_size = patch_size
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.fc_norm = norm_layer(embed_dim)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 256, 512]
        # [1, 1, 512] -> [B, 1, 512]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 257, 512]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed

        nodes, edges = torch.empty(x[:, 1:].shape), torch.empty(x[:, 1:].shape)

        for i in range(self.depth):
            x = self.blocks[i](x)

            if i == self.depth - 2:
                nodes = self.fc_norm(x)[:, 1:]
            elif i == self.depth - 1:
                edges = self.fc_norm(x)[:, 1:]

        x = self.fc_norm(x)

        glb = x[:, 0]

        num_patches = self.img_size // self.patch_size

        nodes, edges = nodes.transpose(1, 2), edges.transpose(1, 2)
        x = x[:, 1:].transpose(1, 2)

        nodes, edges = nodes.reshape((x.shape[0], x.shape[1], num_patches, num_patches)), \
                            edges.reshape((x.shape[0], x.shape[1], num_patches, num_patches)), \

        return nodes, edges, glb

    def normalize_linear(self, module):
        """
        Helper method to normalize Linear layer weights where one dimension matches model dim
        """
        # Find the dimension that matches cfg.dim
        dim_to_normalize = None
        for dim, size in enumerate(module.weight.shape):
            if size == self.embed_dim:
                dim_to_normalize = dim
                break
        
        if dim_to_normalize is not None:
            # Normalize the weights
            module.weight.data = cosine_norm(module.weight.data, dim=dim_to_normalize)

    def enforce_constraints(self):
        """
        Enforces constraints after each optimization step:
        1. Absolute value constraint on eigen learning rate parameters
        2. Cosine normalization on Linear layer weights where one dimension matches model dim
        """
        # Enforce absolute value on eigen learning rates
        for layer in self.blocks:
            layer.alpha_A.s.data.abs_()
            layer.alpha_M.s.data.abs_()
        
        # Cosine normalize relevant Linear layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self.normalize_linear(module)

    def _init_vit_weights(self, module):
        """
        ViT weight initialization
        :param m: module
        """
        std = math.sqrt(self.embed_dim) 

        if isinstance(module, (nn.Linear, nn.Parameter)):
            # specific weight matrices at the end of each layer are given smaller std 
            # originally this was done in GPT-2 to keep the residual stream small
            if hasattr(module, 'GPT_scale_init'):
                std *= (2 * self.depth) ** -0.5

            # carries out the actual initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # biases, if any, should instead be initialized to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # the embedding matrix doesn't count as an nn.Linear so we've gotta do it again for that
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)