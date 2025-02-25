import torch.nn as nn
from torchvision import models
from utils.vit import *
from utils.gmt import *
from utils.swinV2 import *


class VGG16_base(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d(1, 1))

        return node_layers, edge_layers, final_layers


class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)


class Vit_base(nn.Module):
    def __init__(self):
        super(Vit_base, self).__init__()
        self.vit = ViT(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12)
        self.backbone_params = list(self.vit.parameters())

        # --------------------------load parameters for base ViT-----------------------------------
        weights_dict = torch.load('./utils/checkpoints/vit_base.pth')
        del weights_dict['model']['head.weight']
        del weights_dict['model']['head.bias']
        print(self.vit.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------


    @property
    def device(self):
        return next(self.parameters()).device
    
    
class Gmt_base(nn.Module):
    def __init__(self):
        super(Gmt_base, self).__init__()
        self.gmt= Gmt(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12)
        self.backbone_params = list(self.gmt.parameters())

        # --------------------------load parameters for base Gmt--------------------------
        weights_dict = torch.load('./utils/checkpoints/vit_base.pth')
        del weights_dict['model']['head.weight']
        del weights_dict['model']['head.bias']
        print(self.gmt.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------

    @property
    def device(self):
        return next(self.parameters()).device
    
class SwinV2(nn.Module):
    def __init__(self):
        super(SwinV2, self).__init__()
        self.swin = SwinTransformer()
        self.backbone_params = list(self.swin.parameters())

        # --------------------------load parameters for base SwinV2--------------------------
        weights_dict = torch.load('./utils/checkpoints/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth')
        del weights_dict['model']['head.weight']
        del weights_dict['model']['head.bias']
        print(self.swin.load_state_dict(weights_dict['model'], strict=False))
        # ------------------------------------------------------------------------------------------

    @property
    def device(self):
        return next(self.parameters()).device