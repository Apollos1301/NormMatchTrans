import torch
from torch import Tensor
import torch.nn.functional as F

def feature_align(raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None) -> Tensor:
    r"""
    Perform vectorized feature align on the image feature map using grid_sample.
    """
    if device is None:
        device = raw_feature.device

    batch_size, channels, h_feat, w_feat = raw_feature.shape
    _, max_points, _ = P.shape

    # Map point coordinates from [0, ori_size] directly to [-1, 1] range required by grid_sample
    # Note P is [x, y], ori_size is (W, H). 
    # normalize P to [0, 1] then to [-1, 1]
    ori_w, ori_h = ori_size
    
    # Map directly to [-1, 1]. PyTorch's align_corners=False intrinsically handles the -0.5 pixel 
    # centering, so we don't need to manually subtract step/2 like the old logic did.
    ori_w, ori_h = ori_size
    grid_x = P[..., 0] / ori_w * 2.0 - 1.0
    grid_y = P[..., 1] / ori_h * 2.0 - 1.0
    
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(1)  # Shape: [Batch, 1, max_points, 2]

    # Use PyTorch's highly optimized grid_sample to do bilinear interpolation across the entire batch natively
    # align_corners=False matches OpenCV/standard image coordinate behavior mapped directly from pixel sizes
    F_out = F.grid_sample(raw_feature, grid, mode='bilinear', padding_mode='border', align_corners=False)
    
    # Remove the dummy H dimension (F_out is [Batch, Channels, 1, max_points])
    F_out = F_out.squeeze(2)

    # Mask out padded points beyond ns_t sizes to be 0 for exact match of previous logic
    arange = torch.arange(max_points, device=device).expand(batch_size, max_points)
    mask = arange >= ns_t.unsqueeze(1)
    # Mask needs to broadcast over channels: mask is [B, max_points], needs to affect [B, C, max_points]
    F_out = F_out.transpose(1, 2)
    F_out[mask] = 0
    F_out = F_out.transpose(1, 2)

    return F_out

# Keep old functions around just in case, but unused by feature_align now
def interp_2d(z: Tensor, P: Tensor, ori_size: Tensor, feat_size: Tensor, out=None, device=None) -> Tensor:
    r"""
    Interpolate in 2d grid space. z can be 3-dimensional where the first dimension is feature dimension.

    :param z: :math:`(c\times w\times h)` feature map. :math:`c`: number of feature channels, :math:`w`: feature map
     width, :math:`h`: feature map height
    :param P: :math:`(n\times 2)` point set containing point coordinates. The coordinates are at the scale of
     the original image size. :math:`n`: number of points
    :param ori_size: :math:`(2)` size of the original image
    :param feat_size: :math:`(2)` size of the feature map
    :param out: optional output tensor
    :param device: output device. If not specified, it will be the same as the input
    :return: :math:`(c \times n)` extracted feature vectors
    """
    if device is None:
        device = z.device

    step = ori_size / feat_size
    if out is None:
        out = torch.zeros(z.shape[0], P.shape[0], dtype=torch.float32, device=device)
    for i, p in enumerate(P):
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate(z, p[0], p[1])

    return out


def bilinear_interpolate(im: Tensor, x: Tensor, y: Tensor, device=None):
    r"""
    Bi-linear interpolate 3d feature map to 2d coordinate (x, y).
    The coordinates are at the same scale of :math:`w\times h`.

    :param im: :math:`(c\times w\times h)` feature map
    :param x: :math:`(1)` x coordinate
    :param y: :math:`(1)` y coordinate
    :param device: output device. If not specified, it will be the same as the input
    :return: :math:`(c)` interpolated feature vector
    """
    if device is None:
        device = im.device
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)

    x0 = x0.to(torch.int32).to(device)
    x1 = x1.to(torch.int32).to(device)
    y0 = y0.to(torch.int32).to(device)
    y1 = y1.to(torch.int32).to(device)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    # to perform nearest neighbor interpolation if out of bounds
    if x0 == x1:
        if x0 == 0:
            x0 -= 1
        else:
            x1 += 1
    if y0 == y1:
        if y0 == 0:
            y0 -= 1
        else:
            y1 += 1

    x0 = x0.to(torch.float32).to(device)
    x1 = x1.to(torch.float32).to(device)
    y0 = y0.to(torch.float32).to(device)
    y1 = y1.to(torch.float32).to(device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out