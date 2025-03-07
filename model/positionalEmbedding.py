import math
import torch
import torch.nn as nn


class Pointwise2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(Pointwise2DPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.dropout = nn.Dropout(p=0.1)

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        
        self.d_model = int(d_model / 2)

    def forward(self, pos):
        """
        x  : (batch size, n_points, d_model)
        pos: (batch size, n_points, 2) 
        out : (batch size, n_points, d_model)
        """
        
        batch_size, n_points, _= pos.size()

        div_term = torch.exp(torch.arange(0., self.d_model, 2) * 
                             -(math.log(10000.0) / self.d_model)).to(pos.device)
        pe = torch.zeros(batch_size,  n_points, self.d_model * 2).to(pos.device)
        
        pos_x = pos[:,:,0].unsqueeze(2)
        pos_y = pos[:,:,1].unsqueeze(2)

        pe[:,:,0:self.d_model:2] = torch.sin(pos_x * div_term)
        pe[:,:,1:self.d_model:2] = torch.cos(pos_x * div_term)

        pe[:,:, self.d_model::2] = torch.sin(pos_y * div_term)
        pe[:,:, self.d_model + 1::2] = torch.cos(pos_y * div_term)

        return pe

if __name__ == "__main__":
    points = torch.randn((8, 10, 2))
    pos_enc = Pointwise2DPositionalEncoding(512, 256, 256)
    pe  = pos_enc(points)
    print(pe.shape)


