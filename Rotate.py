import torch
import torch.nn as nn

class RotatELayer(nn.Module):
    def __init__(self, dim, margin=1.0):
        super(RotatELayer, self).__init__()
        self.margin = margin
        self.dim = dim // 2  
        self.bn0 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)

    def forward(self, e_embed, r_embed):
        
        

        
        h_embed = e_embed[:, 0, :]  
        h_embed = self.bn0(h_embed)
     
        h_embed = self.input_drop(h_embed)
     

        h_re, h_im = torch.chunk(h_embed, 2, dim=-1)

        r_phase = r_embed[:, :self.dim]  
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        h_r_re = h_re * r_re - h_im * r_im
        h_r_im = h_re * r_im + h_im * r_re
        h_embed= torch.cat([h_r_re, h_r_im], dim=-1)
        

        return h_embed


