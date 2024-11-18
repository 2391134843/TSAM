import torch
import torch.nn as nn
import torch.nn.functional as F

class TransELayer(nn.Module):
    def __init__(self, margin=1, p_norm=1):
        super(TransELayer, self).__init__()
        self.margin = margin
        self.p_norm = p_norm

    def forward(self, h_embed, r_embed):
        
        h_embed = h_embed + r_embed 
        return h_embed