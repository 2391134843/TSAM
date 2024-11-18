
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F




class CLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.svc_neg_num = 16 
        
        self.svc_temperature=0.02
        
     
    def structure_visual_contrastive(self, str_embdding, vis_embdding):
          
        
          str_embdding = torch.nn.functional.normalize(str_embdding, p=2, dim=-1, eps=1e-5)
          vis_embdding = torch.nn.functional.normalize(vis_embdding, p=2, dim=-1, eps=1e-5)
          bs, _ = str_embdding.size()
          neg_sample_id = torch.randint(0, bs, [bs, self.svc_neg_num])  
          neg_str_feat = str_embdding[neg_sample_id]  
          neg_vis_feat = vis_embdding[neg_sample_id]  
          str_samples = torch.cat([str_embdding.unsqueeze(1), neg_str_feat], 1)  
          vis_samples = torch.cat([vis_embdding.unsqueeze(1), neg_vis_feat], 1)  
          s2v_score = torch.matmul(vis_samples, str_embdding.unsqueeze(2)).squeeze(2) / self.svc_temperature  
          v2s_score = torch.matmul(str_samples, vis_embdding.unsqueeze(2)).squeeze(2) / self.svc_temperature  
          label = torch.zeros([bs, ], dtype=torch.long).to(str_embdding.device)
          s2v_loss = torch.nn.functional.cross_entropy(s2v_score, label)
          v2s_loss = torch.nn.functional.cross_entropy(v2s_score, label)
          svc_loss = 0.5 * (s2v_loss + v2s_loss)

          return svc_loss


    def forward(self, str_embdding, vis_embdding,txt_embdding):
        str_embdding, vis_embdding,txt_embdding = str_embdding, vis_embdding,txt_embdding
        loss1=self.structure_visual_contrastive(str_embdding, vis_embdding)
        loss2=self.structure_visual_contrastive(str_embdding, txt_embdding)
        Closs=loss1+loss2
        return Closs
     
     


    


