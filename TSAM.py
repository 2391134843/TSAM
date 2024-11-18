import torch
import torch.nn as nn
import numpy as np
from Tucker import *
from CLossModel import *
from TransELayer import *
from Rotate import *

class TSAM(nn.Module):
    def __init__(
            self, 
            num_ent, 
            num_rel, 
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout = 0.1,
            emb_dropout = 0.6, 
            vis_dropout = 0.1, 
            txt_dropout = 0.1,
            visual_token_index = None, 
            text_token_index = None,
            score_function = "tucker"
        ):
        super(TSAM, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.data_type=torch.float32
        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/textual.pth")
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.score_function = score_function
        self.scale = torch.Tensor([1. / np.sqrt(self.dim_str)]).cuda()
        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        false_ents = torch.full((self.num_ent,1),False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim = 1)
        
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel,1),False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim = 1)
        
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))
        
        self.proj_ent_vis = nn.Linear(32, dim_str)
        self.proj_ent_txt = nn.Linear(768, dim_str)

        ######
        self.context_vec = nn.Parameter(torch.randn((1, dim_str)))
        
        self.act = nn.Softmax(dim=1)
        

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
         
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)
        
        

        
        self.num_con = 256
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        elif self.score_function == "transe":
            self.transE_decoder = TransELayer()
        elif self.score_function == "rotate":
            self.rotate_decoder = RotatELayer(dim_str)
        else:
            pass
        
        self.init_weights()
        torch.save(self.visual_token_embedding, open("visual_token.pth", "wb"))
        torch.save(self.text_token_embedding, open("textual_token.pth", "wb"))
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self):
        
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        
        ent_tkn2 = ent_tkn.squeeze(1)
        ent_seq1 = torch.cat([ent_tkn, rep_ent_str, ], dim = 1)
        ent_seq2 = torch.cat([ent_tkn, rep_ent_vis, ], dim = 1)
        ent_seq3 = torch.cat([ent_tkn,  rep_ent_txt], dim = 1)
        str_embdding = self.ent_encoder(ent_seq1)[:,0]
        vis_embdding = self.ent_encoder(ent_seq2)[:,0]
        txt_embdding = self.ent_encoder(ent_seq3)[:,0]
        clmodel = CLoss()
        closs = clmodel(str_embdding, vis_embdding, txt_embdding)
        # str_embdding = self.ent_encoder(rep_ent_str)[:,0]
        # vis_embdding = self.ent_encoder(rep_ent_vis)[:,0]
        # txt_embdding = self.ent_encoder(rep_ent_txt)[:,0]
        cands = torch.stack([ent_tkn2, str_embdding, vis_embdding, txt_embdding], dim=1)  # (1500, 4, 256)
        # x = torch.arange(self.num_ent).cuda()
        context_vec = self.context_vec
        
        att_weights = torch.sum(context_vec * cands* self.scale , dim=-1, keepdim=True)
        att_weights = self.act(att_weights)  
        ent_embs = torch.sum(att_weights * cands, dim=1)  


        # ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        # ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = self.ent_mask)[:,0]
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))
        return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1), closs


   

    def score(self, emb_ent, emb_rel, triplets):
        
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]     
        
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)        
            scores = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
           
        elif self.score_function == "transe":
            trans_embedding=self.transE_decoder(ctx_emb, rel_emb)
            
            scores = torch.mm(trans_embedding, emb_ent[:-1].transpose(1, 0))
        elif self.score_function == "rotate":
            rotae_emb = self.rotate_decoder(output_dec, rel_emb)
            scores = torch.mm(rotae_emb, emb_ent[:-1].transpose(1, 0))
            

        return scores
