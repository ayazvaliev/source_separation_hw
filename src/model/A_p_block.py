from src.model.moduls import GLN
import torch
from torch import nn
from torch.nn.functional import softmax, interpolate
from sru import SRU


class AP_block(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        C_a = 512
        

        #RTFS
        #Comp
        self.compression_C = nn.Conv1d(in_channels=C_a, out_channels=C_a//comp_coef, kernel_size=1)
        self.compression_TF_1 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.compression_TF_2 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        

        #Attentn
        self.n_feats = n_feats

        self.pos_embeddings = nn.Embedding(n_feats, 50//4)

        self.multi_head = nn.MultiheadAttention(
        embed_dim=50//4,        
        num_heads=8,        
        dropout=0.1,        
        batch_first=True      
        )
        
        self.gln = GLN(n_feats)
        self.conv_mh = nn.Conv1d(n_feats, n_feats*2, kernel_size=1)
        self.gln_2 = GLN(n_feats)
        self.conv_mh_2 = nn.Conv1d(n_feats*2, n_feats*2, kernel_size = 5 , groups=n_feats*2, padding=2)
        self.gln_3 = GLN(n_feats)
        self.conv_mh_3 = nn.Conv1d(n_feats*2, n_feats, kernel_size=1)

        

        #Upscaling
        self.W1 = nn.Conv1d(C_a//comp_coef, C_a//comp_coef, kernel_size=4, padding=1, 
                     groups=C_a//comp_coef)
        
        self.W2 = nn.Conv1d(C_a//comp_coef, C_a//comp_coef, kernel_size=4, padding=1, 
                     groups=C_a//comp_coef)
        
        self.W3 = nn.Conv1d(C_a//comp_coef, C_a//comp_coef, kernel_size=4, padding=1, 
                     groups=C_a//comp_coef)
        
        self.sigmoid_m = nn.Sigmoid()

        self.conv_ups = nn.Conv1d(C_a//comp_coef, C_a, kernel_size=4, padding=1)
    


    
    def RTFS(self,x):
        #Downscaling
        A_1 = self.compression_TF_1(A_0)
        A_2 = self.compression_TF_2(A_1)
        target_sise = x_2.shape[:-2]
        x_comp = self.adaptive_pool(A_0, target_sise)
        x_1 = self.adaptive_pool(A_1, target_sise)
        A_g = torch.sum(x_comp,x_1,x_2)

        #Att

        pos_emb = self.pos_embeddings(self.n_feats)
        B = A_g.shape[0]
        pos_emb = pos_emb.unsqueese(0).repeat(B, 1, 1)

        res = A_g + pos_emb

        mhsa_res = self.multi_head(res)
        mhsa_res = self.multi_head(mhsa_res)

        mhsa_fin = res+ mhsa_res
        res = mhsa_fin.copy() 

        msa_fin = self.gln(self.conv_mh(msa_fin))
        msa_fin = self.gln_2(self.conv_mh_2(msa_fin))
        msa_fin = self.gln_3(self.conv_mh_3(msa_fin))

        att_fin = mhsa_fin + res

        #Next phase

        V_g_hash = att_fin + A_g
        
        #Upscale
        A_0_hatch = self.I(x, A_g_hatch)
        A_1_hatch = self.I(A_1, A_g_hatch)
        

        A_0_hatch_hatch = self.I(A_0_hatch, A_1_hatch) + x

        return self.conv_ups(A_0_hatch_hatch)





    
    def I(m, n):
        interpol_shapes = m.shape[-2:]

        result = (interpolate(self.sigmoid_m(self.W1(n)), interpol_shapes) * self.W2(m) + 
        interpolate(self.W3(n), interpol_shapes))

        return result




    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
