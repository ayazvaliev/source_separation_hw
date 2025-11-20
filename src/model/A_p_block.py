from src.model.moduls import GLN
import torch
from torch import nn
from src.model.moduls import TransformerLayer
from torch.nn.functional import interpolate


class AP_block(nn.Module):

    def __init__(self):
        super().__init__()
        C_a = 512
        comp_coef = 4
        

        #RTFS
        #Comp
        self.compression_TF_1 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.compression_TF_2 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        

        #Attentn
        self.Attentn = TransformerLayer(input_channel = C_a,
        time_dim=50//4,
        mhsa_nhead=8, 
        mhsa_heads_dim=50//4, 
        mhsa_dropout=0.1,
        conv_kernel_size=5,
        conv_stride=1,
        conv_dilation=1,
        conv_padding=2)
        

        

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
        A_1 = self.compression_TF_1(x)
        A_2 = self.compression_TF_2(A_1)
        target_sise = A_2.shape[:-2]
        x_comp = self.adaptive_pool(x, target_sise)
        x_1 = self.adaptive_pool(A_1, target_sise)
        A_g = torch.sum(x_comp,x_1, A_2)


        

        #Att

        att_fin = self.Attentn(A_g)

        #Next phase

        V_g_hash = att_fin + A_g
        
        #Upscale
        A_0_hatch = self.I(x, V_g_hash)
        A_1_hatch = self.I(A_1, V_g_hash)
        

        A_0_hatch_hatch = self.I(A_0_hatch, A_1_hatch) + x

        return self.conv_ups(A_0_hatch_hatch)


    def forward(self, x):
        return self.RTFS(x)


    
    def I(self, m, n):
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
