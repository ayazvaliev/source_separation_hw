from torch import nn
from src.model.moduls import TransformerLayer
from torch.nn.functional import interpolate


class AP_block(nn.Module):
    """
    Simple MLP
    """

    def __init__(self):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        C_a = 512
        comp_coef = 4
        

        #RTFS
        #Comp
        self.compression_TF_1 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.compression_TF_2 = nn.Conv1d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(14)
        

        #Attentn
        self.Attentn = TransformerLayer(
                 input_channel = 512,
                 time_dim = 14,
                 mhsa_nhead = 8,
                 mhsa_dropout = 0.1,
                 conv_kernel_size = 5,
                 conv_stride = 1,
                 conv_dilation = 1,
                 conv_padding = 2,
                 ffn_only = False)


        

        

        #Upscaling
        self.W1 = nn.Conv1d(C_a, C_a, kernel_size=5, padding=2, 
                     groups=C_a)
        
        self.W2 = nn.Conv1d(C_a, C_a, kernel_size=5, padding=2, 
                     groups=C_a)
        
        self.W3 = nn.Conv1d(C_a, C_a, kernel_size=5, padding=2, 
                     groups=C_a)
        
        self.sigmoid_m = nn.Sigmoid()



    
    def RTFS(self,x):
        #Downscaling
        A_1 = self.compression_TF_1(x)
        A_2 = self.compression_TF_2(A_1)
        x_comp = self.adaptive_pool(x)
        x_1 = self.adaptive_pool(A_1)
        A_g = x_comp + x_1 + A_2


        

        #Att

        att_fin = self.Attentn(A_g)

        #Next phase

        V_g_hatch = att_fin
        
        #Upscale
        A_0_hatch = self.I(x, V_g_hatch)
        A_1_hatch = self.I(A_1, V_g_hatch)
    

        A_0_hatch_hatch = self.I(A_0_hatch, A_1_hatch) + x

        return (A_0_hatch_hatch)


    def forward(self, x):
        return self.RTFS(x)


    
    def I(self, m, n):
        interpol_shapes = m.shape[-1]

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
