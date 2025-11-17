from src.model.moduls import GLN
import torch
from torch import nn
from torch.nn.functional import softmax, interpolate
from sru import SRU

class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_class, sru_hidden = 256, C_a =512, comp_coef = 4, fc_hidden=20, h = 20, kernel_audio = 5, kernel_video = 3):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        #encoder
        self.audio_encoder = nn.Conv2d(in_channels=2, out_channels=C_a, kernel_size=3, padding=1)
        self.visual_preprocessing_block = nn.SomeBlock

        #CAF
        self.h = h
        self.C_a = C_a
        self.conv_for_video_1 = nn.Conv1d(in_channels = 512, out_channels=h*C_a, padding= kernel_video//2, kernel_size=kernel_video, groups=C_a)
        self.conv_for_video_2 = nn.Conv1d(in_channels = 512, out_channels=C_a, kernel_size=1, groups=C_a)
        self.glob_layer_norm_1 = GLN(h*C_a)
        self.glob_layer_norm_2 = GLN(C_a)
        self.conv_for_audio_1 = nn.Conv2d(in_channels=n_feats, out_channels=c_a, kernel_size=1, groups = C_a)
        self.conv_for_audio_2 = nn.Conv2d(in_channels=n_feats, out_channels=c_a, kernel_size=1, groups = C_a)
        self.glob_layer_norm_3 = GLN(C_a)
        self.glob_layer_norm_4 = GLN(C_a)
        self.relu = nn.ReLU() 

        #RTFS
        self.compression_C = nn.Conv1d(in_channels=C_a, out_channels=C_a//comp_coef, kernel_size=1)
        self.compression_TF_1 = nn.Conv2d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.compression_TF_2 = nn.Conv2d(in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.unfold = nn.Unfold(kernel_size=(1, 8), stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(C_a//comp_coef)
        self.sru_model = sru = SRU(
    
            input_size = 121,           
            hidden_size=sru_hidden,          
    
    
            num_layers=4,            
            bidirectional=True,      
            layer_norm=True) 
        

        
    def forward(self, audio_mix: torch.Tensor, mouth_embeddings: torch.Tensor,  **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        _, audio_time, audio_feats = audio_mix.shape
        #preprocessing block
        audio = self.audio_encoder(audio_mix) 
        video = self.visual_preprocessing_block(mouth_embeddings)

        #CAF block

        audio_val = self.conv_for_audio_1(audio)
        audio_val = self.glob_layer_norm_3(audio_val)

        audio_gate = self.conv_for_audio_2(audio)
        audio_gate = self.glob_layer_norm_4(audio_gate)
        audio_gate = self.relu(audio_gate)

        b,T,F = video.shape
        video_1 = self.conv_for_video_1(video)
        video_1 = self.glob_layer_norm_1(video_1)
        video_1 = video_1.view(b, self.h, self.C_a, F)
        video_1 = torch.mean(video_1, dim = 1)
        video_1 = softmax(video_1, dim = -1)
        video_1 = interpolate(video_1, mode="nearest", size = audio_time)
        video_1 = torch.einsum("bctf,bct->bctf", audio_val, video_1)

        video_2 = self.conv_for_video_2(video)
        video_2 = self.glob_layer_norm_2(video)
        video_2 = interpolate(video_2, mode="nearest", size = audio_time)
        video_2 = torch.einsum("bctf,bct->bctf", audio, video_2)

        caf_result = video_1 + video_2









        



        return {"logits": x}
    
    def RTFS(self,x):

        x_comp = self.compression_C(x)
        x_1 = self.compression_TF_1(x)
        x_2 = self.compression_TF_2(x_1)
        target_sise = x_2.shape[:-2]
        x_comp = self.adaptive_pool(x_comp, target_sise)
        x_1 = self.adaptive_pool(x_1, target_sise)
        A_g = torch.sum(x_comp,x_1,x_2)
        B, C_a, T, F = A_g.shape

         
    
        # Reshape чтобы анфолд можно было по частотному измерению применять
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(-1, C_a, 1, F)

        unfolded = self.unfold(x_reshaped)
    
        # Reshape обратно
        freq_patches = unfolded.shape[-1]  
        output = unfolded.view(B, T, C_a * 8, freq_patches)
        output = output.permute(0, 2, 1, 3)

        x = self.layer_norm(output)

        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(-1, C_a, 1, F)



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
