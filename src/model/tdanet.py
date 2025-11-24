import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.moduls import GLN
from src.model.moduls import DWConv, Encoder, TransformerLayer


class LADecoder(nn.Module):
    def __init__(self, 
                 mixture_dim,
                 time_dim,
                 kernel_size, 
                 upsample_rate, 
                 upsample_num_layers,
                 use_attn=True,
                 loc_same_size_as_glob=False,
                 collect_residuals=False):
        super().__init__()
        self.collect_residuals = collect_residuals
        self.use_attn = use_attn
        self.convs = nn.ModuleList(
            [
                DWConv(input_channel=mixture_dim, 
                       kernel_size=kernel_size, 
                       padding=kernel_size // 2,
                       use_act=False)
                for _ in range(upsample_num_layers + int(loc_same_size_as_glob))
            ]
        )
        if self.use_attn:
            self.attn_convs = nn.ModuleList(
                [
                    DWConv(input_channel=mixture_dim, 
                           kernel_size=kernel_size, 
                           padding=kernel_size // 2,
                           use_act=False)
                    for _ in range(upsample_num_layers + int(loc_same_size_as_glob))
                ]
            )
        self.upsample_blocks = nn.ModuleList(
            [nn.Upsample(size=time_dim * upsample_rate**(i + (1 - int(loc_same_size_as_glob)))) for i in range(upsample_num_layers + int(loc_same_size_as_glob))]
        )
    
    def forward(self, residuals: list[torch.Tensor]):
        x = residuals[0]
        new_residuals = []
        for i in range(len(residuals) - 1):
            x = self.upsample_blocks[i](x)
            if self.use_attn:
                x = F.sigmoid(self.attn_convs[i](x)) * residuals[i+1] + self.convs[i](x)
            else:
                x = residuals[i+1] + self.convs[i](x)
            if self.collect_residuals:
                new_residuals.append(x)

        if self.collect_residuals:
            return new_residuals
        return x


class GlobalAttention(nn.Module):
    def __init__(self, 
                 mixture_dim,
                 time_dim,
                 kernel_size,
                 upsample_num_layers,
                 upsample_rate,
                 use_transformer=True,
                 ffn_only=False,
                 nhead=None,
                 dropout=None):
        super().__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerLayer(
                mixture_dim,
                time_dim,
                nhead,
                dropout,
                kernel_size,
                conv_stride=1,
                conv_dilation=1,
                conv_padding=kernel_size//2,
                ffn_only=ffn_only
            )
        self.la = LADecoder(
            mixture_dim=mixture_dim,
            time_dim=time_dim,
            kernel_size=kernel_size,
            upsample_rate=upsample_rate,
            upsample_num_layers=upsample_num_layers,
            use_attn=True,
            loc_same_size_as_glob=True,
            collect_residuals=True
        )


    def forward(self, residuals: list[torch.Tensor], attn: torch.Tensor):
        attn = self.transformer(attn) if self.use_transformer else attn
        return self.la([attn] + residuals[::-1])
    
        '''
        new_residuals = [F.sigmoid(residuals[-1]) * attn]
        for residual, upsample_block in zip(residuals[:-1][::-1], self.upsample_blocks):
            attn = upsample_block(attn)
            new_residuals.append(F.sigmoid(attn) * residual)
        return new_residuals
        '''


class TDANet(nn.Module):
    def __init__(self,
                 num_frames,
                 init_mixture_dim,
                 mixture_dim,
                 num_speakers,
                 stem_kernel_size,
                 stem_padding,
                 decoder_kernel_size,
                 ga_kernel_size,
                 num_layers,
                 rate,
                 use_la,
                 use_ga,
                 num_blocks,
                 return_dict=True,
                 **kwargs
                 ):
        super().__init__()

        self.return_dict = return_dict

        self.num_blocks = num_blocks
        self.num_speakers = num_speakers
        self.latent_dim = stem_kernel_size // 2 + 1
        self.encoder_stem =  nn.Conv1d(in_channels=1,
                      out_channels=self.latent_dim,
                      kernel_size=stem_kernel_size,
                      stride=stem_kernel_size//4,
                      padding=stem_padding,
                      bias=False
                      )
        torch.nn.init.xavier_uniform_(self.encoder_stem.weight)

        time_dim = (num_frames + 2 * stem_padding - stem_kernel_size) // (stem_kernel_size // 4) + 1
        last_time_dim = time_dim // (rate**(num_layers))

        self.ln = GLN(self.latent_dim)

        self.bottleneck = nn.Conv1d(
            in_channels=self.latent_dim, 
            out_channels=init_mixture_dim, 
            kernel_size=1
        )

        self.proj_conv = nn.Sequential(
            nn.Conv1d(in_channels=init_mixture_dim,
                      out_channels=mixture_dim,
                      kernel_size=1,
                      groups=1,
                      ),
            GLN(mixture_dim),
            nn.PReLU()
        )

        self.encoder = Encoder(mixture_dim, 
                               downsample_num_layers=num_layers, 
                               downsample_rate=rate)

        self.ga_block = GlobalAttention(mixture_dim=mixture_dim,
                                        time_dim=last_time_dim,
                                        nhead=kwargs['mhsa_nhead'] if use_ga else None,
                                        dropout=kwargs['mhsa_dropout'] if use_ga else None,
                                        ffn_only=kwargs['ga_ffn_only'],
                                        kernel_size=ga_kernel_size,
                                        upsample_num_layers=num_layers,
                                        upsample_rate=rate,
                                        use_transformer=use_ga)
        
        self.decoder = LADecoder(mixture_dim=mixture_dim,
                               kernel_size=decoder_kernel_size,
                               time_dim=last_time_dim,
                               upsample_rate=rate,
                               upsample_num_layers=num_layers,
                               use_attn=use_la)

        self.inverse_proj = nn.Conv1d(mixture_dim, 
                                      init_mixture_dim, 
                                      kernel_size=1)

        self.mask_gen = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels=init_mixture_dim, 
                      out_channels=num_speakers*self.latent_dim,
                      kernel_size=1,
                      stride=1),
            nn.ReLU()
        )
        self.reconstruction_conv = nn.ConvTranspose1d(in_channels=self.latent_dim * num_speakers, 
                                                     out_channels=num_speakers, 
                                                     kernel_size=stem_kernel_size,
                                                     stride=stem_kernel_size // 4,
                                                     padding=stem_padding,
                                                     bias=False)
        torch.nn.init.xavier_uniform_(self.reconstruction_conv.weight)

        self.concat_block = nn.Sequential(
            nn.Conv1d(in_channels=init_mixture_dim,
                      out_channels=init_mixture_dim,
                      kernel_size=1,
                      stride=1,
                      groups=init_mixture_dim),
            nn.PReLU()
        )
        self.concat_block = DWConv(
            input_channel=init_mixture_dim,
            kernel_size=1,
            padding=0,
            use_norm=False
        )
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        r = self.encoder_stem(x)
        encoded_audio = r.clone()

        r = self.ln(r)
        r = self.bottleneck(r)
        x = torch.zeros_like(r)
        for i in range(self.num_blocks):
            x = self.concat_block(x + r)
            x_res = x.clone()

            x = self.proj_conv(x)
            encoder_out = self.encoder(x)
            residuals = self.ga_block(*encoder_out)
            x = self.decoder(residuals)
            x = x_res + self.inverse_proj(x)

        applied_masks = (self.mask_gen(x).view(batch_size, self.latent_dim, self.num_speakers, -1) * encoded_audio.unsqueeze(2)).view(batch_size, self.latent_dim * self.num_speakers, -1)
        logits = self.reconstruction_conv(applied_masks)
        return {"logits": self.reconstruction_conv(applied_masks)} if self.return_dict else logits
    

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