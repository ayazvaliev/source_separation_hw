import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size,
        maxlen
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size).reshape(-1, 1)
        pos = torch.arange(0, maxlen).reshape(1, maxlen)
        pos_embedding = torch.zeros((emb_size, maxlen))
        pos_embedding[0::2, :] = torch.sin(pos * den)
        pos_embedding[1::2, :] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embed: torch.Tensor):
        return embed + self.pos_embedding


class DWConv(nn.Module):
    def __init__(self, 
                 kernel_size, 
                 padding,
                 input_channel,
                 time_dim,
                 stride=1,
                 dilation=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(in_channels=input_channel,
                                 out_channels=input_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=input_channel)
        self.gl_norm = nn.LayerNorm(normalized_shape=(input_channel, time_dim))
        self.act = nn.PReLU()
    
    def forward(self, x: torch.Tensor):
        return self.act(self.gl_norm(self.dw_conv(x)))


class FFN(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 input_channel,
                 time_dim):
        super().__init__()
        self.channel_expand = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=2*input_channel,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU()
        )
        self.gln_expand = nn.LayerNorm(normalized_shape=(input_channel, time_dim))
        self.bottleneck = DWConv(
            kernel_size,
            stride,
            dilation,
            padding,
            2*input_channel,
            time_dim
        )
        self.channel_narrow = nn.Sequential(
            nn.Conv1d(
                in_channels=2*input_channel,
                out_channels=input_channel,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU()
        )
        self.gln_narrow = nn.LayerNorm(normalized_shape=(input_channel, time_dim))

    def forward(self, x: torch.Tensor):
        x = self.gln_expand(self.channel_expand(x))
        x = self.bottleneck(x)
        x = self.gln_narrow(self.channel_narrow(x))
        return x


class MHSA(nn.Module):
    def __init__(self, 
                 embed_dim,
                 heads_dim,
                 nhead,
                 dropout=0.0):
        super().__init__()
        self.qkv_proj = nn.Linear(in_features=embed_dim,
                                    out_features=3*heads_dim*nhead)
        self.nhead = nhead
        self.heads_dim = heads_dim
        self.dropout=dropout
    
    def forward(self, x: torch.Tensor):
        # x (B, N, T)
        print('before mhsa: ', x)
        B, N, T = x.size()
        x = x.transpose(1, 2)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.nhead, self.heads_dim)
        q, k, v = torch.unbind(qkv, dim=2)

        # q, k, v (B, nhead, T, heads_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False
        )

        attn = attn.transpose(1,2).contiguous() # (B, T, nhead, heads_dim)
        print('after mhsa', attn.view(B, T, N).transpose(1, 2))
        return x + attn.view(B, T, N).transpose(1, 2)


class TransformerLayer(nn.Module):
    def __init__(self,
                 input_channel,
                 time_dim,
                 mhsa_nhead,
                 mhsa_heads_dim,
                 mhsa_dropout,
                 conv_kernel_size,
                 conv_stride,
                 conv_dilation,
                 conv_padding
    ):
        super().__init__()
        print('tranformer_layer time_dim', time_dim)
        self.pos_encoder = PositionalEncoding(input_channel, time_dim)
        self.mhsa = MHSA(embed_dim=input_channel, 
                         heads_dim=mhsa_heads_dim, 
                         nhead=mhsa_nhead, 
                         dropout=mhsa_dropout
                         )
        self.ffn = FFN(
            input_channel=input_channel,
            time_dim=time_dim,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            dilation=conv_dilation,
            padding=conv_padding
        )
    
    def forward(self, x: torch.Tensor):
        x = x + self.mhsa(self.pos_encoder(x))
        return x + self.ffn(x)


class Encoder(nn.Module):
    def __init__(self,
                 mixture_dim,
                 time_dim,
                 downsample_num_layers,
                 downsample_rate,
                 use_ga=True
                 ):
        super().__init__()
        time_dims = [time_dim]
        for _ in range(downsample_num_layers):
            time_dims.append(time_dims[-1] // downsample_rate)

        conv_stride = downsample_rate
        conv_dilation = 2
        conv_kernel_size = 5 * downsample_rate // 2
        conv_padding = downsample_rate * 2
        self.convs = nn.ModuleList(
            [DWConv(kernel_size=conv_kernel_size, 
                    stride=conv_stride,
                    padding=conv_padding,
                    dilation=conv_dilation,
                    input_channel=mixture_dim,
                    time_dim=time_dim) for time_dim in time_dims[1:]]
        )
        self.use_ga = use_ga

        if self.use_ga:
            self.avg_pools = nn.ModuleList(
                [nn.AvgPool1d(kernel_size=downsample_rate**i) for i in range(downsample_num_layers, 0, -1)]
            )
    
    def forward(self, x: torch.Tensor):
        residuals = [x]
        for conv_layer in self.convs:
            x = conv_layer(x)
            residuals.append(x)
        if self.use_ga:
            for i, pool_layer in enumerate(self.avg_pools):
                x += pool_layer(residuals[i])
            return residuals, x
        return residuals
    

class GlobalAttention(nn.Module):
    def __init__(self, 
                 mixture_dim,
                 time_dim,
                 nhead,
                 heads_dim,
                 dropout,
                 kernel_size,
                 upsample_num_layers,
                 upsample_rate):
        super().__init__()
        self.transformer = TransformerLayer(
            mixture_dim,
            time_dim,
            nhead,
            heads_dim,
            dropout,
            kernel_size,
            conv_stride=1,
            conv_dilation=1,
            conv_padding=kernel_size//2
        )

        self.upsample_blocks = nn.ModuleList(
            [nn.Upsample(size=time_dim * upsample_rate**(i+1)) for i in range(upsample_num_layers)]
        )

    
    def forward(self, residuals: list[torch.Tensor], attn: torch.Tensor):
        new_residuals = [F.sigmoid(residuals[-1]) * attn]
        for residual, upsample_block in zip(residuals[:-1][::-1], self.upsample_blocks):
            attn = upsample_block(attn)
            new_residuals.append(F.sigmoid(attn) * residual)
        return new_residuals


class Decoder(nn.Module):
    def __init__(self, 
                 mixture_dim,
                 time_dim,
                 kernel_size, 
                 upsample_rate, 
                 upsample_num_layers,
                 use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.convs = nn.ModuleList(
            [
                DWConv(input_channel=mixture_dim, kernel_size=kernel_size, padding=kernel_size // 2, time_dim=time_dim * upsample_rate**(i + 1))
                for i in range(upsample_num_layers)
            ]
        )
        if self.use_attn:
            self.attn_convs = nn.ModuleList(
                [
                    DWConv(input_channel=mixture_dim, kernel_size=kernel_size, padding=kernel_size // 2, time_dim=time_dim * upsample_rate**(i + 1))
                    for i in range(upsample_num_layers)
                ]
            )
        self.upsample_blocks = nn.ModuleList(
            [nn.Upsample(size=time_dim * upsample_rate**(i+1)) for i in range(upsample_num_layers)]
        )
    
    def forward(self, residuals: list[torch.Tensor]):
        x = residuals[0]
        for i in range(len(residuals) - 1):
            x = self.upsample_blocks[i](x)
            if self.use_attn:
                x = F.sigmoid(self.attn_convs[i](x)) * residuals[i+1] + self.convs[i](x)
            else:
                x = residuals[i+1] + self.convs[i](x)
        return x


class TDANet(nn.Module):
    def __init__(self,
                 num_frames,
                 mixture_dim,
                 num_speakers,
                 stem_kernel_size,
                 stem_padding,
                 decoder_kernel_size,
                 num_layers,
                 rate,
                 use_la,
                 use_ga,
                 num_blocks,
                 **kwargs
                 ):
        super().__init__()

        self.use_ga = use_ga
        self.num_blocks = num_blocks
        self.num_speakers = num_speakers
        self.mixture_dim = mixture_dim

        self.encoder_stem =  nn.Conv1d(in_channels=1,
                      out_channels=mixture_dim,
                      kernel_size=stem_kernel_size,
                      stride=stem_kernel_size//4,
                      padding=stem_padding
                      )

        time_dim = (num_frames + 2 * stem_padding - stem_kernel_size) // (stem_kernel_size // 4) + 1
        last_time_dim = time_dim // (rate**(num_layers))

        self.encoder = Encoder(mixture_dim, 
                               time_dim, 
                               downsample_num_layers=num_layers, 
                               downsample_rate=rate,
                               use_ga=self.use_ga)

        if self.use_ga:
            self.ga_block = GlobalAttention(mixture_dim=mixture_dim,
                                            time_dim=last_time_dim,
                                            nhead=kwargs['ga_nhead'],
                                            heads_dim=kwargs['ga_heads_dim'],
                                            dropout=kwargs['ga_dropout'],
                                            kernel_size=kwargs['ga_kernel_size'],
                                            upsample_num_layers=num_layers,
                                            upsample_rate=rate)
        
        self.decoder = Decoder(mixture_dim=mixture_dim,
                               time_dim=last_time_dim,
                               kernel_size=decoder_kernel_size,
                               upsample_rate=rate,
                               upsample_num_layers=num_layers,
                               use_attn=use_la)

        self.mask_gen = nn.Sequential(
            nn.Conv1d(in_channels=mixture_dim, 
                      out_channels=num_speakers*mixture_dim,
                      kernel_size=1,
                      stride=1),
            nn.ReLU()
        )
        self.reconstruction_conv = nn.ConvTranspose1d(in_channels=mixture_dim * num_speakers, 
                                                     out_channels=num_speakers, 
                                                     kernel_size=stem_kernel_size,
                                                     stride=stem_kernel_size // 4,
                                                     padding=stem_padding,
                                                     groups=num_speakers)
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        r = self.encoder_stem(x)
        x = torch.zeros_like(r)
        for i in range(self.num_blocks):
            encoder_out = self.encoder(x + r)
            if self.use_ga:
                residuals = self.ga_block(*encoder_out)[::-1]
            else:
                residuals = encoder_out
            x = self.decoder(residuals[::-1])

        applied_masks = (self.mask_gen(x).view(batch_size, self.mixture_dim, self.num_speakers, -1) * r.unsqueeze(2)).view(batch_size, self.mixture_dim * self.num_speakers, -1)
        return {"logits": applied_masks}