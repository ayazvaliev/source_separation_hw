from torch import nn
import torch.nn.functional as F
import torch
import math

EPS = 1e-8


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size,
        maxlen
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size).reshape(-1, 1)
        pos = torch.arange(0, maxlen).reshape(1, maxlen)
        pos_embedding = torch.zeros((emb_size, maxlen))
        pos_embedding[0::2, :] = torch.sin(pos * den)
        pos_embedding[1::2, :] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embed: torch.Tensor):
        return embed + self.pos_embedding


class GLN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(input_channel), requires_grad=True)
        self.var = nn.Parameter(torch.mean(input_channel), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        # x [B, F, T]
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        x_normalized = (x - mean) / (var + EPS).sqrt()
        return (self.var * x_normalized.transpose(1, 2) + self.mean).transpose(1, 2)


class DWConv(nn.Module):
    def __init__(self, 
                 kernel_size, 
                 padding,
                 input_channel,
                 stride=1,
                 dilation=1,
                 use_act=True,
                 use_norm=True,
                 output_channel=None,
                 bias=None):
        super().__init__()
        if bias is None:
            bias = not use_norm

        if output_channel is None:
            output_channel = input_channel

        self.dw_conv = nn.Conv1d(in_channels=input_channel,
                                 out_channels=output_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=input_channel,
                                 bias=bias)
        if use_norm:
            self.gl_norm = GLN(input_channel)
        else:
            self.gl_norm = nn.Identity()

        if use_act:
            self.act = nn.PReLU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor):
        return self.act(self.gl_norm(self.dw_conv(x)))


class FFN(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 input_channel,
                 dropout):
        super().__init__()

        self.channel_expand = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=2*input_channel,
                kernel_size=1,
                bias=False
            ),
            GLN(2*input_channel)
        )
        self.bottleneck = nn.Sequential(
            DWConv(
                kernel_size=kernel_size,
                padding=padding,
                input_channel=2*input_channel,
                stride=stride,
                dilation=dilation,
                use_act=True
            ),
            nn.Dropout(dropout)
        )
        self.channel_narrow = nn.Sequential(
            nn.Conv1d(
                in_channels=2*input_channel,
                out_channels=input_channel,
                kernel_size=1,
                bias=False
            ),
            GLN(input_channel)
        )

    def forward(self, x: torch.Tensor):
        x = self.channel_expand(x)
        x = self.bottleneck(x)
        x = self.channel_narrow(x)
        return x


class MHSA(nn.Module):
    def __init__(self, 
                 embed_dim,
                 nhead,
                 dropout=0.0):
        super().__init__()
        assert embed_dim % nhead == 0
        heads_dim = embed_dim // nhead
        self.qkv_proj = nn.Linear(in_features=embed_dim,
                                    out_features=3*heads_dim*nhead)
        self.nhead = nhead
        self.heads_dim = heads_dim
        self.dropout=dropout
        self.gln = GLN(heads_dim * nhead)
        self.cross_head_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        # x (B, N, T)
        B, N, T = x.size()
        qkv = self.qkv_proj(x.transpose(1, 2))
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
        return self.gln(x + self.cross_head_linear(attn.view(B, T, -1)).transpose(1, 2)) # (B, nhead * heads_dim, T)


class TransformerLayer(nn.Module):
    def __init__(self,
                 input_channel,
                 time_dim,
                 mhsa_nhead,
                 mhsa_dropout,
                 conv_kernel_size,
                 conv_stride,
                 conv_dilation,
                 conv_padding
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_channel, time_dim)
        self.mhsa = MHSA(embed_dim=input_channel, 
                         nhead=mhsa_nhead, 
                         dropout=mhsa_dropout
                        )
        self.ffn = FFN(
            input_channel=input_channel,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            dilation=conv_dilation,
            padding=conv_padding,
            dropout=mhsa_dropout
        )
    
    def forward(self, x: torch.Tensor):
        x = x + self.mhsa(self.pos_encoder(x))
        return x + self.ffn(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_channel,
                 downsample_num_layers,
                 downsample_rate,
                 ):
        super().__init__()
        conv_stride = downsample_rate
        conv_kernel_size = 2*downsample_rate + 1
        conv_padding = conv_kernel_size // 2

        self.init_conv = DWConv(input_channel=input_channel,
                                 kernel_size=conv_kernel_size, 
                                 stride=1, 
                                 padding=conv_kernel_size//2,
                                 use_act=False)

        self.convs = nn.ModuleList(
            [DWConv(kernel_size=conv_kernel_size, 
                    stride=conv_stride,
                    padding=conv_padding,
                    dilation=1,
                    input_channel=input_channel,
                    use_act=False) for _ in range(downsample_num_layers)]
        )

        self.avg_pools = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=downsample_rate**i) for i in range(downsample_num_layers, 0, -1)]
        )
    
    def forward(self, x: torch.Tensor):
        residuals = [self.init_conv(x)]
        for conv_layer in self.convs:
            x = conv_layer(x)
            residuals.append(x)
        for i, pool_layer in enumerate(self.avg_pools):
            x += pool_layer(residuals[i])
        return residuals, x