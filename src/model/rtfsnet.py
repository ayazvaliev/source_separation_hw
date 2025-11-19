from src.model.moduls import GLN
import torch
from torch import nn
from torch.nn.functional import softmax, interpolate
from sru import SRU


class RTFSNet(nn.Module):

    def __init__(
        self,
        n_feats,
        n_fft,
        hop_length,
        win_length,
        N=6,
        num_layers=4,
        sru_hidden=256,
        C_a=512,
        comp_coef=4,
        h=20,
        proj_kernel_video=3,
        kernel_video=3,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

        self.register_buffer("window", torch.hann_window(self.n_fft), False)

        self.N = N
        self.C_a = C_a

        # encoder
        self.audio_encoder = nn.Conv2d(in_channels=2, out_channels=C_a, kernel_size=3, padding=1)

        # CAF
        self.h = h
        self.C_a = C_a

        self.video_proj = nn.Sequential(
            GLN(C_a),
            nn.Conv1d(
                in_channels=C_a * 2,
                out_channels=C_a,
                kernel_size=proj_kernel_video,
                padding=proj_kernel_video // 2,
                groups=1,
                bias=False
            ),
            GLN(C_a)
        )

        self.conv_for_video_1 = nn.Conv1d(
            in_channels=C_a,
            out_channels=h * C_a,
            padding=kernel_video // 2,
            kernel_size=kernel_video,
            groups=C_a,
        )
        self.conv_for_video_2 = nn.Conv1d(
            in_channels=C_a, out_channels=C_a, kernel_size=1, groups=C_a
        )
        self.glob_layer_norm_1 = GLN(h * C_a)
        self.glob_layer_norm_2 = GLN(C_a)
        self.conv_for_audio_1 = nn.Conv2d(
            in_channels=n_feats, out_channels=C_a, kernel_size=1, groups=C_a
        )
        self.conv_for_audio_2 = nn.Conv2d(
            in_channels=n_feats, out_channels=C_a, kernel_size=1, groups=C_a
        )
        self.glob_layer_norm_3 = GLN(C_a)
        self.glob_layer_norm_4 = GLN(C_a)
        self.relu = nn.ReLU()

        # RTFS
        # Comp
        self.compression_C = nn.Conv1d(
            in_channels=C_a, out_channels=C_a // comp_coef, kernel_size=1
        )
        self.compression_TF_1 = nn.Conv2d(
            in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2
        )
        self.compression_TF_2 = nn.Conv2d(
            in_channels=C_a, out_channels=C_a, kernel_size=4, stride=2, padding=2
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freq
        self.unfold = nn.Unfold(kernel_size=(1, 8), stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(C_a // comp_coef)
        self.sru_model = SRU(
            input_size=C_a // comp_coef * 8,
            hidden_size=sru_hidden,
            num_layers=4,
            bidirectional=True,
            layer_norm=True,
        )
        self.deconv = nn.ConvTranspose2d(
            in_channels=2 * sru_hidden, out_channels=C_a // comp_coef, kernel_size=8
        )

        # Time
        self.unfold_1 = nn.Unfold(kernel_size=(1, 8), stride=1, padding=0)
        self.layer_norm_1 = nn.LayerNorm(C_a // comp_coef)
        self.sru_model_1 = SRU(
            input_size=C_a // comp_coef,
            hidden_size=sru_hidden,
            num_layers=4,
            bidirectional=True,
            layer_norm=True,
        )
        self.deconv_1 = nn.ConvTranspose2d(
            in_channels=2 * sru_hidden, out_channels=C_a // comp_coef, kernel_size=8
        )

        # Attentn
        self.num_layers = num_layers
        self.conv_Q = nn.Conv2d(C_a // comp_coef, C_a // comp_coef, kernel_size=1)
        self.conv_K = nn.Conv2d(C_a // comp_coef, C_a // comp_coef, kernel_size=1)
        self.conv_V = nn.Conv2d(
            C_a // comp_coef // num_layers, C_a // comp_coef // num_layers, kernel_size=1
        )
        self.conv_out = nn.Conv2d(C_a // comp_coef, C_a // comp_coef, kernel_size=1)

        self.prelu = nn.PReLU()

        self.ln_Q = nn.LayerNorm([n_feats // 4, C_a // comp_coef])
        self.ln_K = nn.LayerNorm([n_feats // 4, C_a // comp_coef])
        self.ln_V = nn.LayerNorm([n_feats // 4, C_a // comp_coef])
        self.ln_out = nn.LayerNorm([n_feats // 4, C_a // comp_coef])

        # Upscaling
        self.W1 = nn.Conv2d(
            C_a // comp_coef, C_a // comp_coef, kernel_size=4, padding=1, groups=C_a // comp_coef
        )

        self.W2 = nn.Conv2d(
            C_a // comp_coef, C_a // comp_coef, kernel_size=4, padding=1, groups=C_a // comp_coef
        )

        self.W3 = nn.Conv2d(
            C_a // comp_coef, C_a // comp_coef, kernel_size=4, padding=1, groups=C_a // comp_coef
        )

        self.sigmoid_m = nn.Sigmoid()

        self.conv_ups = nn.Conv2d(C_a // comp_coef, C_a, kernel_size=4, padding=1)

        # SSS

        self.conv_sss = nn.Conv2d(C_a, C_a, kernel_size=1)

        # decoder

        self.dec_tr_conv = nn.ConvTranspose2d(C_a, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, audio_mix: torch.Tensor, mouth1_emb: torch.Tensor, mouth2_emb: torch.Tensor, **batch):
        # mouth embss [B, T, F]

        # preprocessing block
        stft_spec = torch.stft(audio_mix,
                               n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.win_length,
                               window=self.window)
        audio = self.audio_encoder(stft_spec)
        a_0 = audio

        # AP, VP
        audio = self.RTFS(audio)

        # Конкатим и проектируем в нужную канальность [B, F=C_a, T]
        video = self.video_proj(torch.concat([mouth1_emb, mouth2_emb], dim=-1).transpose(1, 2)) 

        # CAF block
        audio_val = self.conv_for_audio_1(audio)
        audio_val = self.glob_layer_norm_3(audio_val)

        audio_gate = self.conv_for_audio_2(audio)
        audio_gate = self.glob_layer_norm_4(audio_gate)
        audio_gate = self.relu(audio_gate)

        b, F, T = video.shape
        video_1 = self.conv_for_video_1(video)
        video_1 = self.glob_layer_norm_1(video_1)
        video_1 = video_1.view(b, self.h, self.C_a, F)
        video_1 = torch.mean(video_1, dim=1)
        video_1 = softmax(video_1, dim=-1)
        video_1 = interpolate(video_1, mode="nearest", size=audio_time)
        video_1 = torch.einsum("bctf,bct->bctf", audio_val, video_1)

        video_2 = self.conv_for_video_2(video)
        video_2 = self.glob_layer_norm_2(video)
        video_2 = interpolate(video_2, mode="nearest", size=audio_time)
        video_2 = torch.einsum("bctf,bct->bctf", audio, video_2)

        caf_result = video_1 + video_2
        for _ in range(self.N):
            caf_result = self.RTFS(caf_result)

        RTFS_res = caf_result
        m = self.relu(self.conv_sss(self.prelu(RTFS_res)))

        # SSS
        m_r = m[:, : self.C_a // 2, :, :]
        m_i = m[:, self.C_a // 2 :, :, :]

        E_r = a_0[:, : self.C_a // 2, :, :]
        E_i = a_0[:, self.C_a // 2 :, :, :]

        z_r = m_r * E_r - m_i * E_i
        z_i = m_r * E_r + m_i * E_i

        z = torch.cat((z_r, z_i), dim=2)

        return {"logits": self.decoder(z)}

    def decoder(self, x):
        res = self.dec_tr_conv(x)
        real = res[:, 0, :, :]
        imag = res[:, 1, :, :]
        complex_spec = torch.complex(real, imag)

        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return waveform

    def RTFS(self, x):
        A_0 = self.compression_C(x)
        A_1 = self.compression_TF_1(A_0)
        A_2 = self.compression_TF_2(A_1)
        target_sise = A_2.shape[:-2]
        x_comp = self.adaptive_pool(A_0, target_sise)
        x_1 = self.adaptive_pool(A_1, target_sise)
        A_g = torch.sum(x_comp, x_1, A_2)

        # Freq domain
        # Reshape чтобы анфолд можно было по частотному измерению применять
        B, C_a, T, F = A_g.shape
        x_reshaped = A_g.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(-1, C_a, 1, F)

        unfolded = self.unfold(x_reshaped)

        # Reshape обратно
        freq_patches = unfolded.shape[-1]
        output = unfolded.view(B, T, C_a * 8, freq_patches)
        output = output.permute(0, 2, 1, 3).contiguous()

        x = self.layer_norm(output)
        B, C, T, F = x.shape
        x_reshaped = x.permute(3, 0, 2, 1).contiguous()
        x_reshaped = x_reshaped.view(F, B * T, C)
        x_reshaped, _ = self.sru_model(x_reshaped)
        C = x_reshaped.shape[-1]
        x_reshaped = x_reshaped.view(F, B, T, C)
        x = x_reshaped.permute(1, 3, 2, 0).contiguous()
        x_freq = self.deconv(x) + A_g

        # Time domain
        B, C_a, T, F = A_g.shape
        x_reshaped = A_g.permute(0, 3, 1, 2).contiguous()
        x_reshaped = x_reshaped.view(B * F, C_a, T)
        unfolded = self.unfold_1(x_reshaped)
        T = unfolded.shape[-1]
        x_reshaped = unfolded.view(B, F, C_a * 8, T)
        output = x_reshaped.permute(0, 2, 3, 1).contiguous()

        output = self.layer_norm_1(output)

        B, C, T, F = output.shape

        x_reshaped = output.permute(2, 0, 3, 1).contiguous()
        x_reshaped = x_reshaped.view(T, B * F, C)
        x_reshaped, _ = self.sru_model_1(x_reshaped)
        C = x_reshaped.shape[-1]
        x_reshaped = x_reshaped.view(T, B, F, C)
        x = x_reshaped.permute(1, 3, 2, 0)
        x_time = self.deconv_1(x)
        x_time = x_time.permute(0, 1, 3, 2)
        R_t_ddd = x_time + x_freq
        A_g_hatch = self.fn_Self_attention(R_t_ddd) + R_t_ddd

        # Upscale
        A_0_hatch = self.I(A_0, A_g_hatch)
        A_1_hatch = self.I(A_1, A_g_hatch)
        A_0_hatch_hatch = self.I(A_0_hatch, A_1_hatch) + A_0

        return self.conv_ups(A_0_hatch_hatch)

    def I(self, m, n):
        interpol_shapes = m.shape[-2:]

        result = interpolate(self.sigmoid_m(self.W1(n)), interpol_shapes) * self.W2(
            m
        ) + interpolate(self.W3(n), interpol_shapes)

        return result

    def fn_Self_attention(self, x: torch.Tensor):
        B_x, C_x, T_x, F_x = x.shape
        x_V = x.view(B, self.num_layers, C_x // self.num_layers, T_x, F_x)
        x_V = x_V.permute(0, 1, 3, 4, 2).contiguous()
        x_new = x.permute(0, 2, 3, 1).contiguous()
        Q = self.conv_Q(x_new)
        K = self.conv_K(x_new)
        V = self.conv_V(x_V)
        Q = self.prelu(Q)
        K = self.prelu(K)
        V = self.prelu(V)
        Q = self.ln_Q(Q)
        K = self.ln_K(K)
        V = self.ln_V(V)
        B, T, F, E = Q.shape
        Q = Q.contiguous().view(B, T, F * E)
        K = K.contiguous().view(B, T, F * E)
        K = K.transpose(1, 2)
        Att_mat = softmax(Q @ K / torch.sqrt(F * E))
        Att_layers = Att_mat.unsqueese(1)
        Att_layers = Att_layers.repeat(1, self.num_layers, 1, 1)

        V = V.view(B, self.num_layers, T, C_x // self.num_layers * F)

        A = Att_layers @ V

        A_permuted = A.permute(0, 2, 1, 3)

        B, T, L, D_L = A_permuted.shape
        x_final = A_permuted.contiguous().view(B, T, L * D_L)
        x_final = x_final.view(B, T, F_x, C_x)
        x_final = self.conv_out(x_final)
        x_final = self.prelu(x_final)
        x_final = self.ln_out(x_final) + x
        x_final = x_final.permute(0, 3, 1, 2)

        return x_final

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
