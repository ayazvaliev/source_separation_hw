from src.model.rtfsnet import RTFSNet
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

block = RTFSNet(
    n_feats=1024 // 2 + 1,  # = 513
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    time_dim=126
).to(device)

video_1 = torch.randn(3, 50, 512).to(device)
video_2 = torch.randn(3, 50, 512).to(device)
Audio_1 = torch.randn(3, 32000).to(device)

res = block(Audio_1, video_1, video_2)

print(res["logits"].shape)
