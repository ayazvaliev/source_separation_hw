from src.model.rtfsnet import RTFSNet
import torch





block = RTFSNet(
    n_feats = 1024//2+1,        
    n_fft = 1024,         
    hop_length = 256,     
    win_length = 1024 ,
    time_dim = 126   
)

video_1 = torch.randn(3, 50, 512)
video_2 = torch.randn(3, 50, 512)
Audio_1 = torch.randn(3, 32000)


res = block(Audio_1, video_1, video_2)
print( res["logits"].shape)