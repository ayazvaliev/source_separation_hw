sudo apt install libgl1-mesa-glx
gdown "1TGFG0dW5M3rBErgU8i0N7M1ys9YMIvgm" -O Lipreading_using_Temporal_Convolutional_Networks/models/lrw_resnet18_dctcn_video_boundary.pth

sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install build-essential dkms

# Add NVIDIA package repository
distribution=$(
  . /etc/os-release
  echo $ID$VERSION_ID
)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-driver-580-server
