import torchaudio
import torch
from pathlib import Path
torchaudio.set_audio_backend("soundfile") 
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    conf = config.feature_gainer["preprocessing_methods"]
    
    for name in conf.keys():
        print("\n","\n","\n",name,"\n","\n","\n","\n")
        func = instantiate(conf[name])
        path = "E:\\Python_DLA_proj\\Speach_sep\\pytorch_project_template\\data\\example\\train\\dla_dataset\\audio\\train\\mix\\00000002026_00315409549.wav"
        data_object, sample_rate = torchaudio.load(path)
        print(func(data_object).shape)

main()