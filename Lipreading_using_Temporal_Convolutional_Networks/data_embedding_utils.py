import torch
import numpy as np
import os

from lipreading.utils import load_json, save2npz, load_model
from lipreading.model import Lipreading
from lipreading.dataloaders import get_preprocessing_pipelines


class MouthEmbeddingExtractor:
    def __init__(self, config_path, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load config
        args_loaded = load_json(config_path)

        # Set up model params
        backbone_type = args_loaded['backbone_type']
        width_mult = args_loaded['width_mult']
        relu_type = args_loaded['relu_type']
        use_boundary = args_loaded.get("use_boundary", False)

        tcn_options = {}
        if args_loaded.get('tcn_num_layers', ''):
            tcn_options = {
                'num_layers': args_loaded['tcn_num_layers'],
                'kernel_size': args_loaded['tcn_kernel_size'],
                'dropout': args_loaded['tcn_dropout'],
                'dwpw': args_loaded['tcn_dwpw'],
                'width_mult': args_loaded['tcn_width_mult'],
            }

        densetcn_options = {}
        if args_loaded.get('densetcn_block_config', ''):
            densetcn_options = {
                'block_config': args_loaded['densetcn_block_config'],
                'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                'reduced_size': args_loaded['densetcn_reduced_size'],
                'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                'squeeze_excitation': args_loaded['densetcn_se'],
                'dropout': args_loaded['densetcn_dropout'],
            }

        self.model = Lipreading(
            modality='video', #modality
            num_classes=500, #default
            tcn_options=tcn_options,
            densetcn_options=densetcn_options,
            backbone_type=backbone_type,
            relu_type=relu_type,
            width_mult=width_mult,
            use_boundary=use_boundary,
            extract_feats=True # default
        ).to(self.device)

        self.model = load_model(model_path, self.model)
        self.preprocessing_func = get_preprocessing_pipelines('video')['test']

    def extract_and_save(self, mouth_path, save_path):
        self.model.eval()
        data = self.preprocessing_func(np.load(mouth_path)['data'])
        with torch.no_grad():
            features = self.model(torch.FloatTensor(data)[None, None, :, :, :].to(self.device), lengths=[data.shape[0]])
        save2npz(save_path, data=features.cpu().detach().numpy())
