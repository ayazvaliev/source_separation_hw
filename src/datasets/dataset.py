import torchaudio
from collections import OrderedDict
from tqdm.auto import tqdm
from pathlib import Path
import os
import yadisk
import zipfile
import subprocess

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        name="train", 
        index_dir=None,
        dataset_url=None,
        get_mouths=False,
        inference_mode=False,
        *args, 
        **kwargs
    ):
        """
        Args:
            data_root (str): path to dataset dir
            name (str): partition name
            index_dir (str): path to index dir (convenient for kaggle as their dataset section is ronly)
            dataset_url (str): URL to dataset.
        """
        self.get_mouths = get_mouths
        self.data_root = Path(data_root)
        self.inference_mode = inference_mode

        if not inference_mode:
            if index_dir is None:
                index_dir = self.data_root
            else:
                index_dir = Path(index_dir)
            index_path = index_path / name / "index.json"

            # each nested dataset class must have an index field that
            # contains list of dicts. Each dict contains information about
            # the object, including label, path, etc.
            if index_path.exists():
                index = read_json(index_path)
            else:
                os.makedirs(str(index_path.parent), exist_ok=True)
                index = self._create_index(name, index_path, dataset_url)
        else:
            index = self._create_index(name=None,
                                       index_path=None,
                                       dataset_url=dataset_url,
                                       write_to_disk=False)

        super().__init__(index, get_mouths, *args, **kwargs)

    def _create_index(self, 
                      name: str, 
                      index_path: Path,
                      dataset_url: None | str,
                      write_to_disk=True):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
            path (str): path to yandex disk file
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        if dataset_url is not None:
            if dataset_url.startswith('http'):
                y = yadisk.YaDisk()
                meta = y.get_public_meta(dataset_url)
                total_size = meta.size
                file_name = meta.name

                print(f"Downloading {file_name} ({total_size / 1e6:.2f} MB)")

                archive_path = self.data_root / file_name
                y.download_public(dataset_url, str(archive_path))

                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_root)
                    top_level_dir = set(
                        name.split(".")[0]
                        for name in zip_ref.namelist()
                        if name.strip()
                    )
                    print(top_level_dir)
                    assert len(top_level_dir) == 1 or len(top_level_dir.intersection({"audio", "mouths"})) == 2, "Wrong format for inference dir"
                    if len(top_level_dir) == 1:
                        top_level_dir = top_level_dir.pop()
                    else:
                        top_level_dir = None

                os.remove(archive_path)
            else:
                raise RuntimeError("dataset path must be either URL or None")
        
        top_level_dir = top_level_dir or ""
        if not self.inference_mode:
            audio_path = self.data_root / top_level_dir / "audio" / name 
            mouths_path  =  self.data_root / top_level_dir / "mouths"
        else:
            audio_path = self.data_root / top_level_dir / "audio"
            mouths_path  =  self.data_root / top_level_dir / "mouths"
     
        for item in tqdm((audio_path / "mix").iterdir()):
            # create dataset
            item_name = item.name

            audio_mix_path = str(item)
            audio_s1_path = str(audio_path / "s1" / item_name)
            audio_s2_path = str(audio_path / "s2" / item_name)

            data_instance = OrderedDict()

            for name, path in zip(["audio_mix", "audio_s1", "audio_s2"], 
                                  [audio_mix_path, audio_s1_path, audio_s2_path]):
                if os.path.exists(path):
                    data_instance[name + "_path"] = path

            # info = torchaudio.info(audio_mix_path)
            # data_instance["length"] = info.num_frames / info.sample_rate
            if self.get_mouths:
                data_instance["mouths_path"] = str(mouths_path / item_name)
                
                mouth1_name, mouth2_name = item_name[:-4].split("_")
                mouth1_npz = str(mouths_path / mouth1_name) + ".npz"
                mouth2_npz = str(mouths_path / mouth2_name) + ".npz"

                mouths_emb_dir = self.data_root / "dla_dataset" / "mouth_embeddings"
                if not os.path.exists(mouths_emb_dir):
                    os.makedirs(mouths_emb_dir)

                data_instance["mouth1_emb_path"] = self.get_mouth_embeddings(mouth_path=mouth1_npz,
                mouth_name=mouth1_name,
                save_dir=mouths_emb_dir)
                data_instance["mouth2_emb_path"] = self.get_mouth_embeddings(mouth_path=mouth2_npz,
                mouth_name=mouth2_name,
                save_dir=mouths_emb_dir)

            index.append(data_instance)

        # write index to disk
        if write_to_disk:
            write_json(index, str(index_path))
   
        return index


    def get_mouth_embeddings(self, mouth_path, mouth_name, save_dir):
        
        save_path = str(save_dir) + "\\" + str(mouth_name) + ".npz"
        
        if os.path.exists(save_path):
            return save_path
        command = [
        "python", "Lipreading_using_Temporal_Convolutional_Networks/main.py",
        "--modality", "video",
        "--extract-feats",
        "--config-path", "Lipreading_using_Temporal_Convolutional_Networks/configs/lrw_resnet18_dctcn_boundary.json",  
        "--model-path", "Lipreading_using_Temporal_Convolutional_Networks/models/lrw_resnet18_dctcn_video_boundary.pth",
        "--mouth-patch-path", mouth_path,
        "--mouth-embedding-out-path", save_path
        ]
        result = subprocess.run(command, capture_output=True)
        return save_path
