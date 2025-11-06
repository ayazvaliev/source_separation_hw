import torchaudio
from collections import OrderedDict
from tqdm.auto import tqdm
from pathlib import Path
import os
import yadisk
import zipfile

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MainDataset(BaseDataset):

    def __init__(
        self,
        data_root,
        name="train", 
        index_dir=None,
        dataset_url="https://disk.360.yandex.ru/d/9k_k6G6a03GURg",
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
        self.data_root = Path(data_root)
        if index_dir is None:
            index_dir = data_root
        else:
            index_dir = Path(index_dir)
        index_path = index_dir if index_dir is not None else self.data_root
        index_path = index_path / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(index_path)
        else:
            os.makedirs(str(index_path.parent), exist_ok=True)
            index = self._create_index(name, index_path, dataset_url)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, 
                      name, 
                      index_path,
                      dataset_url):
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
        if dataset_url is not None and dataset_url.startswith("http"):
            output_path = self.data_root / "dla_dataset.zip"
            y = yadisk.Client()

            print("Downloading ZIP from Yandex.Disk...")
            y.download_public(dataset_url, output_path)

            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(str(self.data_root))
        
            os.remove(output_path)
        elif not dataset_url.startswith("http"):
            raise RuntimeError("dataset path must be either URL or None")

        audio_path = self.data_root / "dla_dataset" / "audio" / name 
        mouths_path  =  self.data_root / "dla_dataset" / "mouths"
        

        for item in tqdm((audio_path / "mix").iterdir()):
            # create dataset
            item_name = item.name

            audio_mix_path = str(audio_path / "mix" / item_name)
            audio_s1_path = str(audio_path / "s1" / item_name)
            audio_s2_path = str(audio_path / "s2" / item_name)

            mix_info = torchaudio.info(audio_mix_path)
            s1_info = torchaudio.info(audio_s1_path)
            s2_info = torchaudio.info(audio_s2_path)

            data_instance = OrderedDict(
                {
                "audio_mix_path" : str(audio_path / "mix" / item_name),
                "audio_s1_path" : str(audio_path / "s1" / item_name),
                "audio_s2_path" : str(audio_path / "s2" / item_name),
                "mouths_path" : str(mouths_path / item_name),
                "audio_mix_time": mix_info.num_frames / mix_info.sample_rate,
                "audio_s1_time": s1_info.num_frames / mix_info.sample_rate,
                "audio_s2_time": s2_info.num_frames / s2_info.sample_rate
                }
            )
            index.append(data_instance)

        # write index to disk
        write_json(index, str(index_path))
   
        return index
