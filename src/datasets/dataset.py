import torchaudio
from collections import OrderedDict
from tqdm.auto import tqdm
from pathlib import Path
import os
import yadisk
import zipfile
import subprocess
import sys
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

#original_dir = Path.cwd()
#os.chdir('Lipreading_using_Temporal_Convolutional_Networks')
sys.path.append('Lipreading_using_Temporal_Convolutional_Networks')
#print(f"Changed to: {Path.cwd()}")
from data_embedding_utils import MouthEmbeddingExtractor
#os.chdir(original_dir)


class MainDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        name="train", 
        index_dir=None,
        dataset_url=None,
        get_mouths=True,
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
        if index_dir is None:
            index_dir = data_root
        else:
            index_dir = Path(index_dir)
        index_path = index_dir if index_dir is not None else self.data_root
        index_path = index_path / name / "index.json"

        if self.get_mouths:
            self.extractor = MouthEmbeddingExtractor(
                config_path="Lipreading_using_Temporal_Convolutional_Networks/configs/lrw_resnet18_dctcn_boundary.json",
                model_path="Lipreading_using_Temporal_Convolutional_Networks/models/lrw_resnet18_dctcn_video_boundary.pth"
            )

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(index_path)
        else:
            os.makedirs(str(index_path.parent), exist_ok=True)
            index = self._create_index(name, index_path, dataset_url)

        super().__init__(index, get_mouths, *args, **kwargs)

    def _create_index(self, 
                      name: str, 
                      index_path: Path,
                      dataset_url: None | str):
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
        print("dataset_url", dataset_url)
        if dataset_url is not None:
            if dataset_url.startswith('http'):
                output_path = self.data_root / "dla_dataset.zip"
                y = yadisk.Client()

                print("Downloading ZIP from Yandex.Disk...")
                y.download_public(dataset_url, str(output_path))

                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(str(self.data_root))
        
                os.remove(output_path)
            else:
                raise RuntimeError("dataset path must be either URL or None")

        audio_path = self.data_root / "dla_dataset" / "audio" / name 
        mouths_path  =  self.data_root / "dla_dataset" / "mouths"
        
     
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
        write_json(index, str(index_path))
   
        return index


    def get_mouth_embeddings(self, mouth_path, mouth_name, save_dir):

        save_path = os.path.join(str(save_dir), str(mouth_name) + ".npz")

        if os.path.exists(save_path):
            return save_path
        self.extractor.extract_and_save(mouth_path, save_path)
        return save_path

