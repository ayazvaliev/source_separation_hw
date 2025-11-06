import logging
import random
from typing import List
from hydra.utils import instantiate
import torchaudio
from pathlib import Path



import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
torchaudio.set_audio_backend("soundfile") 

class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self, 
        index, 
        sr, 
        limit=None, 
        shuffle_index=False, 
        instance_transforms=None,
        min_audio_length=None,
        max_audio_length=None,
        **kwargs
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)
        self._index: List[dict] = self._filter_records_from_dataset(index, min_audio_length, max_audio_length)
        self.target_sr = sr
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """

        data_dict = self._index[ind]

        audios = {
            name[:name.rfind("_")]: self.load_audio(data_dict[name])
            for name in ["audio_mix_path", "audio_s1_path", "audio_s2_path"]
        }

        audio["audio_mix"] = (
            self.instance_transforms["audio_mix"](audios["audio_mix"])
            if (self.instance_transforms is not None and "audio_mix" in self.instance_transforms)
            else audios["audio_mix"]
        )

        for name, audio in zip(["spectrogram_mix", "spectrogram_s1", "spectrogram_s2"],
                               audios):
            data_dict.update(
                {
                    name: self.get_spectrogram(audio)
                }
            )
        data_dict.update({k: v.squeeze(0) for k, v in audios.items()})
        data_dict = self.preprocess_data(data_dict)

        return data_dict

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def get_spectrogram(self, audio: torch.Tensor):
        spectrogram = self.instance_transforms["get_spectrogram"](audio)
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)
        return spectrogram

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor.unsqueeze(0)

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name in {"audio_mix", "get_spectrogram"}:
                    continue
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index,
        min_audio_length,
        max_audio_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        audio_length_tensor = torch.tensor([el["audio_mix_len"] for el in index], dtype=torch.int32)
        if max_audio_length is not None:
            exceeds_audio_length = audio_length_tensor >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        if min_audio_length is not None:
            exceeds_audio_length = exceeds_audio_length | (audio_length_tensor < min_audio_length)
        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset")

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert entry != {}, (
                "Each dataset item should include data" 
            )
            

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
