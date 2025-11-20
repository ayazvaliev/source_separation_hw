import logging
import random
import torchaudio
import numpy as np
import copy



import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
        self._index = self._filter_records_from_dataset(index, min_audio_length, max_audio_length)
        self.target_sr = sr
        self._index = self._shuffle_and_limit_index(index, limit, shuffle_index)

        self.instance_transforms = instance_transforms or {}

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

        data_dict = copy.deepcopy(self._index[ind])
        

        audio_names = ["audio_s1", "audio_s2", "audio_mix"]
        data_dict.update(
            {
                name: self.load_audio(data_dict[name + "_path"])
                for name in audio_names
            }
        )

        mouth_embs = ["mouth1_emb", "mouth2_emb"]
        mouth_emb_dict = {}
        for name in mouth_embs:
            file = np.load(data_dict[name + "_path"])
            mouth_emb_dict[name] = torch.from_numpy(file["data"])
        
        data_dict.update(mouth_emb_dict)

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
        return audio_tensor

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
        if "length" not in index[0] or (min_audio_length is None and max_audio_length is None):
            return index

        initial_size = len(index)
        audio_length_tensor = torch.tensor([el["audio_mix_time"] for el in index], dtype=torch.int32)
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
