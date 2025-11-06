import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrogram_names = ["spectrogram_mix", "spectrogram_s1", "spectrogram_s2"]
    audio_names = ["audio_mix", "audio_s1", "audio_s2"]

    result_batch = {
        name: pad_sequence([elem[name] for elem in dataset_items], batch_first=True) 
        for name in spectrogram_names
    }
    result_batch.update(
         {
            name: pad_sequence([elem[name] for elem in dataset_items], batch_first=True)
            for name in audio_names
        }
    )

    result_batch.update(
        {
            name + "_length": torch.tensor(
                [elem[name].size(0) for elem in dataset_items], dtype=torch.int32)
            for name in spectrogram_names
        }
    )

    result_batch.update(
        {
            name + "_length": torch.tensor(
                [elem[name].size(0) for elem in dataset_items], dtype=torch.int32)
            for name in audio_names
        }
    )
  
    excluded_keys = set(result_batch.keys())
    for k in dataset_items[0].keys():
        if k not in excluded_keys:
            result_batch[k] = [elem[k] for elem in dataset_items]

    return result_batch
