import torch


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
    audio_names = ["audio_s1", "audio_s2", "audio_mix"]
    mouth_names = ["mouth1_emb", "mouth2_emb"]

    all_keys = set(dataset_items[0].keys())

    audio_batch = {
        name: torch.stack([elem[name] for elem in dataset_items])
        for name in audio_names if name in all_keys
    }

    mouth_batch = {
        name: torch.stack([elem[name] for elem in dataset_items])
        for name in mouth_names if name in all_keys
    }

    result_batch = audio_batch | mouth_batch

    excluded_keys = set(result_batch.keys())
    for k in dataset_items[0].keys():
        if k not in excluded_keys:
            result_batch[k] = [elem[k] for elem in dataset_items]

    return result_batch
