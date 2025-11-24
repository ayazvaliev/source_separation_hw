import time

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def bytes_to_readable(n):
    for unit in ["B", "KB", "MB"]:
        if n < 1024.0:
            return f"{n:0.5f} {unit}"
        n /= 1024.0
    return f"{n:0.5f} MB"


def get_state_dict_size(model):
    total = 0
    for p in model.state_dict().values():
        total += p.numel() * p.element_size()
    return total


def get_peak_memory(model, dummy_input, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    model.eval()
    with torch.inference_mode():
        model(dummy_input)
    peak = torch.cuda.max_memory_allocated(device)
    return peak - baseline


@torch.inference_mode()
def count_time_per_step(model, dummy_input, device, n_warmup=10, n_iter=30):
    model.eval()

    for _ in range(n_warmup):
        model(dummy_input)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_iter):
        model(dummy_input)
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    avg = (end - start) / n_iter
    return avg


def count_flops_macs(model, dummy_input):
    model.eval()
    with torch.inference_mode():
        macs, _ = get_model_complexity_info(
            model, tuple(dummy_input.shape[1:]), as_strings=False, print_per_layer_stat=False
        )

    return 2 * macs, macs
