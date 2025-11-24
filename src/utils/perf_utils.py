import time
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from thop import profile


def bytes_to_readable(n):
    for unit in ['B','KB','MB']:
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
        out = model(dummy_input)
    peak = torch.cuda.max_memory_allocated(device)
    return peak - baseline


@torch.inference_mode()
def count_time_per_step(model, dummy_input, device, n_warmup=10, n_iter=30):
    model.eval()

    for _ in range(n_warmup):
        out = model(dummy_input)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(n_iter):
        out = model(dummy_input)
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    avg = (end - start) / n_iter
    return avg


def count_flops_macs(model, dummy_input):
    model.eval()
    with torch.inference_mode():
        fca = FlopCountAnalysis(model, dummy_input)
        flops = fca.total()
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        
    return flops, macs


def profile_model(model, input_size, device=None, batch_size=1, measure_backward=False,
                  n_warmup=10, n_iter=30, dtype=torch.float32):
    """
    input_size: tuple of input shape EXCLUDING batch, e.g. (3,224,224)
    device: torch.device or None (auto choose cuda if available)
    Returns dict with metrics.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device).eval()
    dummy_input = torch.randn((batch_size, *input_size), dtype=dtype, device=device)

    # FLOPs
    flops = estimate_flops(model, dummy_input)

    # Peak memory for 1 batch
    peak_mem_bytes = measure_peak_memory(model, dummy_input, device, measure_backward=measure_backward)

    # state dict size
    state_bytes = get_state_dict_size_bytes(model)

    # time per step
    time_per_step = measure_time_per_step(model, dummy_input, device, n_warmup=n_warmup, n_iter=n_iter, measure_backward=measure_backward)

    return {
        'device': str(device),
        'flops': flops,  # may be None if tool not installed
        'peak_memory_bytes': peak_mem_bytes,
        'peak_memory_readable': bytes_to_readable(peak_mem_bytes),
        'state_dict_bytes': state_bytes,
        'state_dict_readable': bytes_to_readable(state_bytes),
        'time_per_step_s': time_per_step
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    # example model
    model = torchvision.models.resnet18(pretrained=False) if 'torchvision' in globals() else nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)
    )
    res = profile_model(model, input_size=(3,224,224), batch_size=8, measure_backward=False, n_warmup=5, n_iter=20)
    for k,v in res.items():
        print(f"{k}: {v}")
