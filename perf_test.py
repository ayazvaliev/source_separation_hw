import warnings

import torch
import hydra
from hydra.utils import instantiate
from src.utils.perf_utils import (
    bytes_to_readable,
    get_state_dict_size,
    get_peak_memory,
    count_time_per_step,
    count_flops_macs
)


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="perf")
def main(config):
    """
    Main script for evaluating model performance
    and resource consumption.

    Args:
        config (DictConfig): hydra experiment config.
    """

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    model = instantiate(config.model).to(device).eval()
    state_dict_size = get_state_dict_size(model)

    dummy_input = torch.randn(size=(1, 1, 32000), dtype=torch.float32, device=device, requires_grad=False) 

    if config.get("compile", False):
        assert not config.get("ts_compile", False)
        model = torch.compile(model, fullgraph=True, mode='reduce-overhead')
    elif config.get("ts_compile", False):
        model = torch.jit.trace(model, dummy_input)

    peak_memory = get_peak_memory(model, dummy_input, device)

    n_warmup = config.get("time_per_step_n_warmup", 10)
    n_iter = config.get("time_per_step_n_iter", 30)
    time_per_step = count_time_per_step(model, dummy_input, device, n_warmup, n_iter)
    flops, macs = count_flops_macs(model, dummy_input)

    res = {
        "state dict size": bytes_to_readable(state_dict_size),
        "peak memory usage": bytes_to_readable(peak_memory),
        "time_per_step": f"{time_per_step} seconds (was calculaed using n_warmup={n_warmup}, n_iter={n_iter})",
        "FLOPS": flops,
        "MACs": macs
    }

    for name, val in res:
        print(f"    {name:15s}: {val}")

if __name__ == "__main__":
    main()