import inspect


def get_optimizer_grouped_parameters(model, weight_decay=1e-2):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name.lower() or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def has_param(func, param_name):
    sig = inspect.signature(func)
    return param_name in sig.parameters