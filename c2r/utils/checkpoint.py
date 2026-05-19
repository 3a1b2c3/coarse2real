from contextlib import contextmanager
from typing import Any

import torch
from safetensors import safe_open


@contextmanager
def init_weights_on_device(device: torch.device = torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    old_register_buffer = torch.nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    tensor_constructors = {}
    if include_buffers:
        tensor_constructors = {name: getattr(torch, name) for name in ["empty", "zeros", "ones", "full"]}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
            for name in tensor_constructors:
                setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
            for name, old_fn in tensor_constructors.items():
                setattr(torch, name, old_fn)


def load_state_dict(file_path: str, torch_dtype: torch.dtype | None = None, device: str = "cpu") -> dict[str, Any]:
    if file_path.endswith(".safetensors"):
        state_dict = _load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        state_dict = _load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)
    return state_dict


def _load_state_dict_from_safetensors(
    file_path: str,
    torch_dtype: torch.dtype | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(file_path, framework="pt", device=str(device)) as file:
        for key in file.keys():
            tensor = file.get_tensor(key)
            if torch_dtype is not None:
                tensor = tensor.to(torch_dtype)
            state_dict[key] = tensor
    return state_dict


def _load_state_dict_from_bin(
    file_path: str,
    torch_dtype: torch.dtype | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(torch_dtype)
    return state_dict
