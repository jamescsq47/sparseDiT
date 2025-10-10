from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

if dist.is_available():
    import torch.distributed._functional_collectives as ft_c
    import torch.distributed.distributed_c10d as c10d
else:
    ft_c = None
    c10d = None


def get_group(group=None):
    if group is None:
        group = c10d._get_default_group()

    if isinstance(group, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = group
    else:
        pg = group.get_group()

    return pg


def get_world_size(group=None):
    pg = get_group(group)
    return dist.get_world_size(pg)


def get_rank(group=None):
    pg = get_group(group)
    return dist.get_rank(pg)


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def all_gather_tensor_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x_shape = x.shape
    x = x.flatten()
    x_numel = x.numel()
    x = ft_c.all_gather_tensor(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    x_shape = list(x_shape)
    x_shape[0] *= x.numel() // x_numel
    x = x.reshape(x_shape)
    return x


def all_gather_tensor_autograd_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x_shape = x.shape
    x = x.flatten()
    x_numel = x.numel()
    x = ft_c.all_gather_tensor_autograd(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    x_shape = list(x_shape)
    x_shape[0] *= x.numel() // x_numel
    x = x.reshape(x_shape)
    return x


def all_to_all_single_sync(x, *args, **kwargs):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, *args, **kwargs)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def all_to_all_single_autograd_sync(x, *args, **kwargs):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single_autograd(x, *args, **kwargs)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def all_reduce_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x = ft_c.all_reduce(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    return x


def get_buffer(
    shape_or_tensor: Union[Tuple[int], torch.Tensor],
    *,
    repeats: int = 1,
    dim: int = 0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    group=None,
) -> torch.Tensor:
    if repeats is None:
        repeats = get_world_size(group)

    if isinstance(shape_or_tensor, torch.Tensor):
        shape = shape_or_tensor.shape
        dtype = shape_or_tensor.dtype
        device = shape_or_tensor.device

    assert dtype is not None
    assert device is not None

    shape = list(shape)
    if repeats > 1:
        shape[dim] *= repeats

    buffer = torch.empty(shape, dtype=dtype, device=device)
    return buffer


def get_assigned_chunk(
    tensor: torch.Tensor,
    dim: int = 0,
    idx: Optional[int] = None,
    group=None,
) -> torch.Tensor:
    if idx is None:
        idx = get_rank(group)
    world_size = get_world_size(group)
    total_size = tensor.shape[dim]
    # assert total_size % world_size == 0, f"tensor.shape[{dim}]={total_size} is not divisible by world_size={world_size}"
    # chunk_size = (total_size + world_size - 1) // world_size
    # pad_size = chunk_size * world_size - total_size
    # if pad_size > 0:
    #     # 在 dim 维度后面 pad
    #     pad_shape = list(tensor.shape)
    #     pad_shape[dim] = pad_size
    #     pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    #     tensor = torch.cat([tensor, pad_tensor], dim=dim)
    return tensor.chunk(world_size, dim=dim)[idx]


def get_complete_tensor(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    group=None,
) -> torch.Tensor:
    tensor = tensor.transpose(0, dim).contiguous()
    output_tensor = all_gather_tensor_sync(tensor, gather_dim=0, group=group)
    output_tensor = output_tensor.transpose(0, dim)
    return output_tensor
