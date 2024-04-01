from typing import Optional

import torch


def fill_nans(x: torch.Tensor, value: float, return_missing: bool = False):
    out = x.clone()
    missing = torch.isnan(out)
    out[missing] = value
    if return_missing:
        return out, missing
    else:
        return out


def add_gaussian_noise(x: torch.Tensor, sigma: float, mask: Optional[torch.Tensor] = None):
    out = x.clone()
    if mask is None:
        out += torch.randn_like(out) * sigma
    else:
        out[~mask] += torch.randn_like(out[~mask]) * sigma
    return out

def normalize(x: torch.Tensor, mask: Optional[torch.Tensor] = None, norm_type: str = 'std'):
    if mask is None:
        mask = torch.zeros_like(x[:,0]).bool()
    for c in range(x.shape[-1]):
        if norm_type == 'std':
            x[:,c] = ((x[:,c] - x[:,c][~mask.squeeze()].mean())
                                       / (x[:,c][~mask.squeeze()].std() + 1e-8))
        elif norm_type == 'median':
            x[:, c] /= x[:, c][~mask.squeeze()].median()
        elif norm_type == 'minmax':
            mini = x[:,c][~mask.squeeze()].min().values
            maxi = x[:,c][~mask.squeeze()].max().values
            x[:,c] = (x[:,c] - mini) / (maxi - mini)
    return x