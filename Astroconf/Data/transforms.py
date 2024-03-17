import torch
import warnings
import numpy as np

class Compose:
    """Composes several transforms together. 
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=None):
        out = x
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
        return out, mask, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class RandomCrop:
    def __init__(self, width, exclude_missing_threshold=None):
        self.width = width
        self.exclude_missing_threshold = exclude_missing_threshold
        assert exclude_missing_threshold is None or 0 <= exclude_missing_threshold <= 1

    def __call__(self, x, mask=None, info=None):
        seq_len = x.shape[0]
        if seq_len < self.width:
            left_crop = 0
            warnings.warn(
                'cannot crop because width smaller than sequence length')
        else:
            left_crop = np.random.randint(seq_len-self.width)
        info['left_crop'] = left_crop

        out_x = x[left_crop:left_crop+self.width]
        if mask is None:
            return (out_x, mask, info)
        if self.exclude_missing_threshold is not None and np.isnan(out_x).mean() >= self.exclude_missing_threshold:
            return self.__call__(x, mask=mask, info=info)
        out_m = mask[left_crop:left_crop+self.width]

        return (out_x, out_m, info)

class Slice():
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __call__(self, x, mask=None, info=None):
        sliced_mask = mask[self.start:self.end] if mask is not None else None
        return x[self.start:self.end], sliced_mask, info

class Detrend():
    def __init__(self, type='diff'):
        self.type = type
    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            out = x.copy()
        elif isinstance(x, torch.Tensor):
            out = x.clone()
        out[1:] = x[1:] - x[:-1]
        out[0] = out[1]
        if mask is not None:
            mask = mask[1:]
        info['detrend'] = self.type
        return out*10**6, mask, info

class moving_avg():
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def __call__(self, x, mask=None, info=None):
        # padding on the both ends of time series
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.unsqueeze(-1).unsqueeze(0)
        front = x[:, 0:1, :].repeat(1,(self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        info['moving_avg'] = self.kernel_size
        return x.squeeze(), mask, info
