import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import warnings



class WeightedMSELoss(nn.Module):
    def __init__(self, factor=2):
        super(WeightedMSELoss, self).__init__()
        self.factor = factor

    def forward(self, predicted, target):
        # Calculate the mean squared error loss
        weights = torch.ones((target.shape[0], 1)).to(target.device)
        weights[target[:, 0] < 20] = self.factor
        weights[target[:, 0] > 60] = self.factor
        loss = torch.mean(weights * (predicted - target) ** 2)
        return loss


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 

'''A wrapper class for scheduled optimizer '''

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.param_groups = self._optimizer.param_groups


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        self.param_groups = self._optimizer.param_groups

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class CQR(nn.Module):
    """
    Confirmalized Quantile Regression. modified from:
    https://github.com/yromano/cqr/tree/master
    for more details see the original paper:
    "Conformalized Quantile Regression"
    Y. Romano, E. Patterson, E.J Candess, 2019, https://arxiv.org/pdf/1905.03222
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


    def calc_nc_error(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err


    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])
    def calibrate(self, preds, target):
        print("calibrate: ", preds.shape, target.shape)
        errs = []
        for i in range(len(self.quantiles)//2):
            y_lower = preds[:, i][:, None]
            y_upper = preds[:, -(i + 1)][:, None]
            q_pair = np.concatenate((y_lower, y_upper), axis=1)
            q_error = self.calc_nc_error(q_pair, target)
            errs.append(q_error)
        # self.nc_errs = np.array(errs)
        return np.array(errs).T

    def predict(self, preds, nc_errs):
        conformal_intervals = np.zeros_like(preds)
        for i in range(len(self.quantiles) // 2):
            significance = self.quantiles[-(i+1)] - self.quantiles[i]
            err_dist = self.apply_inverse(nc_errs[:, i], significance)
            err_dist = np.hstack([err_dist] * preds.shape[0])
            conformal_intervals[:, i] = preds[:, i] - err_dist[0, :]
            conformal_intervals[:, -(i+1)] = preds[:, -(i + 1)] + err_dist[1, :]
        conformal_intervals[:, len(self.quantiles) // 2] = preds[:, len(self.quantiles) // 2]
        return conformal_intervals


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class L1PercentageLoss(nn.Module):
    def __init__(self):
        super(L1PercentageLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure y_true has no zeros to avoid division by zero
        # assert torch.all(y_true != 0), "y_true contains zero values, which would cause division by zero."

        # Compute the L1 percentage loss
        loss = torch.abs(y_pred - y_true) / (y_true + 1e-4)
        
        # Return the mean loss
        return torch.mean(loss)



class _MaskedLoss(nn.Module):
    """Base class for masked losses"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    def forward(self, input, target, mask=None):
        """Compute a loss between input and target for given mask.
        Note that this implementation is faster than loss(input[mask], target[mask])
        for a given loss, and is nan-proof."""
        if not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2,
            )
        if mask is None:
            mask = torch.ones_like(input, dtype=bool)

        target_proxy = target
        if self.ignore_nans:
            target_proxy = target.clone()
            nans = torch.isnan(target)
            if nans.any():
                with torch.no_grad():
                    mask = mask & ~nans
                    target_proxy[nans] = 0
        full_loss = self.criterion(input, target_proxy)

        if not mask.any():
            warnings.warn(
                "Evaluation mask is False everywhere, this might lead to incorrect results.")
        full_loss[~mask] = 0

        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        if self.reduction == 'mean':
            return full_loss.sum() / mask.to(full_loss.dtype).sum()


class MaskedMSELoss(_MaskedLoss):
    """Masked MSE loss"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.MSELoss(reduction='none')


class MaskedL1Loss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.L1Loss(reduction='none')


class MaskedHuberLoss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True, delta=1):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.HuberLoss(reduction='none', delta=delta)
