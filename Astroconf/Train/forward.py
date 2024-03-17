import numpy as np
import torch
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss
# from .utils import check_nan

# NCE Loss
def infoNCELoss(y, temperature=0.07):
  y = F.normalize(y, dim=1)
  logits = y @ y.transpose(0, 1)/temperature
  logits.fill_diagonal_(float('-1e9'))
  labels = torch.block_diag(*torch.ones(len(y)//2, 2, 2, device=y.device))
  labels.fill_diagonal_(0)
  loss = F.cross_entropy(logits, labels)
  return loss

def model_fn_nomask(batch, model, args, rank):
  """Forward a batch through the model."""
  lcs, labels, kids = batch
  lcs = lcs.to(rank)
  labelsondevice = labels.to(rank)
  
  with torch.autocast(device_type=args.device_type, dtype=torch.float16, enabled=args.use_amp):
    outs = model(lcs)
    loss_fn = mse_loss if args.loss == 'mse' else l1_loss if args.loss == 'l1' else smooth_l1_loss
    loss = loss_fn(outs, labelsondevice)
  
  # if loss.isnan():


  return loss, outs, labels, kids
  
def model_fn_mask(batch, model, args, rank):
  """Forward a batch through the model."""
  lc, mask, label, kid = batch
  lc = lc.to(rank)
  mask = mask.to(rank)
  labelsondevice = label.to(rank)

  with torch.autocast(device_type=args.device_type, dtype=torch.float16, enabled=args.use_amp):
    outs = model(lc, key_padding_mask = mask)
    loss_fn = mse_loss if args.loss == 'mse' else l1_loss if args.loss == 'l1' else smooth_l1_loss
    loss = loss_fn(outs, labelsondevice)

  return loss, outs, label, kid

def model_fn_semisupervision(batch, model, args):
  pass

def valid(dataloader, model, model_fn, args, rank): 
  """Validate on validation set."""
  if dataloader is None:
    return None, None, None, None
  
  model.eval()
  transform = dataloader.dataset.dataset.transform
  dataloader.dataset.dataset.transform = False
  preds = []
  labels = []
  kids = []
  
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      loss, pred, label, kid = model_fn(batch, model, args, rank)
      preds.append(pred.detach().cpu().numpy())
      labels.append(label.detach().cpu().numpy())
      kids.append(kid.detach().cpu().numpy())

  model.train()
  # preds = torch.cat(preds)
  # labels = torch.cat(labels)
  dataloader.dataset.dataset.transform = transform
  return loss.item(), np.concatenate(preds), np.concatenate(labels), np.concatenate(kids)