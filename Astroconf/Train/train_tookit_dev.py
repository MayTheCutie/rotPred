import time
import copy
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils import same_seeds
from .utils import save_checkpoint, check_gpu_use, log_parameters_gradients_in_model, uniform_sample, outlier_detection, inspect_snapshot, visualize_attention
from .forward import valid



def train(model, model_fn, optimizer, scheduler, scaler, train_loader, valid_loader, test_loader, args, rank):
  '''Train the model for the full set of steps.'''

  init_step = 0
  epoch = 0
  null_step = 0
  best_loss = 500
  snapshot = []
  # samples = uniform_sample(train_loader, args)
  print("args per epoch: ", args.batch_per_epoch, flush=True)
  if args.use_checkpoint:
    # if there is a checkpoint, load checkpoint
    model_dir = args.checkpoint_dir
    state = torch.load(model_dir)

    if not args.from_pretrained:
      model_fold = state['fold']
      if model_fold > args.fold:
        with open(args.log_dir+'log.txt', 'a') as fp:
          stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
          fp.write(f'{stamp} The checkpoint is not in the same fold as the current fold. Skip loading checkpoint.\n')
        return 
    
    try:
      model.load_state_dict(state['net'])
      with open(args.log_dir+'log.txt', 'a') as fp:
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        fp.write(f'{stamp} Load checkpoint from {model_dir} successfully!\n')
      if not args.from_pretrained:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        init_step = state['step']+1
        epoch = init_step // args.batch_per_epoch
        null_step = state['null_step']
        snapshot = state['snapshot']
        best_loss = snapshot[1][0]
        
    except Exception as e:
      with open(args.log_dir+'log.txt', 'a') as fp:
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        fp.write(f'{stamp} Failed to strictly load the model. Error: {e}\n')
      model.load_state_dict(state['net'], strict=False)
      

  # training
  model.train()
  args.tb_dir = f'/g/data/y89/jp6476/Learning_curves/{args.target}/{args.comment}_{args.fold}/'
  writer = SummaryWriter(log_dir=args.tb_dir)
  same_seeds(args.randomseed)
  pbar = tqdm(range(init_step, args.total_steps))
  epoch_train_loss = []
  epoch_val_loss = []
  epoch_val_acc = []
  tr_losses = []
  for step in pbar:
    batch = next(iter(train_loader))
    tr_loss, tr_pred, tr_label, tr_kid = model_fn(batch, model, args, rank)
    tr_losses.append(tr_loss.cpu().item())
    scaler.scale(tr_loss).backward()
    scaler.unscale_(optimizer)
    if args.grad_clip:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()
    tr_loss = tr_loss.cpu().item()
    if rank == 0:
        writer.add_scalar('Loss/train', tr_loss, step)

    if step == 0 and args.check_gpu:
      check_gpu_use(args)

    if (step + 1) % args.batch_per_epoch == 0:
      epoch += 1
      if args.distributed:
        epoch_loss = torch.tensor(tr_losses, device=rank).mean()
        torch.distributed.reduce(epoch_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        epoch_loss /= torch.distributed.get_world_size()
        epoch_train_loss.append(epoch_loss.cpu().item())
      # if torch.distributed.get_rank() == 0:
      if torch.distributed.get_rank() == 0:
        print(f"Epoch {epoch}, Loss {np.mean(epoch_train_loss):.5f}")
    pbar.set_description(f"Epoch {epoch}, Step {step}, Loss {tr_loss:.5f}, lr {optimizer.param_groups[0]['lr']:.5f}")
    #validation
    if step % args.valid_steps == 0:
      if args.save_checkpoint:
        save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, 'model_now')
      if args.check_param:
        log_parameters_gradients_in_model(model, writer, step)
      # if torch.distributed.get_rank() == 0:
        
      val_loss, val_pred, val_label, val_kid = valid(valid_loader, model, model_fn, args, rank)
      print("val shapes", val_pred.shape, val_label.shape)
      corrects = np.abs(val_pred - val_label) < 0.1*val_label
      val_acc = np.sum(corrects, axis=0) / len(val_label)
      print(f"step {step}, Training loss {tr_loss:.5f}, Validation loss {val_loss:.5f}, Validation accuracy, {val_acc.tolist()}")

      # if args.distributed:
      val_loss, val_acc =torch.tensor(val_loss, device=rank), torch.tensor(val_acc, device=rank)
      torch.distributed.reduce(val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
      val_loss = (val_loss / torch.distributed.get_world_size()).cpu().item()
      epoch_val_loss.append(val_loss)
      torch.distributed.reduce(val_acc, dst=0, op=torch.distributed.ReduceOp.SUM)
      val_acc = (val_acc / torch.distributed.get_world_size()).cpu()
      epoch_val_acc.append(val_acc)
      if rank == 0:
        writer.add_scalar('Loss/valid', val_loss, step)
        writer.add_scalar('Acc/valid', val_acc.mean(), step)
    
        with open(args.log_dir+'log.txt', 'a') as fp:
          stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
          # kind of difficult to switch line in a string
          # pred_layer = model.pred_layer[3].bias.item() if not args.distributed else model.module.pred_layer[3].bias.item()
          fp.write(stamp+f" rank {rank},  epoch {step//args.batch_per_epoch}, step {step},tr_loss={tr_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc.tolist()}, lr={optimizer.param_groups[0]['lr']}\n")

      if val_loss < best_loss:
        null_step = 0
        best_loss = val_loss
        with open(args.log_dir+'log.txt', 'a') as fp:
          fp.write(f"step {step + 1}, best model. (loss={best_loss:.5f})\n")
        snapshot = [[tr_loss, tr_pred.detach().cpu(), tr_label, tr_kid], [val_loss, val_pred, val_label, val_kid]]
        if args.save_checkpoint:
          save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, f'modelbestloss_{args.fold}')

    # early stop
    null_step += 1
    if null_step == args.early_stop:
      break

    # if step % args.visual_steps == 0 or step == args.total_steps - 1:
    #   model_temp = copy.deepcopy(model)
    #   model_temp.load_state_dict(torch.load(args.log_dir + f'modelbestloss_{args.fold}.ckpt')['net']).to(args.device)
    #   outliers = outlier_detection(test_loader, test_pred, test_label, test_kid, args)
    #   inspect_snapshot(step, snapshot, args)
      # if args.visulize_attention_map:
      #   visulize_attention(step, model, samples, args)
  
  test_loss, test_pred, test_label, test_kid = valid(test_loader, model, model_fn, args, rank)
  test_acc = np.sum(np.abs((test_pred - test_label)) < 0.1*test_label) / len(test_label)
  if test_loader is not None:
    writer.add_scalar('Loss/test', test_loss, step)
    writer.add_scalar('Acc/test', test_acc.mean(), step)
  writer.close()

  return best_loss