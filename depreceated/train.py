import torch
import numpy as np

class Trainer2(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x, y,_,_) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            y_pred = self.model(x.float())
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                conf_y = torch.abs(y - y_pred) 
            # print(f"inclination range: {y_pred[:,1].min()} - {y_pred[:,1].max()}")
            # print(f"true y: {y[:10,:]}")
            # print(f"pred y: {y_pred[:10,:]}")

            # print("y_pred: ", y_pred.shape, "y: ", y.shape, conf)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"train_acc: {all_acc}, train_loss:  {loss.item()}") 
            all_accs = all_accs + all_acc
            
            # mean_acc = (diff[:,0]/(y[:,0])).sum().item()
            # train_acc2 += (diff[:,self.num_classes-1] < y[:,self.num_classes-1]/10).sum().item()
        print("number of train_accs: ", train_acc)
        return train_loss, all_accs/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        for i,(x, y,_,_) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            with torch.no_grad():
                y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                    conf_y = torch.abs(y - y_pred)
                else:
                    conf_pred = torch.ones_like(y_pred)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            # print("diff: ", diff.shape, "y: ", y.shape)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()  
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"val_acc: {all_acc}, val_loss:  {loss.item()}")
            all_accs = all_accs + all_acc
        return val_loss, all_accs/len(self.val_dl.dataset)

    def predict(self, test_dataloader, device, conf=True, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes, len(self.quantiles)))
        targets = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        for i,(x, y,_,_) in enumerate(test_dataloader):
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        return preds, targets

class latTrainer(Trainer):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i,(x, y, _,_)  in enumerate(self.train_dl):
            x, y = x.to(device), y.to(device)
            if torch.any(torch.isnan(x)):
                print("nan x")
            if torch.any(torch.isnan(y)):
                print("nan y")
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            # print("train: ", y_pred[:10], y[:10])           

            # print(self.model.named_parameters()[0][1].grad()[:10])
            # print("y_pred: ", y_pred[:10], "y: ", y[:10])
            loss = self.criterion(y_pred, y)
            
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     print(f'Parameter: {name}, Gradient flag: {param.requires_grad}, Value: {param.grad}')
            self.optimizer.step()
            acc = (y_pred.argmax(-1) == y).sum().item()
            train_loss.append(loss.item())
            train_acc += acc
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, _, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            # with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.val_dl))
            acc = (y_pred.argmax(-1) == y).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        for x, y, _, _ in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.zeros((0, self.num_classes))
    

class IncTrainer(Trainer):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i,(x, y, params,_)  in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            if torch.any(torch.isnan(x)):
                print("nan x")
            if torch.any(torch.isnan(y)):
                print("nan y")
            self.optimizer.zero_grad()
            # print('nans: ', torch.any(torch.isnan(x)).item(), torch.any(torch.isnan(params)).item())
            y_pred = self.model(x, params)

            # print("train: ", y_pred[:10], y[:10])           

            # print(self.model.named_parameters()[0][1].grad()[:10])
            # print("y_pred: ", y_pred[:10], "y: ", y[:10])
            loss = self.criterion(y_pred, y)
            
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     print(f'Parameter: {name}, Gradient flag: {param.requires_grad}, Value: {param.grad}')
            self.optimizer.step()
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum().item() 
            train_loss.append(loss.item())
            train_acc += acc
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item()}")
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, params, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            # with torch.no_grad():
            y_pred = self.model(x, params)
            loss = self.criterion(y_pred, y)
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.val_dl))
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        for x, y, params, _ in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, params)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.zeros((0, self.num_classes))


class EncoderTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y  in self.train_dl:
            # print("grads :", len(g), "last grad: ", g[-1])
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y) 
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss += loss.item()
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss/len(self.train_dl), train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.val_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss += loss.item()
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss/len(self.val_dl), val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets

class EncoderDecoderTrainer(Trainer):
    def __init__(self, recon_loss, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss = recon_loss
        self.alpha = alpha


    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for x, y,_,_  in pbar:
            x, y= x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            y_std = torch.std(y.squeeze(), dim=1)
            y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            # y_pred_std = torch.masked_select(y_pred_std, ~torch.isnan(y_pred_std))
            # y_std = torch.masked_select(y_std, ~torch.isnan(y_pred_std))
            # print("is nan: ", torch.any(torch.isnan(y_pred_std)).item(), torch.any(torch.isnan(y_std)).item())
            # print("y_pred_std: ", y_pred_std.shape, "y_std: ", y_std.shape, "y: ", y.shape, "y_pred: ", y_pred.shape)
            # print(y_pred.squeeze().shape, y.squeeze().shape)
            loss_pred = self.criterion(y_pred.squeeze(), y.squeeze())
            loss_std = self.criterion(y_pred_std, y_std)
            loss = loss_pred + self.alpha * loss_std
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            pbar.set_description(f"train_loss:  {loss.item()}")

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for x, y,_,_ in self.val_dl:
            x, y= x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss_pred = self.criterion(y_pred.squeeze(), y.squeeze())
            y_std = torch.std(y.squeeze(), dim=1)
            y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            loss_std = self.criterion(y_pred_std, y_std)
            # loss_recon = self.recon_loss(logits.argmax(dim=-1), x[:,0,:])
            # print(f"val - loss_pred: {loss_pred}, loss_recon: {loss_recon}")
            # loss = loss_pred + self.alpha * loss_recon
            loss = loss_pred + self.alpha * loss_std
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            # acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y,_,_ in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred, _ = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets
    
class DenoisingTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.train()
        train_loss = []
        for x, y  in self.train_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            if y_pred.isnan().any():
                print("y_pred is nan")
            # print(y_pred - y)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        return train_loss, 0 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.eval()
        val_loss = []
        for x, y  in self.val_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            val_loss.append(loss.item())
        return val_loss, 0
    


