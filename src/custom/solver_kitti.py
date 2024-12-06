import numpy as np
import torch.nn as nn
import time
import torch
import copy
from models import MyModel
import matplotlib.pyplot as plt
import pathlib
from data_processing_cifar import DataProcessing
from data_processing_kitti import DataProcessorKitti
from evalutation import compute_loss
from src.custom.SimpleYOLO import SimpleYOLO
from torchinfo import summary


class SolverKitti(object):
    def __init__(self, **kwargs):
        '''
        Class for managing the training of deep learning models
        
        Args:
            *kwargs
            
        Returns:
            None
        '''
        
        # Define some parameters if they are not passed in, and add all to object
        self.batch_size = kwargs.pop("batch_size", 10)
        self.device = kwargs.pop("device", "cpu")
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup = kwargs.pop("warmup", 0)
        self.save_best = kwargs.pop("save_best", True)
        self.model_type = kwargs.pop("model_type", "linear")
        self.data_type = kwargs.pop("data_type", "cifar")
        
        # Define the data
        if self.data_type == "cifar":
            self.train_loader, self.val_loader, self.test_dataset = DataProcessing(self.batch_size)
        elif self.data_type == "kitti": 
            self.train_loader, self.val_loader, self.test_dataset = DataProcessorKitti(self.batch_size)

        # Define the NN model
        self.model = SimpleYOLO()
        summary(self.model, input_size=(1, 3, 365, 1220))
        # print(self.model)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            weight_decay=self.reg
        )
        
        # Define the criterion (loss)
        self.criterion = nn.CrossEntropyLoss()
        
        # Move stuff to the given device
        self.model = self.model.to(self.device)
        self.criterion.to(self.device)
        
        # Initialize losses vector and best model storage
        self.train_losses = []
        self._reset()

    def _reset(self):
        '''
        Reset the logger for storing best results
        
        Args:
            None
        
        Returns:
            None
        '''
        
        self.best = 0.0
        self.best_cm = None
        self.best_model = None
    
    def train(self):
        '''
        Train the model
        
        Args:
            None, takes data from the __init__
            
        Returns:
            None, saves models and figs
        '''
        
        # Log start time of training
        train_time_start_overall = time.time()
        
        # Main training loop
        for epoch in range(self.epochs):
            
            # Adjust learning rate (for SGD optimizer. Adam does this automatically)
            self._adjust_learning_rate(epoch)
            
            # Train
            print(f'Training epoch {epoch}')
            self.model.train()
            loss = self.MainLoop(epoch, self.train_loader)
            
            print(f'Validating epoch {epoch}')
            
            # Validate
            self.model.eval()
            acc, cm = self.MainLoop(epoch, self.val_loader)
            
            # Store best model
            if acc > self.best:
                self.best = acc
                self.best_cm = cm
                self.best_model = copy.deepcopy(self.model)
            
            if self.save_best:
                basedir = pathlib.Path(__file__).parent.parent.parent.resolve()
                torch.save(
                    self.best_model.state_dict(),
                    str(basedir) + "/models/" + self.model_type.lower() + '_loss_' + str(round(loss,3)) + ".pt",
                )
                
            # Plot
            self.plot(loss)
            
        # Print training time
        train_time_end = time.time()
        print(f'Train Time: {round(train_time_end-train_time_start_overall,2)} Seconds')
        
        # Save the figs
        fig_name_png = f'figs/loss_{loss:.4f}.png'
        fig_name_eps = f'figs/loss_{loss:.4f}.eps'
        plt.savefig(fig_name_png)
        plt.savefig(fig_name_eps)

    def MainLoop(self, epoch, data_loader):
        '''
        Runs a single pass (train or eval) through the provided data_loader.
        In training mode, updates model parameters. In eval mode, computes accuracy and confusion matrix.

        Args:
            epoch (int): Current epoch number
            data_loader (DataLoader): PyTorch DataLoader for either training or validation set

        Returns:
            If training: Returns the average loss for the epoch
            If not training: Returns the accuracy and confusion matrix for the epoch
        '''

        # Initialize meters for timing, loss, and accuracy
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # Initialize confusion matrix (used during validation)
        num_class = 10
        cm = torch.zeros(num_class, num_class, device=self.device)

        # Determine if we are in training or evaluation mode
        is_training = self.model.training

        for batch_idx, batch_data in enumerate(data_loader):

            # batch_data should be something like a list/tuple of items where each item is (image, label)
            # Extract images and targets
            images = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]

            # Stack images into a single tensor
            images = torch.stack(images).to(self.device)
            # Move targets to the device
            targets = [t.to(self.device) for t in targets]

            # Record start time
            start_batch = time.time()

            # Compute outputs, loss, and accuracy
            out, loss, batch_acc = self.ComputeLossAccUpdateParams(images, targets)

            # Update metrics
            batch_size = out.shape[0]
            losses.update(loss.item(), batch_size)
            acc.update(batch_acc, batch_size)
            iter_time.update(time.time() - start_batch)

            if is_training:
                # Print training status every 10 batches
                if batch_idx % 10 == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})"
                        .format(
                            epoch, batch_idx, len(data_loader),
                            iter_time=iter_time, loss=losses, top1=acc
                        )
                    )
            else:
                # Update confusion matrix if evaluating
                # out is expected to be a class prediction tensor; adjust if needed
                with torch.no_grad():
                    _, preds = torch.max(out, 1)
                    for t, p in zip(torch.cat(targets).view(-1), preds.view(-1)):
                        cm[t.long(), p.long()] += 1

                # Print evaluation status every 10 batches
                if batch_idx % 10 == 0:
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})"
                        .format(
                            epoch, batch_idx, len(data_loader),
                            iter_time=iter_time
                        )
                    )

            # Optionally clear large tensors if needed
            # del out, loss  # Uncomment if you want to ensure memory release after each iteration

        if is_training:
            # Return the average loss during training
            return losses.avg
        else:
            # Compute accuracy per class, print results
            cm_sum = cm.sum(dim=1, keepdim=True)
            # Avoid division by zero
            cm_sum[cm_sum == 0] = 1.0
            cm_norm = cm / cm_sum
            per_cls_acc = cm_norm.diag().detach().cpu().numpy().tolist()

            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

            print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
            return acc.avg, cm

    def ComputeLossAccUpdateParams(self, data, target):
        '''
        Computee the loss, update gradients, and get the output of the model
        
        Args:
            data: input data to model
            target: true labels
            
        Returns:
            output: output of model
            loss: loss value from data
        '''
        
        output = None
        loss = None
        acc = None
        
        # If in training mode, update weights, otherwise do not
        if self.model.training:

            # Call the forward pass on the model. The data model() automatically calls model.forward()
            pred = self.model(data)

            output = pred.view(self.batch_size, 114, 2, 9)

            # Calculate loss
            loss, acc = self.LossCalc(output, target)
            
            # Main backward pass to Update gradients
            self.optimizer.zero_grad()
            loss.backward() # Compute gradients of all the parameters wrt the loss
            self.optimizer.step() # Takes a optimization step
            
        else:
            
            # Do not update gradients
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)

        return output, loss, acc
    
    def LossCalc(self, output, target):
        '''
        Calcualtes the loss for the kitti dataset model output
        '''

        # assumes num of classes is 4 but can be customized using last parameter
        loss, class_accuracy = compute_loss(output, target)

        return loss, class_accuracy
        
    def plot(self, loss):
        '''
        Plot loss live during training
        
        Args:
            loss (int): loss at end of each epoch
            
        Returns:
            None, plots loss over epoch
        '''
        
        self.train_losses.append(float(loss))
        plt.plot(np.arange(1, len(self.train_losses) + 1), self.train_losses, label ='', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.pause(0.0000001)
    
    def _adjust_learning_rate(self, epoch):
        if isinstance(self.optimizer, torch.optim.SGD):
            epoch += 1
            if epoch <= self.warmup:
                lr = self.lr * epoch / self.warmup
            elif epoch > self.steps[1]:
                lr = self.lr * 0.01
            elif epoch > self.steps[0]:
                lr = self.lr * 0.1
            else:
                lr = self.lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count