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
        self.model = MyModel(self.model_type, self.batch_size)
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
        This function generically runs data through the model, for training, eval, or testing
        
        Args:
            model
            data
            
        Returns:
            
        '''
        
        # Initialize a meter for printing info to terminal
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        
        # Validation stuff
        num_class = 10
        cm = torch.zeros(num_class, num_class)

        # Train on training data by batch. Note: enumerate() provides both the idx and data
        for idx, data in enumerate(data_loader):
            
            # Parsing data from enumerated data
            data_tensors = [item[0] for item in data]
            target = [item[1] for item in data]
            data = torch.stack(data_tensors)
                
            # Log start time of this batch training
            start_batch = time.time()
            
            # Gather data to be trained on the chosen device
            data = data.to(self.device)
            for idx, label in enumerate(target):
                target[idx] = label.to(self.device)

            # Get loss, accuracy, and update the model
            out, loss, batch_acc = self.ComputeLossAccUpdateParams(data, target)
            
            # Update and print the average loss and accuracy for this epoch
            losses.update(loss.item(), out.shape[0]) # .item() extracts the floating number from the tensor type
            acc.update(batch_acc, out.shape[0])
            iter_time.update(time.time() - start_batch)
                
            if self.model.training:
                if idx % 10 == 0:
                    print(
                        (
                            "Epoch: [{0}][{1}/{2}]\t"
                            "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                            "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                        ).format(
                            epoch,
                            idx,
                            len(data_loader),
                            iter_time=iter_time,
                            loss=losses,
                            top1=acc,
                        )
                    )
                
            else:
                # update confusion matrix
                _, preds = torch.max(out, 1)
                for t, p in zip(target.view(-1), preds.view(-1)):
                    cm[t.long(), p.long()] += 1

                if idx % 10 == 0:
                    print(
                        (
                            "Epoch: [{0}][{1}/{2}]\t"
                            "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        ).format(
                            epoch,
                            idx,
                            len(data_loader),
                            iter_time=iter_time,
                            loss=losses,
                            top1=acc,
                        )
                    )
        if self.model.training:
            return losses.avg
        
        else:
            cm = cm / cm.sum(1)
            per_cls_acc = cm.diag().detach().numpy().tolist()
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
        
        # If in training mode, update weights, otherwise do not
        if self.model.training:

            # Call the forward pass on the model. The data model() automatically calls model.forward()
            output = self.model(data)

            # Calculate loss
            loss = self.LossCalc(output, target)
            
            # Main backward pass to Update gradients
            self.optimizer.zero_grad()
            loss.backward() # Compute gradients of all the parameters wrt the loss
            self.optimizer.step() # Takes a optimization step
            
        else:
            
            # Do not update gradients
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)

        # Calculate accuracy
        batch_size = target.shape[0]
        _, pred = torch.max(output, dim=-1) # Finding the class with the highest score
        correct = pred.eq(target).sum() * 1.0 # Count the number of correct predictions
        acc = correct / batch_size

        return output, loss, acc
    
    def LossCalc(self, output, target):
        '''
        Calcualtes the loss for the kitti dataset model output
        '''
        
        loss = self.criterion(output,target)
        
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