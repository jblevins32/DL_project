import os.path
from math import floor
import numpy as np
import torch.nn as nn
import time
import torch
import copy
import matplotlib.pyplot as plt
from data_processing_cifar import DataProcessorCIFAR
from data_processing_kitti import DataProcessorKitti
from evalutation import compute_loss
from SimpleYOLO import SimpleYOLO
from MidYOLO import MidYOLO
from torchinfo import summary
from datetime import datetime
from globals import root_directory
from process_output_img import DrawBBox

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
        self.model_type = kwargs.pop("model_type", "simpleYOLO")
        self.data_type = kwargs.pop("data_type", "cifar")
        self.training_split_percentage = kwargs.pop("training_split_percentage", 0.8)
        self.dataset_percentage = kwargs.pop("dataset_percentage", 1.0)
        self.save_delay_percent = kwargs.pop("save_delay_percent", 0.1)
        self.num_classes = kwargs.pop("num_classes", 4)
        
        # Define the data
        if self.data_type == "cifar":
            self.train_loader, self.val_loader, self.test_dataset = DataProcessorCIFAR(self.batch_size)
        elif self.data_type == "kitti": 
            self.train_loader, self.val_loader, self.test_dataset = DataProcessorKitti(self.batch_size,
                                                                                       self.training_split_percentage,
                                                                                       self.dataset_percentage,
                                                                                       num_classes=self.num_classes)

        # Define the NN model
        if self.model_type == "simpleYOLO":
            self.model = SimpleYOLO(num_classes=self.num_classes)
        elif self.model_type == "midYOLO":
            self.model = MidYOLO(num_classes=self.num_classes)

        summary(self.model, input_size=(self.batch_size, 3, 365, 1220))

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
        self.bbox_losses = []
        self.conf_losses = []
        self.backgnd_losses = []
        self.cls_losses = []
        self.f1_scores = []

        self.previous_best_model_path = None

        self._reset()

    def _reset(self):
        '''
        Reset the logger for storing best results
        
        Args:
            None
        
        Returns:
            None
        '''
        
        self.best_loss = 0.0
        self.best_f1 = 0.0
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

        # Build directories for models and figs
        model_dir = os.path.join(root_directory, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        root_fig_dir = os.path.join(root_directory, "figs")
        if not os.path.exists(root_fig_dir):
            os.makedirs(root_fig_dir)

        # Format the time
        current_time = datetime.now()
        formatted_time = current_time.strftime("%d_%m-%H:%M:%S")
        specific_model_dir = os.path.join(model_dir, formatted_time)
        specific_fig_dir = os.path.join(root_fig_dir, formatted_time)

        # Main training loop
        for epoch in range(self.epochs):
            
            epoch_start_time = time.time()
            
            # Adjust learning rate (for SGD optimizer. Adam does this automatically)
            self._adjust_learning_rate(epoch)
            
            # Train
            print(f'Training epoch {epoch}')
            self.model.train()
            loss, specific_losses = self.MainLoop(epoch, self.train_loader) # run the main loop of training to get the loss
            
            # Validate
            print(f'Validating epoch {epoch}')
            self.model.eval()
            f1_score = self.MainLoop(epoch, self.val_loader)

            # Store best model
            if epoch == 0:
                self.best_loss = loss
                self.best_model = copy.deepcopy(self.model)

            if loss < self.best_loss: # was accuracy but accuracy is not functional yet
                self.best_loss = loss
                self.best_model = copy.deepcopy(self.model)

                # Only save model beyond a certain percent of epochs
                if epoch >= floor(self.epochs * self.save_delay_percent):

                    if not os.path.exists(specific_model_dir):
                        os.makedirs(specific_model_dir)

                    loss_string = '_loss_' + str(round(loss, 3))
                    f1_string = '_f1_' + f"{f1_score:.4f}"
                    epoch_string = "_epoch_" + str(epoch)

                    model_name = self.model_type.lower() + loss_string + f1_string + epoch_string + ".pt"
                    model_path = os.path.join(specific_model_dir, model_name)

                    torch.save(self.best_model.state_dict(), model_path)

                    # delete last best model
                    if self.previous_best_model_path != None and os.path.exists(self.previous_best_model_path):
                        os.remove(self.previous_best_model_path)

                    # update last best model path
                    self.previous_best_model_path = model_path

            # Plot
            self.PlotAndSave(loss, specific_losses, specific_fig_dir, f1_score)
            
            print(f'Epoch {epoch} took {round(time.time()-epoch_start_time,2)} seconds')
            
        # Print training time
        print(f'Train Time: {round(time.time()-train_time_start_overall,2)} seconds')

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
        f1_score = AverageMeter()
        bboxLosses = AverageMeter()
        confidenceLosses = AverageMeter()
        backgroundLosses = AverageMeter()
        classScoreLosses = AverageMeter()

        # Determine if we are in training or evaluation mode
        is_training = self.model.training

        for batch_idx, batch_data in enumerate(data_loader):

            # batch_data should be something like a list/tuple of items where each item is (image, label)
            # Extract images and targets from batch data
            images = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]

            # Stack images into a single tensor to our desired shape of (batch_size,in-channels,height,width)
            images = torch.stack(images).to(self.device)
            # Move targets to the device
            targets = [t.to(self.device) for t in targets]
            
            # Uncomment below if you want to view current image and true labels for debugging purposes
            # images = images.squeeze(0).permute(1,2,0)
            # DrawBBox(images, targets[0], self.num_classes)

            # Record start time
            start_batch = time.time()

            # Compute outputs, losses, and f1 score
            out, loss, batch_f1Score, specific_losses = self.ComputeLossAccUpdateParams(images, targets)
            
            bboxLoss, confidenceLoss, backgroundLoss, classScoreLoss = specific_losses

            # Update metrics
            batch_size = out.shape[0]
            losses.update(loss.item(), batch_size)
            f1_score.update(batch_f1Score, batch_size)
            iter_time.update(time.time() - start_batch)
            bboxLosses.update(bboxLoss.item(), batch_size)
            confidenceLosses.update(confidenceLoss.item(), batch_size)
            backgroundLosses.update(backgroundLoss.item(), batch_size)
            classScoreLosses.update(classScoreLoss.item(), batch_size)

            if is_training:
                # Print training status every 10 batches
                if batch_idx % 10 == 0:
                    print(
                        "Epoch: [{0}/{1}][{2}/{3}] | "
                        "Time {iter_time.val:.2f} ({iter_time.avg:.2f}) | "
                        "Loss {loss.val:.2f} ({loss.avg:.2f}) | "
                        "bboxL {bboxLoss.val:.2f} ({bboxLoss.avg:.2f}) | "
                        "confL {confidenceLoss.val:.2f} ({confidenceLoss.avg:.2f}) | "
                        "backgndL {backgroundLoss.val:.2f} ({backgroundLoss.avg:.2f}) | "
                        "clsL {classScoreLoss.val:.2f} ({classScoreLoss.avg:.2f}) | "
                        "F1 {top1.val:.2f} ({top1.avg:.2f})"
                        .format(
                            epoch, self.epochs, batch_idx, len(data_loader),
                            iter_time=iter_time, loss=losses, top1=f1_score, bboxLoss = bboxLosses, confidenceLoss = confidenceLosses, backgroundLoss = backgroundLosses, classScoreLoss = classScoreLosses
                        )
                    )
            else:
                # Update confusion matrix if evaluating
                # out is expected to be a class prediction tensor; adjust if needed
                # with torch.no_grad():
                #     _, preds = torch.max(out, 1)
                #     for t, p in zip(torch.cat(targets).view(-1), preds.view(-1)):
                #         cm[t.long(), p.long()] += 1

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
            return losses.avg, (bboxLosses.avg, confidenceLosses.avg, backgroundLosses.avg, classScoreLosses.avg)
        else:
            # # Compute accuracy per class, print results
            # cm_sum = cm.sum(dim=1, keepdim=True)
            # # Avoid division by zero
            # cm_sum[cm_sum == 0] = 1.0
            # cm_norm = cm / cm_sum
            # per_cls_acc = cm_norm.diag().detach().cpu().numpy().tolist()
            #
            # for i, acc_i in enumerate(per_cls_acc):
            #     print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

            print("* F1_Score: {top1.avg:.4f}".format(top1=f1_score))
            return f1_score.avg

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
        precision = None

        num_anchors = 2
        bbox_coords = 4
        conf_measure = 1

        # If in training mode, update weights, otherwise do not
        if self.model.training:

            # Call the forward pass on the model. The data model() automatically calls model.forward()
            pred = self.model(data)

            # Reshape output
            output = pred.view(self.batch_size, 114, num_anchors, bbox_coords + conf_measure + self.num_classes)

            # Calculate loss
            loss, f1_score, specific_losses = compute_loss(output, target, self.num_classes)
            
            # Main backward pass to Update gradients
            self.optimizer.zero_grad()
            loss.backward() # Compute gradients of all the parameters wrt the loss
            self.optimizer.step() # Takes a optimization step
            
        else:
            
            # Do not update gradients
            with torch.no_grad():
                pred = self.model(data)
                output = pred.view(self.batch_size, 114, 2, 5 + self.num_classes)
                loss, f1_score, specific_losses = compute_loss(output, target, self.num_classes)

        return output, loss, f1_score, specific_losses
        
    def PlotAndSave(self, loss, specific_losses, specific_fig_dir, f1_score_avg):
        '''
        Plot loss live during training
        
        Args:
            loss (int): loss at end of each epoch
            
        Returns:
            None, plots loss over epoch
        '''

        if not os.path.exists(specific_fig_dir):
            os.makedirs(specific_fig_dir)
        
        self.train_losses.append(float(loss))
        self.bbox_losses.append(float(specific_losses[0]))
        self.conf_losses.append(float(specific_losses[1]))
        self.backgnd_losses.append(float(specific_losses[2]))
        self.cls_losses.append(float(specific_losses[3]))
        self.f1_scores.append(float(f1_score_avg))
        
        x_plot = np.arange(1, len(self.train_losses) + 1)
        
        # Loss plot
        plt.figure()
        plt.plot(x_plot, self.train_losses, label='Total Loss', color='blue')
        plt.plot(x_plot, self.bbox_losses, label='BBox Loss', color='black')
        plt.plot(x_plot, self.conf_losses, label='Confidence Loss', color='green')
        plt.plot(x_plot, self.backgnd_losses, label='Background Loss', color='purple')
        plt.plot(x_plot, self.cls_losses, label='Classification Loss', color='yellow')        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Add a legend showing what each line represents only on first iteration
        plt.legend() 
                        
        plt.pause(0.0000001)
        
        fig_png_path = os.path.join(specific_fig_dir, "loss.png")
        fig_name_eps = os.path.join(specific_fig_dir, "loss.eps")
        plt.savefig(fig_png_path)
        plt.savefig(fig_name_eps)
        
        # Precision plot
        plt.figure()
        plt.plot(x_plot, self.f1_scores, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        
        plt.pause(0.0000001)
        
        fig_png_path = os.path.join(specific_fig_dir, "precision.png")
        fig_name_eps = os.path.join(specific_fig_dir, "precision.eps")
        plt.savefig(fig_png_path)
        plt.savefig(fig_name_eps)
            
    
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