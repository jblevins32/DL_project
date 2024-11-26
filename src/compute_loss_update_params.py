import torch

def ComputeLossAccUpdateParams(data, target, model, criterion, optimizer = None):
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
    if model.training:

        # Call the forward pass on the model. The data model() automatically calls model.forward()
        output = model(data)

        # Calculate loss
        loss = criterion(output,target)
        
        # Main backward pass to Update gradients
        optimizer.zero_grad()
        loss.backward() # Compute gradients of all the parameters wrt the loss
        optimizer.step() # Takes a optimization step
        
    else:
        
        # Do not update gradients
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

    # Calculate accuracy
    batch_size = target.shape[0]
    _, pred = torch.max(output, dim=-1) # Finding the class with the highest score
    correct = pred.eq(target).sum() * 1.0 # Count the number of correct predictions
    acc = correct / batch_size

    return output, loss, acc

