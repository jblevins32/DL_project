from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches

def ProcessOutputImg(img, output, label, num_classes):
    '''
    Process the output image by putting the bounding boxes and classifications on the image
    
    Args:
        input tensor (1,3,365,1220)
        output (1,18,6,19) --> here we convert to (114,2,9) where 2 bbox per grid and 9 is (bbox coords, conf, class probabilities) 
        
    Returns:
        shows output img with bounding boxes
    '''
    
    output = output.reshape(228, 5 + num_classes)
    
    # sigmoid or softmax everything
    output[..., 0:5] = torch.sigmoid(output[..., 0:5])
    output[..., 5:5+num_classes] = torch.softmax(output[:, 5:5+num_classes], dim=-1)

    # Find gridboxes in output with some confidence
    conf_level = 0.00001 # This is the min confidence we want our bbox model to be
    conf_mask = output[:,4] > conf_level
    
    # Keep only the confident bboxes and bring them back to correct img size
    resulting_boxes = output[conf_mask]
    resulting_boxes[:,0:4] = resulting_boxes[:,0:4]*torch.tensor([1220,365,1220,365])
    
    # Get predicted class for each bbox
    preds = resulting_boxes[:,:5]
    preds[:,4] = torch.argmax(resulting_boxes[:,5:],dim=1)
    
    # Bring image back to normal state
    denormalize = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
    img = denormalize(img).permute(1,2,0)
    
    fig, ax = plt.subplots(1)

    class_colors = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'purple'
    }

    if num_classes == 5:
        class_colors = {
            0: 'orange',
            1: 'red',
            2: 'green',
            3: 'blue',
            4: 'purple'
        }
     
    # Loop through detected objects in predictions then labels and plot the bboxes
    for obj in preds:
            
        x, y, w, h, clas = [float(val.item()) for val in obj]
        x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
        
        start_x = min(y1, y2)
        start_y = min(x1, x2)
        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        
        rect = mpatches.Rectangle((start_y, start_x), box_w, box_h,
                                fill=False, 
                                edgecolor=class_colors[int(clas)], 
                                linewidth=1.5, 
                                linestyle='-',
                                label="True" if clas == 0 else "_nolegend_") # Use _nolegend_ to avoid repeating in legend
        
        ax.add_patch(rect)
    
    for obj in label:
            
        x, y, w, h, clas = [float(val.item()) for val in obj]
        x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
        
        start_x = min(y1, y2)
        start_y = min(x1, x2)
        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        
        rect = mpatches.Rectangle((start_y, start_x), box_w, box_h,
                                fill=False, 
                                edgecolor=class_colors[int(clas)], 
                                linewidth=1.5, 
                                linestyle='--',
                                label="True" if clas == 0 else "_nolegend_") # Use _nolegend_ to avoid repeating in legend
    
        ax.add_patch(rect)
          
    # Create separate legends for True and Predicted
    true_patch = mpatches.Patch(color='black', label='True Labels (solid)')
    pred_patch = mpatches.Patch(color='black', label='Predictions (dashed)')
    
    # Class legend (just color reference)
    red_patch = mpatches.Patch(color='red', label='Car')
    green_patch = mpatches.Patch(color='green', label='Van')
    blue_patch = mpatches.Patch(color='blue', label='Pedestrian')
    purple_patch = mpatches.Patch(color='purple', label='Cyclist')
    
    plt.legend(handles=[true_patch, pred_patch, red_patch, green_patch, blue_patch, purple_patch],
               loc='upper right')
    plt.axis('off')
    ax.imshow(img)
    plt.show()
    
def ShowResults(img, output):
    '''
    Process the output image by putting the bounding boxes and classifications on the image
    
    Args:
        input tensor (1,3,365,1220)
        output (1,18,6,19)
        
    Returns:
        output with bounding boxes
    '''
    
    # Demormalize the test image
    denormalize = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
    normal_img = denormalize(img).permute(1,2,0)
    img = img.permute(1,2,0)
    
    # Show images
    fig, axes = plt.subplots(2, 1)

    axes[0].imshow(normal_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot the second image
    axes[1].imshow(img)
    axes[1].set_title("Normalized Image")
    axes[1].axis("off")
    
    
    
    plt.show()