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
        output with bounding boxes
    '''
    
    output = output.reshape(228,9)
    
    # sigmoid or softmax everything
    output[..., 0:5] = torch.sigmoid(output[..., 0:5])
    output[..., 5:5+num_classes] = torch.softmax(output[:, 5:5+num_classes], dim=-1)

    # Find gridboxes in output with some confidence
    conf_level = 0.8 # This is the min confidence we want our bbox model to be
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
    img_copy = img.clone()
    
    # Loop through detected objects in predictions then labels and plot the bboxes
    for obj_list in [preds, label]:
        for obj in obj_list:
            x,y,w,h,clas = [int(val) for val in torch.round(obj)]

            if clas == 0: # red
                RGB = torch.tensor([1,0,0])
            elif clas == 1: # green
                RGB = torch.tensor([0,1,0])
            elif clas == 2: # blue
                RGB = torch.tensor([0,0,1])
            else: # purple
                RGB = torch.tensor([.5,0,.5])
            
            if y>h:
                start_x = h
                end_x = y
            else:
                start_x = y
                end_x = h
            if x>w:
                start_y = w
                end_y = x
            else:
                start_y = x
                end_y = w
            
            img[start_x:end_x,start_y:end_y,:] = RGB
            
            bw = 5 # border width
            img[start_x+bw:end_x-bw,start_y+bw:end_y-bw,:] = img_copy[start_x+bw:end_x-bw,start_y+bw:end_y-bw,:]
            
            # ensure future bounding boxes aren't overwritten by the original img
            img_copy = img.clone()
        
        
    # After plotting your boxes and before `plt.show()`:
    red_patch = mpatches.Patch(color='red', label='Car')
    green_patch = mpatches.Patch(color='green', label='Van')
    blue_patch = mpatches.Patch(color='blue', label='Pedestrian')
    purple_patch = mpatches.Patch(color='purple', label='Cyclist')

    plt.legend(handles=[red_patch, green_patch, blue_patch, purple_patch])

    plt.imshow(img)            
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