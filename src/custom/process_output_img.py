from torchvision import transforms
import matplotlib.pyplot as plt

def ProcessOutputImg(img, output, truth):
    '''
    Process the output image by putting the bounding boxes and classifications on the image
    
    Args:
        input tensor (1,3,365,1220)
        output (1,18,6,19) --> here we convert to (114,2,9) where 2 bbox per grid and 9 is (bbox coords, conf, class probabilities) 
        
    Returns:
        output with bounding boxes
    '''

    denormalize = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
    normal_img = denormalize(img)
    
    output = output.reshape(114,2,9)
    
    # Loop over gridboxes in output to find top confidence bboxes
    conf_mask = output[:,:,4]
    
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