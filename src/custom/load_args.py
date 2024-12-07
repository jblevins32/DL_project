import yaml
import torch

def LoadArgs():
    '''
    Unpacking parameters from config yaml file to kwargs dictionary. Kwargs allows for a function to accept any number of arguments
    '''
    # Choose calculation device - Use cpu if CUDA gpu not available
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device = " + device)

    # Unload config to keyword arguments
    with open("config.yaml", "r") as read_file:
        config = yaml.safe_load(read_file)

    kwargs = {k: v for section in config for k, v in config[section].items() if k != 'description'}
    kwargs['device'] = device

    print("Configurations:", kwargs)

    return kwargs