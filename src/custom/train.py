'''
Run this script to train a model
'''

import yaml
from solver import Solver
from solver_kitti import SolverKitti
import torch

# Choose calculation device - Use cpu if CUDA gpu not available
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device = " + device)

# Unpacking parameters from config yaml file to kwargs dictionary. Kwargs allows for a function to accept any number of arguments
with open("config.yaml", "r") as read_file:
  config = yaml.safe_load(read_file)

kwargs = {k: v for section in config for k, v in config[section].items() if k != 'description'}

kwargs['device'] = device
print("Configurations:", kwargs)

# Instantiate training object which loads all model parameters
solver = SolverKitti(**kwargs)

# Train model
solver.train()