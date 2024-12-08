'''
Run this script to train a model
'''

from load_args import LoadArgs
from solver import Solver # use this for cifar data
from solver_kitti import SolverKitti # use this for Kitti data

# Load arguments from config file
kwargs = LoadArgs()

# Instantiate training object which loads all model parameters
solver = SolverKitti(**kwargs)

# Train model
solver.train()