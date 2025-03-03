import flwr as fl
from flwr.common import Metrics
from logging import INFO, log, DEBUG
import importlib

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy.aggregate import aggregate

import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

import multiprocessing
import numpy as np
import os
from parameters import CLUSTERING, K1_ALLOCATION
from model import Net, train, test, get_parameters, set_parameters
# Select the appropriate module based on K1_ALLOCATION
if K1_ALLOCATION in ["reversed", "non_reversed"]:
    HierFL_module = importlib.import_module("HierFL_Dynamic_allocation")
elif K1_ALLOCATION == "uniform":
    HierFL_module = importlib.import_module("HierFL_Uniform_allocation")
else:
    raise ValueError(f"Invalid K1_ALLOCATION value: {K1_ALLOCATION}")

# Import the HierFL function dynamically
HierFL = getattr(HierFL_module, "HierFL")
from client import FlowerClient
from strategy import FedAvgCustom
from load_datasets import load_datasets
from utils import generate_random_sizes



        

def main():
    args = {
    'EXPERIMENT': 27,
    'SEED': 1,
    'DATASET': "mnist",
    'DATA_DISTRIBUTION': "NON-IID",
    'TYPE_DISTRIBUTION': "pathological-balanced",
    'NUM_CLIENTS': 50, # 30
    'LOCAL_ITERATIONS': 5, #
    'EDGE_AGGREGATIONS': 10, # k2 10
    'GLOBAL_ROUNDS': 25,
    'TRAIN_PHASES': 5,
    #'NUM_EDGES_SERVERS': 5,
    'ACTUAL_GLOBAL_ROUND': 0,
    'ACTUAL_EDGE_ROUND': 0,
    'EDGE_CLIENTS': 10,
    'CLIENT_FRACTION': 1,#
    'EVALUATE_FRACTION': 1,#
    'MIN_FIT_CLIENT': 50,
    'MIN_EVALUATE_CLIENT': 50,
    'MIN_AVAILABLE_CLIENT': 50,
    'BATCH_SIZE': 20,
    'LEARNING_RATE': 0.01,
    'EXPONENTIAL_DECAY_RATE': 0.995
    }

    # MAKE EXPERIMENTS REPEATABLE
    torch.manual_seed(args['SEED'])
    np.random.seed(args['SEED'])

    trainloaders, valloaders, testloader = load_datasets(args)

    args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Try "cuda" to train on GPU
    print(f"Training on {args['DEVICE']} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    # Specify the resources each of your clients need. 
    # By default, each client will be allocated 1x CPU and 0x GPUs
    args['client_resources'] = {"num_cpus": 1, "num_gpus": 0.0}
    if args['DEVICE'].type == "cuda":
        # here we are assigning an entire GPU for each client.
        args['client_resources'] = {"num_cpus": 1, "num_gpus": 1}

    # Leaf directory
    directory = "Experiment_" + str(args['EXPERIMENT'])
    # Parent Directories
    parent_dir = "Experiments"
    # Path
    dir_path = os.path.join(parent_dir, directory)
    args['dir_path'] = dir_path
    # Create the directory 'ihritik'
    try:
        os.makedirs(dir_path, exist_ok=True)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print("Directory '%s' can not be created")
          

    HierFL(args, trainloaders, valloaders, testloader)
   

    
if __name__ == "__main__":
    print(f"Running experiment with clustering: {CLUSTERING} and k1 allocation: {K1_ALLOCATION}")
    main()
    

