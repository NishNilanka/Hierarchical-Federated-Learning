import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
import random
import copy
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
from strategy import FedAvgCustom
from model import Net, test, get_parameters, set_parameters
from client import FlowerClient
from device_manager import DeviceManager
from metrics import weighted_average, evaluate
from utils import show_distribution, log_experiment_file


def HierFL(args, trainloaders, valloaders, testloader):

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        print("fit_config function called on configure_fit")
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_iterations": train_args['LOCAL_ITERATIONS'],
            "learning_rate": args['LEARNING_RATE'],
            "exponential_decay_rate": args['EXPONENTIAL_DECAY_RATE'],
        }
        return config

    def client_fn_cluster(cid: str):
        net = Net().to(args['DEVICE'])
        trainloader = trainloaders_cluster[int(cid)]
        valloader = valloaders_cluster[int(cid)]
        deviceManager = copy.deepcopy(clients_cluster[int(cid)])
        return FlowerClient(net, trainloader, valloader, deviceManager, cid).to_client()

    model = Net()
    num_params = sum(p.numel() for p in model.parameters())
    size_bits = num_params * 32  # Assuming 32-bit floating point numbers
    print(f"Model size in bits: {size_bits}")

    name_model_params = args['DATASET'] + '_model_parameters.pth'
    torch.save(model.state_dict(), name_model_params)
    model.load_state_dict(torch.load(name_model_params))
    initialisation_params = get_parameters(model)

    global_results = []
    global_training_time = {}
    global_metrics = {}

    train_args_conf = [
        {
            'LOCAL_ITERATIONS': 5,
            'EDGE_AGGREGATIONS': 10,
        }
    ]

    # Initialize the clients (DeviceManager objects)
    clients = []
    for id in range(args['NUM_CLIENTS']):
        clients.append(DeviceManager(id))

    for train_args in train_args_conf:
        log_experiment_file(args, train_args)
        params = initialisation_params
        phase = 0

        cumulative_statistics = {}
        CompEnergyConsumed = 0.0
        CommEnergyConsumed = 0.0
        trainTime = 0.0
        numCommunications = 0

        for global_round in range(args['GLOBAL_ROUNDS']):
            
            # Change cluster configurations after each 5 global rounds
            if global_round % args['TRAIN_PHASES'] == 0:
                print(f"PHASE {phase+1} - CONFIGURING RANDOM CLUSTERS\n")
                with open(train_args['file_path'], "a") as file:
                    file.write(f"\n----------------------------------------------------------------------------------\n")
                    file.write(f"PHASE {phase+1} - CONFIGURING RANDOM CLUSTERS\n")
                
                # Randomly shuffle clients and partition into clusters
                NUM_CLUSTERS = 5
                NUM_CLIENTS = len(clients)
                cluster_size = NUM_CLIENTS // NUM_CLUSTERS  # Assuming equal cluster sizes
                
                random.shuffle(clients)
                clustered_clients = [clients[i:i + cluster_size] for i in range(0, NUM_CLIENTS, cluster_size)]

                # Distribute remaining clients if NUM_CLIENTS % NUM_CLUSTERS != 0
                if len(clustered_clients) > NUM_CLUSTERS:
                    for idx, leftover_client in enumerate(clustered_clients[NUM_CLUSTERS]):
                        clustered_clients[idx % NUM_CLUSTERS].append(leftover_client)
                    clustered_clients = clustered_clients[:NUM_CLUSTERS]

                # Create cluster_dataloaders for random clusters
                cluster_dataloaders = {}
                for cluster_id, cluster_group in enumerate(clustered_clients, start=1):
                    selected_trainloaders = [trainloaders[device.deviceId] for device in cluster_group]
                    selected_valloaders = [valloaders[device.deviceId] for device in cluster_group]

                    cluster_dataloaders[cluster_id] = {
                        'train': selected_trainloaders,
                        'validation': selected_valloaders,
                        'clients': cluster_group,
                    }

                phase += 1

            # Start the Federated Learning process for each cluster
            print(f"GLOBAL ROUND {global_round+1} is running!")
            CompEnergyConsumedRound = 0.0
            CommEnergyConsumedRound = 0.0
            trainTimeRound = 0.0
            numCommunicationsRound = 0

            for edge in range(len(cluster_dataloaders)):
                trainloaders_cluster = cluster_dataloaders[edge+1]['train']
                valloaders_cluster = cluster_dataloaders[edge+1]['validation']
                clients_cluster = cluster_dataloaders[edge+1]['clients']
                num_clients_cluster = len(trainloaders_cluster)

                strategy_cluster = FedAvgCustom(
                    fraction_fit=args['CLIENT_FRACTION'],  # C fraction, meaning 10% of clients are selected each round
                    fraction_evaluate=args['EVALUATE_FRACTION'],
                    initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                )

                # Run simulation for each cluster
                fl.simulation.start_simulation(
                    client_fn=client_fn_cluster,
                    num_clients=num_clients_cluster,  # Number of clients in cluster
                    config=fl.server.ServerConfig(num_rounds=train_args['EDGE_AGGREGATIONS']),
                    strategy=strategy_cluster,
                    client_resources=args['client_resources'],
                )

                final_weights_cluster = strategy_cluster.final_weights
                total_samples = strategy_cluster.total_samples
                devices_info = strategy_cluster.devices_info
                global_results.append((parameters_to_ndarrays(final_weights_cluster), total_samples))

                # Update energy and communication statistics
                for deviceid in devices_info:
                    deviceId = devices_info[deviceid]['deviceId']
                    consumedEnergyComputation = devices_info[deviceid]['consumedEnergyComputation']
                    CompEnergyConsumedRound += consumedEnergyComputation
                    consumedEnergyCommunication = devices_info[deviceid]['consumedEnergyCommunication']
                    CommEnergyConsumedRound += consumedEnergyCommunication
                    clients[deviceId].decreaseEnergyLevel(consumedEnergyComputation)
                    clients[deviceId].decreaseEnergyLevel(consumedEnergyCommunication)
                    numCommunicationsRound += devices_info[deviceid]['num_communications']
                    trainTimeRound += devices_info[deviceid]['trainTimeComputation']

            # Aggregate global results
            global_parameters_aggregated = ndarrays_to_parameters(aggregate(global_results))
            global_results.clear()
            global_net = Net().to(args['DEVICE'])
            global_parameters_ndarrays = parameters_to_ndarrays(global_parameters_aggregated)
            params = global_parameters_ndarrays
            set_parameters(global_net, global_parameters_ndarrays)  # Update model with the latest parameters
            global_loss, global_accuracy = test(global_net, testloader)

            with open(train_args['file_path'], "a") as file:
                file.write(f"\n\nGLOBAL TEST ROUND: {global_round+1} Evaluation -")
                file.write(f"Loss: {global_loss} \tAccuracy: {global_accuracy} \t")
                file.write(f"Training time: {trainTimeRound}s\n")
                file.write(f"Energy Computation: {CompEnergyConsumedRound} \tEnergy Communication: {CommEnergyConsumedRound} \tNumber of communications: {numCommunicationsRound} ")

            CompEnergyConsumed += CompEnergyConsumedRound
            CommEnergyConsumed += CommEnergyConsumedRound
            trainTime += trainTimeRound
            numCommunications += numCommunicationsRound

            cumulative_statistics[global_round+1] = {
                "Loss": global_loss,
                "Accuracy": global_accuracy,
                "Training time": trainTime,
                "Computation Energy": CompEnergyConsumed,
                "Communication Energy": CommEnergyConsumed,
                "Total Energy": CompEnergyConsumed + CommEnergyConsumed,
                "Communications": numCommunications,
            }

        with open(train_args['file_path'], "a") as file:
            file.write(f"\n\nSUMMARY EXPERIMENTS-\n")
            for round_num in cumulative_statistics:
                file.write(f"Loss: {cumulative_statistics[round_num]['Loss']} \tAccuracy: {cumulative_statistics[round_num]['Accuracy']} \t")
                file.write(f"Training time: {cumulative_statistics[round_num]['Training time']}s\t")
                file.write(f"Energy Computation: {cumulative_statistics[round_num]['Computation Energy']} \t \
                            Energy Communication: {cumulative_statistics[round_num]['Communication Energy']} \t \
                            Total Energy: {cumulative_statistics[round_num]['Total Energy']} \t \
                            Number of communications: {cumulative_statistics[round_num]['Communications']}\n ")

        # Device statistics
        with open(train_args['file_path'], "a") as file:
            file.write(f"\n\nSUMMARY DEVICE STATISTICS-\n")
            for device in clients:
                file.write(f"\nDevice {device.deviceId} -\t Battery Level: {device.actual_batteryLevel_percentage}% -\t Communications with base stations: {device.num_communications} ")

