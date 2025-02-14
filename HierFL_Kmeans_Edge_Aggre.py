import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
import random
import numpy as np
import copy
from sklearn.cluster import KMeans
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
from EdgeServer import EdgeServer
from sklearn.preprocessing import StandardScaler


def HierFL(args, trainloaders, valloaders, testloader):
    """
    Hierarchical Federated Learning with Edge Servers and K-means clustering of clients (devices).
    """

    def fit_config(server_round: int):
        """Return training configuration for each round."""
        config = {
            "server_round": server_round,
            "local_iterations": train_args['LOCAL_ITERATIONS'],
            "learning_rate": args['LEARNING_RATE'],
            "exponential_decay_rate": args['EXPONENTIAL_DECAY_RATE'],
        }
        return config

    def client_fn_cluster(cid: str):
        """Define the client setup for each cluster."""
        net = Net().to(args['DEVICE'])
        trainloader = trainloaders_cluster[int(cid)]
        valloader = valloaders_cluster[int(cid)]
        deviceManager = copy.deepcopy(clients_cluster[int(cid)])
        return FlowerClient(net, trainloader, valloader, deviceManager, cid).to_client()

    # Initialize the model
    model = Net()
    num_params = sum(p.numel() for p in model.parameters())
    size_bits = num_params * 32  # Assuming 32-bit floating point numbers
    print(f"Model size in bits: {size_bits}")

    # Save the initial model state and reload it
    name_model_params = args['DATASET'] + '_model_parameters.pth'
    torch.save(model.state_dict(), name_model_params)
    model.load_state_dict(torch.load(name_model_params))
    initialisation_params = get_parameters(model)

    global_results = []
    global_training_time = {}
    global_metrics = {}

    # Configuration for local iterations and edge aggregations
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

    # Create edge servers (5 edge servers for 5 clusters)
    num_edge_servers = 5
    edge_servers = [EdgeServer(server_id=i, num_clients=0) for i in range(num_edge_servers)]

    for train_args in train_args_conf:
        log_experiment_file(args, train_args)
        params = initialisation_params
        phase = 0

        cumulative_statistics = {}
        CompEnergyConsumed = 0.0
        CommEnergyConsumed = 0.0
        trainTime = 0.0
        numCommunications = 0

        # Federated learning rounds
        for global_round in range(args['GLOBAL_ROUNDS']):
            # Change cluster configurations after each global round
            if global_round % args['TRAIN_PHASES'] == 0:
                print(f"PHASE {phase+1} - CONFIGURING K-MEANS CLUSTERS\n")
                with open(train_args['file_path'], "a") as file:
                    file.write(f"\n----------------------------------------------------------------------------------\n")
                    file.write(f"PHASE {phase+1} - CONFIGURING K-MEANS CLUSTERS\n")

                    # Perform K-means clustering on clients based on their features
                    NUM_CLUSTERS = 5  # Number of clusters
                    client_features = np.array([[client.energy_comp_sample, client.train_time_sample] for client in clients])

                    # Normalize client features for better clustering
                    scaler = StandardScaler()
                    client_features = scaler.fit_transform(client_features)

                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
                    kmeans.fit(client_features)
                    labels = kmeans.labels_

                    # Group clients based on their cluster assignment
                    clustered_clients = {i: [] for i in range(NUM_CLUSTERS)}
                    for i, label in enumerate(labels):
                        clustered_clients[label].append(clients[i])

                    # Log clustering statistics
                    for cluster_id, devices in clustered_clients.items():
                        avg_energy = np.mean([device.energy_comp_sample for device in devices])
                        avg_time = np.mean([device.train_time_sample for device in devices])
                        file.write(f"Cluster {cluster_id}: Avg Energy: {avg_energy:.6f}, Avg Time: {avg_time:.6f}\n")
                        for device in devices:
                            file.write(f"    Device {device.deviceId} - Energy Comp Sample: {device.energy_comp_sample}, Train Time Sample: {device.train_time_sample}, Battery Level: {device.getEnergyLevel()}%\n")

                    # Assign clusters to edge servers and log assignments
                    edge_server_idx = 0
                    for cluster_id, cluster_devices in clustered_clients.items():
                        edge_server = edge_servers[edge_server_idx]
                        edge_server.assign_devices(cluster_devices)
                        print(f"Edge Server {edge_server.get_server_id()} assigned to Cluster {cluster_id} with {len(cluster_devices)} clients.")
                        file.write(f"Edge Server {edge_server.get_server_id()} assigned to Cluster {cluster_id} with {len(cluster_devices)} clients.\n")
                        for device in cluster_devices:
                            file.write(f"    Device {device.deviceId} - Battery Level: {device.getEnergyLevel()}%\n")
                        edge_server_idx = (edge_server_idx + 1) % num_edge_servers

                phase += 1

            # Start the Federated Learning process for each edge server
            print(f"GLOBAL ROUND {global_round+1} is running!")
            CompEnergyConsumedRound = 0.0
            CommEnergyConsumedRound = 0.0
            trainTimeRound = 0.0
            numCommunicationsRound = 0

            # Process each edge server
            for edge_server in edge_servers:
                print(f"Running edge aggregation for Edge Server {edge_server.get_server_id()} with {edge_server.get_num_clients()} clients.")
                trainloaders_cluster = [trainloaders[client.deviceId] for client in edge_server.get_devices()]
                valloaders_cluster = [valloaders[client.deviceId] for client in edge_server.get_devices()]
                clients_cluster = edge_server.get_devices()

                strategy_cluster = FedAvgCustom(
                    fraction_fit=args['CLIENT_FRACTION'],  # Fraction of clients selected for training
                    fraction_evaluate=args['EVALUATE_FRACTION'],
                    initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                )

                # Run simulation for each cluster
                fl.simulation.start_simulation(
                    client_fn=client_fn_cluster,
                    num_clients=len(trainloaders_cluster),
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
                    devoceId = devices_info[deviceid]['deviceId']
                    consumedEnergyComputation = devices_info[deviceid]['consumedEnergyComputation']
                    CompEnergyConsumedRound += consumedEnergyComputation
                    consumedEnergyCommunication = devices_info[deviceid]['consumedEnergyCommunication']
                    CommEnergyConsumedRound += consumedEnergyCommunication
                    clients[devoceId].decreaseEnergyLevel(consumedEnergyComputation)
                    clients[devoceId].decreaseEnergyLevel(consumedEnergyCommunication)
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

            # Update cumulative statistics
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

        # Log the final statistics for the experiment
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
            file.write(f"\n\nSUMMARY Device STATISTICS-\n")
            for device in clients:
                file.write(f"\nDevice {device.deviceId} -\t Battery Level: {device.actual_batteryLevel_percentage}% -\t Communications with base stations: {device.num_communications} ")