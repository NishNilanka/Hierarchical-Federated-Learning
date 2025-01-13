from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

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

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


# Define a custom strategy extending FedAvg
class FedAvgCustom(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_weights = None  # To store the final model weights
        self.total_samples = 0
        self.devices_info = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        # parameters_aggregated = parameters_to_ndarrays(parameters_aggregated)
        self.final_weights = parameters_aggregated
        print(f"Paramters Aggregated : {self.final_weights.tensor_type}")

        # Total number of samples
        total_samples = sum([num_examples for _, num_examples in weights_results])
        self.total_samples = total_samples
        print(f"Total samples: {self.total_samples}")

        devices_metrics = [
             fit_res.metrics
            for _, fit_res in results
        ]
        print(devices_metrics)

        for devices in devices_metrics:
            deviceId = devices['deviceId']
            #print(f"deviceId: {deviceId}")
            if deviceId not in self.devices_info:
                self.devices_info[deviceId] = devices
            else:
                self.devices_info[deviceId]['consumedEnergyComputation'] += devices['consumedEnergyComputation']
                self.devices_info[deviceId]['trainTimeComputation'] += devices['trainTimeComputation']
                self.devices_info[deviceId]['consumedEnergyCommunication'] += devices['consumedEnergyCommunication']
                self.devices_info[deviceId]['num_communications'] += devices['num_communications']

        

        for device_info_id in self.devices_info:
            print(f"DeviceId: {device_info_id} - consumedEnergyComputation: {self.devices_info[device_info_id]['consumedEnergyComputation']}")
        #return aggregated_weights, aggregated_metrics
        metrics_aggregated = {}

        return self.final_weights, metrics_aggregated
    
