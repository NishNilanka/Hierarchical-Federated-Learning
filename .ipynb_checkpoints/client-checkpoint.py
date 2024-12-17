import flwr as fl
import torch

from model import Net, train, test, get_parameters, set_parameters, compute_updated_size



class FlowerClient(fl.client.NumPyClient):
    #def __init__(self, net, trainloader, valloader, cid, clusterid: int):
    def __init__(self, net, trainloader, valloader, droneManager, cid):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.droneManager = droneManager
        #self.clusterid = clusterid

        #self.writer = SummaryWriter("runs/exp1/Cluster "+str(self.clusterid)+"/client_"+cid) # nel client al posto di exp1 si potrebbe mettere il cid




    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    


    
    def fit(self, parameters, config):  #set the parameters we received from the server
        # Read values from config
        # server_round = config["server_round"]

        print(f"[Client {self.cid}] Starting fit...")
        hyperparameters = {
            'lr': config['learning_rate'],
            'exp_decay_rate': config['exponential_dacey_rate'],
            'local_iterations': config['local_iterations']
        }

    
        # Save the initial state of the model (before local training)
        initial_state = {key: value.clone() for key, value in self.net.state_dict().items()}
        set_parameters(self.net, parameters)
        localUpdateEnergyComputation, localUpdateTrainTime = train(self.net, self.trainloader, self.droneManager, hyperparameters, verbose=True)

        # Log the start of training
        print(f"[Client {self.cid}] Training started with hyperparameters: {hyperparameters}")
        
        # Save the updated state of the model (after local training)
        updated_state = {key: value.clone() for key, value in self.net.state_dict().items()}

        # Calculate the size in bits of the local updates (assuming 32-bit precision)
        update_size_bits = compute_updated_size(initial_state, updated_state, precision_bits=32)

        print(f"[Client {self.cid}] Local update size: {update_size_bits / 8 / 1e6:.2f} MB ({update_size_bits} bits)")

        # Compute communication energy consumption
        communicationEnergyComputation = self.droneManager.computeEnergyCommunication(update_size_bits)
        print(f"[Client {self.cid}] Communication energy consumed: {communicationEnergyComputation} J")

        # Update the number of communications
        self.droneManager.setNumCommunications(1)
        droneCommunications = self.droneManager.getNumCommunications()

        # Log the end of training
        print(f"[Client {self.cid}] Training completed. Energy consumption: {localUpdateEnergyComputation} J, Training time: {localUpdateTrainTime} s")

        # Return updated parameters and metadata
        return get_parameters(self.net), len(self.trainloader), {
            "droneId": self.droneManager.droneId,
            "consumedEnergyComputation": localUpdateEnergyComputation,
            "trainTimeComputation": localUpdateTrainTime,
            "consumedEnergyCommunication": communicationEnergyComputation,
            "num_communications": droneCommunications,
        }
    
    def evaluate(self, parameters, config):
        """Perform local evaluation."""
        print(f"[Client {self.cid}] Starting evaluation with config: {config}")
        set_parameters(self.net, parameters)
        
        # Perform evaluation
        loss, accuracy = test(self.net, self.valloader)
        
        # Log evaluation results
        print(f"[Client {self.cid}] Evaluation completed. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Return evaluation results
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        
