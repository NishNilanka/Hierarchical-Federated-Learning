import random

class EdgeServer:
    def __init__(self, server_id, num_clients):
        self.server_id = server_id
        self.num_clients = num_clients  # Number of clients assigned to this edge server
        self.devices = []  # List to hold the devices (clients) assigned to this edge server
        
        # Add CPU power (in GHz), memory (in GB), and other configurations
        self.cpu_power = random.uniform(1.5, 3.5)  # Random CPU power between 1.5GHz and 3.5GHz
        self.memory = random.randint(4, 32)  # Random memory between 4GB and 32GB
        self.storage = random.randint(50, 500)  # Random storage between 50GB and 500GB

        # Transmitter power for edge-to-cloud communication in watts
        self.transmitter_power = 5.0  # Example: Higher power compared to device-to-edge

    def set_transmitter_power(self, power):
        """
        Update the transmitter power for edge-to-cloud communication.
        """
        self.transmitter_power = power

    def get_transmitter_power(self):
        """
        Get the transmitter power for edge-to-cloud communication.
        """
        return self.transmitter_power

    def assign_devices(self, devices):
        """
        Assign devices (clients) to the edge server.
        """
        self.devices = devices

    def get_devices(self):
        """
        Return the list of devices assigned to this edge server.
        """
        return self.devices

    def update_configuration(self, cpu_power=None, memory=None, storage=None):
        """
        Update the edge server's configuration.
        """
        if cpu_power is not None:
            self.cpu_power = cpu_power
        if memory is not None:
            self.memory = memory
        if storage is not None:
            self.storage = storage

    def get_configuration(self):
        """
        Return the current configuration of the edge server.
        """
        return {
            "CPU Power (GHz)": self.cpu_power,
            "Memory (GB)": self.memory,
            "Storage (GB)": self.storage
        }

    def get_server_id(self):
        return self.server_id

    def get_num_clients(self):
        return len(self.devices)
