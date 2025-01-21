class EdgeServer:
    def __init__(self, server_id, num_clients):
        self.server_id = server_id
        self.num_clients = num_clients  # Number of clients assigned to this edge server
        self.devices = []  # List to hold the devices (clients) assigned to this edge server

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

    def get_server_id(self):
        return self.server_id

    def get_num_clients(self):
        return len(self.devices)
