class EdgeServer:
    def __init__(self, server_id, num_clients):
        self.server_id = server_id
        self.num_clients = num_clients  # Number of clients assigned to this edge server
        self.drones = []  # List to hold the drones (clients) assigned to this edge server

    def assign_drones(self, drones):
        """
        Assign drones (clients) to the edge server.
        """
        self.drones = drones

    def get_drones(self):
        """
        Return the list of drones assigned to this edge server.
        """
        return self.drones

    def get_server_id(self):
        return self.server_id

    def get_num_clients(self):
        return len(self.drones)
