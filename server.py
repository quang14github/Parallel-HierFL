from options import args_parser
from tensorboardX import SummaryWriter

import socket
import threading


class Server:
    def __init__(self, args):
        # config server ip and port
        self.host = "127.0.0.1"
        self.port = 40000
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        # config training
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        self.train_loaders = {}
        self.test_loaders = {}
        self.v_train_loader = {}
        self.v_test_loader = {}
        # define the edges
        self.edges = []
        self.edge_ports = [40001, 40002]
        # define the clients
        self.clients = []

    def handle_client(self, conn, addr):
        while True:
            msg = conn.recv(1024).decode("utf-8")
            if msg == "DISCONNECT" or msg == "":
                print(f"{addr} disconnected.")
                break
            if msg == "client":
                self.clients.append(addr)
                print(f"New client {addr} connected.")
                # send edge port to the client
                if len(self.clients) == 1:
                    print(
                        f"Redirect client {addr} to edge with port number {self.edge_ports[0]}"
                    )
                    conn.send(str(self.edge_ports[0]).encode("utf-8"))
                else:
                    print(
                        f"Redirect client {addr} to edge with port number {self.edge_ports[1]}"
                    )
                    conn.send(str(self.edge_ports[1]).encode("utf-8"))
                break
            if msg == "edge":
                self.edges.append(addr)
                print(f"New edge {addr} connected.")
            print(f"New message from {addr}: {msg}")
        conn.close()

    def start(self):
        print("Server Started. Waiting for clusters...")
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()


args = args_parser()
server = Server(args)
server.start()
