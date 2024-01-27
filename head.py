import socket
import threading
import socket
import threading
import argparse
import torch
import io
import struct


class Edge:
    def __init__(self, server_host="127.0.0.1", server_port=40000, edge_port=40001):
        # connect server
        self.edge = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.edge.connect((server_host, server_port))
        # config edge ip and port and listen to the clients
        self.edge_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.edge_client.bind(("127.0.0.1", edge_port))
        self.edge_client.listen()
        #  config training
        # self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.clock = []

    def handle_client(self, conn, addr):
        print(f"New client {addr} connected.")
        client_id = 0
        # self.client_register(addr)
        # self.all_trainsample_num += self.sample_registration[addr]
        while True:
            msg == conn.recv(1024).decode("utf-8")
            if msg != "":
                client_id = int(msg)
                self.id_registration.append(client_id)
                break
        while True:
            # msg = conn.recv(1024)
            # try:
            #     decoded_msg = msg.decode("utf-8")
            #     if decoded_msg == "DISCONNECT" or decoded_msg == "":
            #         print(f"Client {addr} disconnected.")
            #         break

            # except:
            #     print("Received message is not utf-8 encoded.")
            try:
                size = struct.unpack("!I", conn.recv(4))[0]
                state_dict_bytes = b""
                while len(state_dict_bytes) < size:
                    state_dict_bytes += conn.recv(1024)
                    # Load the state_dict from the byte stream
                buffer = io.BytesIO(state_dict_bytes)
                shared_state_dict = torch.load(buffer)
                self.receive_from_client(client_id, shared_state_dict)
                break
            except:
                pass
            # Store the received state_dict in the receiver_buffer
            # self.receiver_buffer = shared_state_dict
            # print(f"New message from {addr}")

            # self.edge.send(msg.encode("utf-8"))  # Forward message to server
        conn.close()

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def send_msg(self, msg):
        self.edge.send(msg.encode("utf-8"))

    def start(self):
        print("Edge Started. Waiting for clients...")
        while True:
            if len(self.id_registration) == 1:
                break
            conn, addr = self.edge_client.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_port", type=int, default=40001, help="Port number for the edge"
    )
    args = parser.parse_args()

    cluster_edge = Edge(edge_port=args.edge_port)
    cluster_edge.send_msg("edge")
    cluster_edge.start()
