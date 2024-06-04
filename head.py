from options import args_parser
import socket
import threading
import torch
import io
import struct
from average import average_weights
import pickle


class Edge:
    def __init__(self, args, server_host="127.0.0.1", server_port=40000):
        # connect server
        self.edge_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.edge_server.connect((server_host, server_port))
        # config edge ip and port and listen to the clients
        self.edge_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.edge_client.bind(("127.0.0.1", args.edge_port))
        self.edge_client.listen()
        #  config training
        # self.cids = cids
        self.start_training = False
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.edge_trainsample_num = 0
        self.total_sample = 0
        self.loss = 0.0
        self.correct = 0.0
        self.total = 0.0
        self.aggregated_metrics = [
            (0.0, 0.0, 0.0) for i in range(args.num_edge_aggregation)
        ]
        self.clock = []
        self.received_clients = {}
        self.client_conns = {}
        self.num_local_update = 0
        self.socket_volumn = args.socket_volumn

    def handle_client(self, condition, conn, addr):
        print(f"New client {addr} connected.")
        client_id = 0
        while True:
            msg = conn.recv(1024).decode("utf-8")
            if msg == "DISCONNECT" or msg == "":
                print(f"{addr} disconnected.")
                break
            if msg != "":
                client_id = int(msg.split(" ")[0])
                client_trainsample_num = int(msg.split(" ")[1])
                print(f"Client {client_id} connected.")
                self.client_register(conn, client_id, client_trainsample_num)
                break
        with condition:
            condition.wait()
        self.send_msg_to_client(client_id, "start")
        while True:
            with condition:
                condition.wait()
            if self.start_training == False:
                break
            self.receive_data_from_client(client_id, conn)
            print(f"Received data from client {client_id}.")

        print(f"Client {client_id} disconnected.")
        conn.close()

    def send_msg_to_client(self, client_id, msg):
        self.client_conns[client_id].send(msg.encode("utf-8"))

    def send_msg_to_server(self, msg):
        self.edge_server.send(msg.encode("utf-8"))

    def receive_msg_from_server(self):
        return self.edge_server.recv(1024).decode("utf-8")

    def client_register(self, conn, client_id, client_trainsample_num):
        self.received_clients[client_id] = 0
        self.client_conns[client_id] = conn
        self.id_registration.append(client_id)
        self.sample_registration[client_id] = client_trainsample_num
        self.edge_trainsample_num += client_trainsample_num
        return None

    def send_data_to_client(self, client_id, conn):
        # Serialize the state_dict to a byte stream
        buffer = io.BytesIO()
        torch.save(self.shared_state_dict, buffer)

        # Get the byte stream from the buffer
        state_dict_bytes = buffer.getvalue()

        # Send the size of the state_dict_bytes before sending state_dict_bytes
        size = len(state_dict_bytes)
        conn.sendall(struct.pack("!I", size))
        conn.sendall(state_dict_bytes)
        conn.sendall(str(self.num_local_update).encode("utf-8"))
        return None

    def receive_data_from_client(self, client_id, conn):
        while True:
            try:
                # Receive the size of the state_dict_bytes before receiving state_dict_bytes
                state_dict_size = struct.unpack("!I", conn.recv(4))[0]
                state_dict_bytes = b""
                while len(state_dict_bytes) < state_dict_size:
                    msg = conn.recv(
                        min(
                            self.socket_volumn,
                            state_dict_size - len(state_dict_bytes),
                        )
                    )
                    state_dict_bytes += msg
                # Load the state_dict from the byte stream
                buffer = io.BytesIO(state_dict_bytes)
                shared_state_dict = torch.load(buffer)
                # Store the state_dict in the receiver_buffer
                self.receiver_buffer[client_id] = shared_state_dict
                print(f"Received state_dict from client {client_id}")
                # Receive the loss, correct and total from the client
                client_loss, client_correct, client_total = map(
                    float, conn.recv(1024).decode("utf-8").split(" ")
                )
                print(
                    f"Received metrics from client {client_id}: loss {client_loss} {client_correct} {client_total}."
                )
                self.loss += (
                    client_loss * self.sample_registration[client_id]
                ) / self.total_sample
                self.correct += client_correct
                self.total += client_total
                self.received_clients[client_id] = 1
                break
            except:
                pass
        return None

    def close_client_conn(self, client_id):
        self.client_conns[client_id].close()
        return None

    def send_data_to_server(self):
        # Serialize the state_dict to a byte stream
        buffer = io.BytesIO()
        torch.save(self.shared_state_dict, buffer)

        # Get the byte stream from the buffer
        state_dict_bytes = buffer.getvalue()

        # Send the size of the state_dict_bytes before sending state_dict_bytes
        size = len(state_dict_bytes)

        aggregated_metrics_bytes = pickle.dumps(self.aggregated_metrics)
        aggregated_metrics_size = len(aggregated_metrics_bytes)

        self.edge_server.sendall(struct.pack("!I", size))
        self.edge_server.sendall(state_dict_bytes)
        self.edge_server.sendall(struct.pack("!I", aggregated_metrics_size))
        self.edge_server.sendall(aggregated_metrics_bytes)
        return None

    def receive_data_from_server(self):
        while True:
            try:
                state_dict_size = struct.unpack("!I", self.edge_server.recv(4))[0]
                state_dict_bytes = b""
                while len(state_dict_bytes) < state_dict_size:
                    msg = self.edge_server.recv(
                        min(
                            self.socket_volumn,
                            state_dict_size - len(state_dict_bytes),
                        )
                    )
                    state_dict_bytes += msg
                    # Load the state_dict from the byte stream
                buffer = io.BytesIO(state_dict_bytes)
                shared_state_dict = torch.load(buffer)
                self.shared_state_dict = shared_state_dict
                self.num_local_update = int(self.edge_server.recv(1024).decode("utf-8"))
                break
            except:
                pass
        return None

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        for i in self.received_clients.keys():
            self.received_clients[i] = 0
        # del self.id_registration[:]
        # self.sample_registration.clear()
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)

    def start(self):
        condition = threading.Condition()
        print("Edge Started. Waiting for clients...")
        while threading.active_count() - 1 < args.num_clients / args.num_edges:
            conn, addr = self.edge_client.accept()
            thread = threading.Thread(
                target=self.handle_client, args=(condition, conn, addr)
            )
            thread.start()
        print("All clients connected.")
        while True:
            msg = self.receive_msg_from_server()
            if msg.split(" ")[0] == "start":
                self.start_training = True
                self.total_sample = int(msg.split(" ")[1])
                print("Start training.")
                with condition:
                    condition.wait(timeout=10)
                    condition.notify_all()
                break
        with condition:
            condition.wait(timeout=10)
        for num_comm in range(args.num_communication):
            print("Waiting for data from server.")
            self.receive_data_from_server()
            print("Received data from server.")
            for num_edgeagg in range(args.num_edge_aggregation):
                self.loss = 0.0
                self.correct = 0.0
                self.total = 0.0
                print("Start sending data to clients.")
                for client_id in self.id_registration:
                    self.send_data_to_client(client_id, self.client_conns[client_id])
                    print(f"Sended data to client {client_id}.")
                print("start receiving data from clients.")
                with condition:
                    condition.notify_all()
                while sum(self.received_clients.values()) < len(self.id_registration):
                    pass
                print("Received data from all clients.")
                self.aggregated_metrics[num_edgeagg] = (
                    self.loss,
                    self.correct,
                    self.total,
                )
                self.aggregate(args)
                print("Aggregated.")
                self.refresh_edgeserver()
            print("Start sending data to server.")
            self.send_data_to_server()
            self.aggregated_metrics = [
                (0.0, 0.0, 0.0) for i in range(args.num_edge_aggregation)
            ]
            print("Sended data to server.")

        self.start_training = False
        with condition:
            condition.notify_all()
        print("Training finished.")
        self.edge_client.close()
        self.edge_server.close()


if __name__ == "__main__":
    args = args_parser()
    cluster_edge = Edge(args)
    cluster_edge.send_msg_to_server(f"edge {args.edge_port}")
    cluster_edge.start()
