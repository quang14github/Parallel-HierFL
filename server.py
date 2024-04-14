from options import args_parser
from tensorboardX import SummaryWriter
from datasets.get_data import get_dataloaders, show_distribution
from models.initialize_model import initialize_model

import socket
import threading
import struct
import torch
import io
import pickle

from average import average_weights
import numpy as np
import copy

from tqdm import tqdm
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_logistic import LogisticRegression

default_socket_volumn = 1048576


class Server:
    def __init__(self, args):
        # config server ip and port
        self.host = "127.0.0.1"
        self.port = 40000
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        # config training
        self.start_training = False
        self.model = initialize_model(args)
        self.global_loss = 0.0
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        (
            self.train_loaders,
            self.test_loaders,
            self.v_train_loader,
            self.v_test_loader,
        ) = get_dataloaders(args)
        self.total_sample = sum([len(loader.dataset) for loader in self.train_loaders])
        # define the edges
        self.edges = []
        self.edge_conns = {}
        self.edge_ports = []
        self.received_egdes = {}
        self.edge_metrics = [[] for _ in range(args.num_edges)]
        # define the clients
        self.clients = []

    def handle_connection(self, condition, conn, addr):
        while True:
            if self.start_training == False:
                msg = conn.recv(1024).decode("utf-8")
                if msg == "DISCONNECT" or msg == "":
                    print(f"{addr} disconnected.")
                    conn.close()
                    print("Disconnected with client/edge ", addr)
                    with condition:
                        condition.wait()
                    break
                if msg == "client":
                    self.clients.append(addr)
                    num_clients = len(self.clients)
                    print(f"New client {addr} connected.")
                    # clustering the clients
                    # send the edge port to the client
                    if num_clients < 3:
                        print(
                            f"Redirect client {addr} to edge with port number {self.edge_ports[0]}"
                        )
                        conn.send(
                            f"{num_clients - 1} {self.edge_ports[0]}".encode("utf-8")
                        )
                    else:
                        print(
                            f"Redirect client {addr} to edge with port number {self.edge_ports[1]}"
                        )
                        conn.send(
                            f"{num_clients - 1} {self.edge_ports[1]}".encode("utf-8")
                        )
                    conn.close()
                    print("Disconnected with client ", addr)
                    with condition:
                        condition.wait()
                    break
                if msg.split(" ")[0] == "edge":
                    edge_listen_port = int(msg.split(" ")[1])
                    edge_id = len(self.edges)
                    self.edge_register(conn, addr, edge_listen_port)
                    print(f"New edge {addr} connected.")
                    with condition:
                        condition.wait()
                    self.send_msg_to_edge(edge_id, f"start {self.total_sample}")
                    while True:
                        with condition:
                            condition.wait()
                        if self.start_training == False:
                            break
                        self.receive_data_from_edge(edge_id, conn)
                        print(f"Received data from edge {edge_id}.")
                    break

    def send_msg_to_edge(self, edge_id, msg):
        self.edge_conns[edge_id].send(msg.encode("utf-8"))

    # training functions
    def initialize_global_nn(self, args):
        if args.dataset == "mnist":
            if args.model == "lenet":
                global_nn = mnist_lenet(input_channels=1, output_channels=10)
            elif args.model == "logistic":
                global_nn = LogisticRegression(input_dim=1, output_dim=10)
            else:
                raise ValueError(f"Model{args.model} not implemented for mnist")
        elif args.dataset == "cifar10":
            if args.model == "cnn_complex":
                global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
            elif args.model == "resnet18":
                global_nn = ResNet18()
            else:
                raise ValueError(f"Model{args.model} not implemented for cifar")
        else:
            raise ValueError(f"Dataset {args.dataset} Not implemented")
        return global_nn

    def fast_all_clients_test(self, v_test_loader, global_nn, device):
        correct_all = 0.0
        total_all = 0.0
        with torch.no_grad():
            for data in v_test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = global_nn(inputs)
                _, predicts = torch.max(outputs, 1)
                total_all += labels.size(0)
                correct_all += (predicts == labels).sum().item()
        return correct_all, total_all

    def edge_register(self, conn, addr, edge_listen_port):
        edge_id = len(self.edges)
        self.received_egdes[edge_id] = 0
        self.edges.append(addr)
        self.edge_conns[edge_id] = conn
        self.edge_ports.append(edge_listen_port)
        self.id_registration.append(edge_id)
        edge_all_trainsample_num = sum(
            [
                len(self.train_loaders[i].dataset)
                for i in range(edge_id * 2, (edge_id + 1) * 2)
            ]
        )
        self.sample_registration[edge_id] = edge_all_trainsample_num
        return None

    def receive_data_from_edge(self, edge_id, conn):
        while True:
            try:
                state_dict_size = struct.unpack("!I", conn.recv(4))[0]
                state_dict_bytes = b""
                while len(state_dict_bytes) < state_dict_size:
                    msg = conn.recv(
                        min(
                            default_socket_volumn,
                            state_dict_size - len(state_dict_bytes),
                        )
                    )
                    state_dict_bytes += msg
                    # Load the state_dict from the byte stream
                buffer = io.BytesIO(state_dict_bytes)
                shared_state_dict = torch.load(buffer)
                self.receiver_buffer[edge_id] = shared_state_dict
                print(f"Received state_dict from edge {edge_id}")
                edge_aggregated_metrics_size = struct.unpack("!I", conn.recv(4))[0]
                edge_aggregated_metrics_bytes = b""
                while len(edge_aggregated_metrics_bytes) < edge_aggregated_metrics_size:
                    msg = conn.recv(
                        min(
                            default_socket_volumn,
                            edge_aggregated_metrics_size
                            - len(edge_aggregated_metrics_bytes),
                        )
                    )
                    edge_aggregated_metrics_bytes += msg
                edge_aggregated_metrics = pickle.loads(edge_aggregated_metrics_bytes)
                self.edge_metrics[edge_id] = edge_aggregated_metrics
                print(f"Received metrics from edge {edge_id}")
                print(edge_aggregated_metrics)
                self.received_egdes[edge_id] = 1
                break
            except:
                pass
        return None

    def send_data_to_edge(self, edge_id, conn):
        # Serialize the state_dict to a byte stream
        buffer = io.BytesIO()
        torch.save(self.shared_state_dict, buffer)

        # Get the byte stream from the buffer
        state_dict_bytes = buffer.getvalue()

        # Send the size of the state_dict_bytes before sending state_dict_bytes
        size = len(state_dict_bytes)
        conn.sendall(struct.pack("!I", size))
        conn.sendall(state_dict_bytes)
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        for i in self.received_egdes.keys():
            self.received_egdes[i] = 0
        self.edge_metrics = [[] for _ in range(len(self.edges))]
        # del self.id_registration[:]
        # self.sample_registration.clear()
        return None

    def close_edge_conn(self, edge_id):
        self.edge_conns[edge_id].close()
        return None

    def start(self, args):
        FILEOUT = (
            f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_"
            f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}"
            f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}"
            f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}"
            f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
        )
        writer = SummaryWriter(comment=FILEOUT)
        condition = threading.Condition()
        print("Server Started. Waiting for clusters...")
        while True:
            if self.start_training == False:
                if threading.active_count() - 2 < (args.num_edges + args.num_clients):
                    conn, addr = self.server.accept()
                    thread = threading.Thread(
                        target=self.handle_connection, args=(condition, conn, addr)
                    )
                    thread.start()
                else:
                    with condition:
                        condition.wait(timeout=10)
                        print("All clusters connected. Server ready to start training.")
                        self.start_training = True
                        condition.notify_all()
                    break
        # New an NN model for testing error
        global_nn = self.initialize_global_nn(args)
        with condition:
            condition.wait(timeout=5)
        # Start training
        for num_comm in tqdm(range(args.num_communication)):
            print(f"Communication round {num_comm}")
            print("Start sending data to all edges.")
            for edge_id in self.id_registration:
                self.send_data_to_edge(edge_id, self.edge_conns[edge_id])
                print(f"Sended data to edge {edge_id}")
            print("Sended data to all edges.")
            with condition:
                condition.notify_all()
            while sum(self.received_egdes.values()) < len(self.id_registration):
                pass
            print("All edges have sent their local models.")
            self.aggregate(args)
            print("Aggregation finished.")
            for num_edgeagg in range(args.num_edge_aggregation):
                all_loss = 0.0
                correct_all = 0.0
                total_all = 0.0
                for edge_id in self.id_registration:
                    all_loss += self.edge_metrics[edge_id][num_edgeagg][0]
                    correct_all += self.edge_metrics[edge_id][num_edgeagg][1]
                    total_all += self.edge_metrics[edge_id][num_edgeagg][2]
                avg_acc = float(correct_all / total_all)
                writer.add_scalar(
                    f"Partial_Avg_Train_loss",
                    all_loss,
                    num_comm * args.num_edge_aggregation + num_edgeagg + 1,
                )
                writer.add_scalar(
                    f"All_Avg_Test_Acc_edgeagg",
                    avg_acc,
                    num_comm * args.num_edge_aggregation + num_edgeagg + 1,
                )
            self.refresh_cloudserver()
            global_nn.load_state_dict(state_dict=copy.deepcopy(self.shared_state_dict))
            global_nn.train(False)
            correct_all_v, total_all_v = self.fast_all_clients_test(
                self.v_test_loader, global_nn, device="cpu"
            )
            avg_acc_v = correct_all_v / total_all_v
            writer.add_scalar(
                f"All_Avg_Test_Acc_cloudagg_Vtest", avg_acc_v, num_comm + 1
            )

        self.start_training = False
        with condition:
            condition.notify_all()
        print("Training finished.")
        print(f"The final virtual acc is {avg_acc_v}")
        self.server.close()
        writer.close()


def main():
    args = args_parser()
    server = Server(args)
    print(server.total_sample)
    server.shared_state_dict = server.model.shared_layers.state_dict()
    server.start(args)


if __name__ == "__main__":
    main()
