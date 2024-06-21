from options import args_parser
from tensorboardX import SummaryWriter
from datasets.get_data import get_dataloaders

import sys
import socket
import threading
import struct
import torch
import io
import pickle
from datetime import datetime
import os
from sklearn.preprocessing import normalize
from pathlib import Path

from average import average_weights
import numpy as np
import copy

from tqdm import tqdm
from models.mnist_cnn import Net as MNISTNet
from models.gquic_cnn import Net as GQUICNet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv

from sac.sac import SACAgent

test_losses = []
rewards = []
local_updates = []
global_accuracies = []
aggregated_accuracies = []
global_f1 = []
aggregated_f1 = []


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
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        (
            self.train_loaders,
            self.val_loaders,
            self.test_loaders,
        ) = get_dataloaders(args)
        self.total_sample = sum([len(loader.dataset) for loader in self.train_loaders])
        self.socket_volumn = args.socket_volumn
        # Config SAC
        self.num_local_update = args.num_local_update
        self.min_local_update = 1
        self.max_local_update = 50
        self.pre_state = None
        self.pre_action = None
        self.state_dims = 13
        self.action_dims = 3
        self.alpha = 0.01
        self.agent = SACAgent(
            state_dims=self.state_dims,
            action_dims=self.action_dims,
            hidden_dims=64,
            batch_size=args.batch_size,
        )
        self.aggregated_accuracy = 0.0
        self.aggregated_f1_score = 0.0
        # define the edges
        self.edges = []
        self.edge_conns = {}
        self.edge_ports = []
        self.received_egdes = {}
        self.metrics_SAC = []
        # define the clients
        self.clients = []

    def handle_connection(self, condition, conn, addr, args):
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
                    if num_clients < args.num_clients // 2 + 1:
                        print(
                            f"Redirect client {addr} to edge with port number {self.edge_ports[0]}"
                        )
                        conn.send(
                            f"{num_clients - 1} {self.edge_ports[0]}".encode("utf-8")
                        )
                        self.sample_registration[0] += len(
                            self.train_loaders[num_clients - 1].dataset
                        )
                    else:
                        print(
                            f"Redirect client {addr} to edge with port number {self.edge_ports[1]}"
                        )
                        conn.send(
                            f"{num_clients - 1} {self.edge_ports[1]}".encode("utf-8")
                        )
                        self.sample_registration[1] += len(
                            self.train_loaders[num_clients - 1].dataset
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
                        self.receive_data_from_edge(edge_id, conn, args)
                        print(f"Received data from edge {edge_id}.")
                    break

    def send_msg_to_edge(self, edge_id, msg):
        self.edge_conns[edge_id].send(msg.encode("utf-8"))

    # training functions
    def initialize_global_nn(self, args):
        if args.dataset == "mnist":
            if args.model == "mnist_cnn":
                global_nn = MNISTNet()
            else:
                raise ValueError(f"Model{args.model} not implemented for mnist")
        elif args.dataset == "cifar10":
            if args.model == "cifar10_cnn":
                global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
            else:
                raise ValueError(f"Model{args.model} not implemented for cifar")
        elif args.dataset == "gquic":
            if args.model == "gquic_cnn":
                global_nn = GQUICNet()
            else:
                raise ValueError(f"Model{args.model} not implemented for gquic")
        else:
            raise ValueError(f"Dataset {args.dataset} Not implemented")
        if args.cuda:
            global_nn = global_nn.cuda(torch.device("cuda"))
        return global_nn

    def fast_all_clients_test(self, test_loaders, global_nn, device, num_class):
        correct_all = 0.0
        total_all = 0.0
        precision = 0.0
        recall = 0.0
        count_class = [[0, 0, 0] for _ in range(num_class)]
        actual_classes = set()
        with torch.no_grad():
            for data in test_loaders:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = global_nn(inputs)
                _, predicts = torch.max(outputs, 1)
                total_all += labels.size(0)
                correct_all += (predicts == labels).sum().item()
                for class_idx in range(num_class):
                    if class_idx not in labels and class_idx not in predicts:
                        continue
                    actual_classes.add(class_idx)
                    TP, FP, FN = 0, 0, 0
                    true_class_mask = labels == class_idx
                    predicted_class_mask = predicts == class_idx
                    # True Positives (TP): Predicted as current class and actually belongs to the current class
                    TP += torch.sum(predicted_class_mask & true_class_mask).item()
                    # False Positives (FP): Predicted as current class but actually belongs to a different class
                    FP += torch.sum(predicted_class_mask & (~true_class_mask)).item()
                    # False Negatives (FN): Predicted as a different class but actually belongs to the current class
                    FN += torch.sum((~predicted_class_mask) & true_class_mask).item()
                    count_class[class_idx][0] += TP
                    count_class[class_idx][1] += FP
                    count_class[class_idx][2] += FN

        for i in actual_classes:
            TP = count_class[i][0]
            FP = count_class[i][1]
            FN = count_class[i][2]
            precision += TP / (TP + FP) if TP + FP != 0 else 0
            recall += TP / (TP + FN) if TP + FN != 0 else 0
        precision /= len(actual_classes)
        recall /= len(actual_classes)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall != 0
            else 0
        )
        accuracy = correct_all / total_all
        return f1_score, accuracy

    def edge_register(self, conn, addr, edge_listen_port):
        edge_id = len(self.edges)
        self.received_egdes[edge_id] = 0
        self.edges.append(addr)
        self.edge_conns[edge_id] = conn
        self.edge_ports.append(edge_listen_port)
        self.id_registration.append(edge_id)
        self.sample_registration[edge_id] = 0
        return None

    def receive_data_from_edge(self, edge_id, conn, args):
        while True:
            try:
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
                self.receiver_buffer[edge_id] = shared_state_dict
                print(f"Received state_dict from edge {edge_id}")
                edge_aggregated_metrics_size = struct.unpack("!I", conn.recv(4))[0]
                edge_aggregated_metrics_bytes = b""
                while len(edge_aggregated_metrics_bytes) < edge_aggregated_metrics_size:
                    msg = conn.recv(
                        min(
                            self.socket_volumn,
                            edge_aggregated_metrics_size
                            - len(edge_aggregated_metrics_bytes),
                        )
                    )
                    edge_aggregated_metrics_bytes += msg
                edge_aggregated_metrics = pickle.loads(edge_aggregated_metrics_bytes)
                for i in range(args.num_clients // args.num_edges):
                    self.metrics_SAC.append(edge_aggregated_metrics[i])
                self.aggregated_f1_score += edge_aggregated_metrics[-2]
                self.aggregated_accuracy += edge_aggregated_metrics[-1]
                print(f"Received metrics from edge {edge_id}")
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
        conn.sendall(str(self.num_local_update).encode("utf-8"))
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def refresh_cloudserver(self, args):
        self.receiver_buffer.clear()
        for i in self.received_egdes.keys():
            self.received_egdes[i] = 0
        self.metrics_SAC = []
        self.aggregated_f1_score = 0.0
        self.aggregated_accuracy = 0.0
        # del self.id_registration[:]
        # self.sample_registration.clear()
        return None

    def close_edge_conn(self, edge_id):
        self.edge_conns[edge_id].close()
        return None

    def cal_SAC(self, num_edges, apply_algorithm):
        self.metrics_SAC = np.array(self.metrics_SAC)
        # calculate test_loss and reward
        total_example, total_loss = 0, 0
        for i in self.metrics_SAC:
            total_example += i[2]
            total_loss += i[0] * i[2]
        loss = total_loss / total_example
        curr_reward = -loss + self.alpha / self.num_local_update
        rewards.append(curr_reward)
        test_losses.append(loss)

        # calculate f1_score and accuracy
        self.aggregated_f1_score /= num_edges
        print("Aggregated f1 score: ", self.aggregated_f1_score)
        global aggregated_f1
        aggregated_f1.append(self.aggregated_f1_score)

        self.aggregated_accuracy /= num_edges
        print("Aggragated accuracy: ", self.aggregated_accuracy)
        global aggregated_accuracies
        aggregated_accuracies.append(self.aggregated_accuracy)

        if not apply_algorithm:
            return None

        curr_state = np.concatenate(
            (
                self.metrics_SAC[:, 0],
                self.metrics_SAC[:, 1],
                self.metrics_SAC[:, 2],
                np.array([self.num_local_update]),
            )
        )
        curr_state = normalize(curr_state.reshape(1, -1))[0]
        local_updates.append(self.num_local_update)

        # add to experiment buffer
        if self.pre_state is not None:
            self.agent.train_on_transition(
                self.pre_state, self.pre_action, curr_reward, curr_state
            )
        self.pre_state = curr_state
        # predict next action
        action = self.agent.get_next_action(curr_state, False)
        # Keep the number of local update if no algorithm is applied
        self.pre_action = action

        # action mapping: 0 -> -1, 1 -> 0, 2 -> 1
        if action == 0:
            self.num_local_update -= 1
        elif action == 1:
            pass
        else:
            self.num_local_update += 1
        # clip the number of local update
        self.num_local_update = max(
            self.min_local_update, min(self.num_local_update, self.max_local_update)
        )
        print("Next number of local update: ", self.num_local_update)

    def saveFile(self, args):
        # Declare storage file
        global test_losses
        global local_updates
        global rewards
        global global_accuracies
        global aggregated_accuracies
        global global_f1
        global aggregated_f1

        # Declare storage file
        this_dir = Path.cwd()
        algorithm = "SAC" if args.apply_algorithm else "FedAvg"
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        data_distribution = "iid" if args.iid else "non-iid"
        FILEOUT = (
            f"local-update-{args.num_local_update}_edgeagg-{args.num_edge_aggregation}"
            f"_{data_distribution}_alpha-{self.alpha}"
        )

        output_dir = this_dir / "runs" / algorithm / f"{FILEOUT}_{current_time}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Store results to files
        test_loss_file = str(output_dir) + "/test_loss.pkl"
        test_losses = np.array(test_losses)

        with open(test_loss_file, "wb") as f:
            pickle.dump(test_losses, f)

        local_update_file = str(output_dir) + "/local_update.pkl"
        local_updates = np.array(local_updates)
        with open(local_update_file, "wb") as f:
            pickle.dump(local_updates, f)

        reward_file = str(output_dir) + "/reward.pkl"
        rewards = np.array(rewards)
        with open(reward_file, "wb") as f:
            pickle.dump(rewards, f)

        global_accuracy_file = str(output_dir) + "/global_accuracy.pkl"
        global_accuracies = np.array(global_accuracies)
        with open(global_accuracy_file, "wb") as f:
            pickle.dump(global_accuracies, f)

        aggregated_accuracy_file = str(output_dir) + "/aggregated_accuracy.pkl"
        aggregated_accuracies = np.array(aggregated_accuracies)
        with open(aggregated_accuracy_file, "wb") as f:
            pickle.dump(aggregated_accuracies, f)

        global_f1_file = str(output_dir) + "/global_f1.pkl"
        global_f1 = np.array(global_f1)
        with open(global_f1_file, "wb") as f:
            pickle.dump(global_f1, f)

        aggregated_f1_file = str(output_dir) + "/aggregated_f1.pkl"
        aggregated_f1 = np.array(aggregated_f1)
        with open(aggregated_f1_file, "wb") as f:
            pickle.dump(aggregated_f1, f)

    def start(self, args):
        condition = threading.Condition()
        print("Server Started. Waiting for clusters...")
        DEVICE = torch.device("cuda" if args.cuda else "cpu")
        print("Training on device: ", DEVICE)
        while True:
            if self.start_training == False:
                if threading.active_count() - 1 < (args.num_edges + args.num_clients):
                    conn, addr = self.server.accept()
                    thread = threading.Thread(
                        target=self.handle_connection,
                        args=(condition, conn, addr, args),
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
        self.shared_state_dict = global_nn.state_dict()
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

            # Calculating the number of local updates
            self.cal_SAC(num_edges=args.num_edges, apply_algorithm=args.apply_algorithm)
            self.refresh_cloudserver(args)

            # Validate model with server's test dataset
            global_nn.load_state_dict(state_dict=copy.deepcopy(self.shared_state_dict))
            global_nn.eval()
            f1_score, global_acc = self.fast_all_clients_test(
                self.test_loaders, global_nn, device=DEVICE, num_class=args.num_class
            )
            print("Global f1 score: ", f1_score)
            print("Global accuracy: ", global_acc)
            global global_f1
            global_f1.append(f1_score)
            global global_accuracies
            global_accuracies.append(global_acc)
        self.saveFile(args)
        self.start_training = False
        with condition:
            condition.notify_all()
        print("Training finished.")
        self.server.close()


def main():
    args = args_parser()
    server = Server(args)
    server.start(args)


if __name__ == "__main__":
    main()
