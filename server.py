from options import args_parser
from datasets.prepare_data import get_dataset
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
import random
import time

from tqdm import tqdm
from models.mnist_cnn import Net as MNISTNet
from models.gquic_cnn import Net as GQUICNet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv

from algorithms.sac.sac import Agent as SAC_Agent
from algorithms.dql.dql_epsilon_agent import Agent as DQL_Epsilon_Agent
from algorithms.dql.dql_softmax_agent import Agent as DQL_Softmax_Agent
from algorithms.ddpg.ddpg import Agent as DDPG_Agent
from algorithms.ppo.ppo import Agent as PPO_Agent
from algorithms.ucb1.ucb1 import UCB1

test_losses = []
rewards = []
local_updates = []
global_accuracies = []
aggregated_accuracies = []
start_time = None


class Server:
    def __init__(self, agent, args):
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
        ) = get_dataset(args)
        self.socket_volumn = args.socket_volumn
        # Config algorithm
        self.num_local_update = args.num_local_update
        self.min_local_update = 1
        self.max_local_update = 100
        self.pre_state = None
        self.pre_action = None
        self.pre_action_ucb1 = None
        self.pre_log_prob = None
        self.epsilon_max = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.epsilon = self.epsilon_max
        self.alpha = args.alpha
        self.agent = agent
        self.ucb1 = UCB1(2)
        self.aggregated_accuracy = 0.0
        # define the edges
        self.edges = []
        self.edge_conns = {}
        self.edge_ports = []
        self.received_egdes = {}
        self.metrics = []
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
                    client_id = num_clients - 1
                    print(f"Client {client_id} connected.")
                    # clustering the clients
                    # send the edge port to the client
                    edge_id = client_id % args.num_edges
                    conn.send(f"{client_id} {self.edge_ports[edge_id]}".encode("utf-8"))
                    conn.close()
                    print("Disconnected with client ", addr)
                    with condition:
                        condition.wait()
                    break
                if msg.split(" ")[0] == "edge":
                    edge_listen_port = int(msg.split(" ")[1])
                    edge_id = len(self.edges)
                    self.edge_register(conn, addr, edge_listen_port)
                    print(f"Edge {edge_id} connected.")
                    with condition:
                        condition.wait()
                    self.send_msg_to_edge(edge_id, f"start")
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

    def fast_all_clients_test(self, test_loaders, global_nn, device):
        correct_all = 0.0
        total_all = 0.0
        with torch.no_grad():
            for data in test_loaders:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = global_nn(inputs)
                _, predicts = torch.max(outputs, 1)
                total_all += labels.size(0)
                correct_all += (predicts == labels).sum().item()
        accuracy = correct_all / total_all
        return accuracy

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
                for i in range(len(edge_aggregated_metrics) - 1):
                    self.metrics.append(edge_aggregated_metrics[i])
                    self.sample_registration[edge_id] += edge_aggregated_metrics[i][2]
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
        self.metrics = []
        self.aggregated_accuracy = 0.0
        # del self.id_registration[:]
        for i in self.sample_registration.keys():
            self.sample_registration[i] = 0
        return None

    def close_edge_conn(self, edge_id):
        self.edge_conns[edge_id].close()
        return None

    def calculate_sac(self, curr_reward, curr_state):
        # add to experiment buffer
        if self.pre_state is not None:
            self.agent.train_on_transition(
                self.pre_state, self.pre_action, curr_reward, curr_state
            )
        self.pre_state = curr_state
        # predict next action
        action = self.agent.get_next_action(curr_state, False)

        self.pre_action = action
        return action

    def calculate_dql_epsilon(self, curr_reward, curr_state):
        # add to experiment buffer
        if self.pre_state is not None:
            self.agent.step(self.pre_state, self.pre_action, curr_reward, curr_state)
        self.pre_state = curr_state

        # predict next action
        action = self.agent.act(curr_state, self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.pre_action = action
        return action

    def calculate_dql_ucb1(self, curr_reward, curr_state):
        # update ucb1
        if self.pre_action_ucb1 is not None:
            self.ucb1.update(self.pre_action_ucb1, curr_reward)

        # add to experiment buffer
        if self.pre_state is not None:
            self.agent.step(self.pre_state, self.pre_action, curr_reward, curr_state)
        self.pre_state = curr_state

        # predict next action
        action_ucb1 = self.ucb1.select_arm()
        self.pre_action_ucb1 = action_ucb1

        # action mapping: 0 -> random, 1 -> deep q-learning
        if action_ucb1 == 0:
            action = random.choice(np.arange(3))
        else:
            action = self.agent.act(curr_state)

        self.pre_action = action
        return action

    def calculate_dql_softmax(self, curr_reward, curr_state):
        # add to experiment buffer
        if self.pre_state is not None:
            self.agent.step(self.pre_state, self.pre_action, curr_reward, curr_state)
        self.pre_state = curr_state

        # predict next action
        action = self.agent.act(curr_state)

        self.pre_action = action
        return action

    def calculate_ddpg_epsilon(self, curr_reward, curr_state):
        # add to experiment buffer
        BATCH_SIZE = 64
        if self.pre_state is not None:
            self.agent.memory.push(
                self.pre_state, self.pre_action, curr_reward, curr_state
            )
        self.pre_state = curr_state

        # predict next action
        if np.random.random() > self.epsilon:
            action = self.agent.get_action(curr_state)
        else:
            action = random.uniform(-1, 1)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.pre_action = np.array([action])
        if len(self.agent.memory) > BATCH_SIZE:
            self.agent.update(BATCH_SIZE)
        return action

    def calculate_ddpg_ucb1(self, curr_reward, curr_state):
        # add to experiment buffer
        BATCH_SIZE = 64
        if self.pre_state is not None:
            self.agent.memory.push(
                self.pre_state, self.pre_action, curr_reward, curr_state
            )
        self.pre_state = curr_state

        # predict next action
        action_ucb1 = self.ucb1.select_arm()
        self.pre_action_ucb1 = action_ucb1

        # action mapping: 0 -> random, 1 -> ddpg
        if action_ucb1 == 0:
            action = random.uniform(-1, 1)
        else:
            action = self.agent.get_action(curr_state)

        self.pre_action = np.array([action])

        if len(self.agent.memory) > BATCH_SIZE:
            self.agent.update(BATCH_SIZE)

        return action

    def calculate_ppo(self, curr_reward, curr_state):
        # add to experiment buffer
        BATCH_SIZE = 2
        if self.pre_state is not None:
            self.agent.memory.push(
                self.pre_state,
                self.pre_action,
                self.pre_log_prob,
                curr_reward,
                curr_state,
            )
        self.pre_state = curr_state

        # predict next action
        action, log_prob, _ = self.agent.choose_action(curr_state)
        self.pre_action = action
        self.pre_log_prob = log_prob

        if len(self.agent.memory) > BATCH_SIZE:
            self.agent.learn(BATCH_SIZE)

        return action

    def calculate_local_updates(self, num_edges, algorithm):
        self.metrics = np.array(self.metrics)
        # calculate test_loss and reward
        total_example, total_loss = 0, 0
        for i in self.metrics:
            total_example += i[2]
            total_loss += i[0] * i[2]
        loss = total_loss / total_example
        curr_reward = -(loss + self.num_local_update * self.alpha)
        rewards.append(curr_reward)
        test_losses.append(loss)

        self.aggregated_accuracy /= total_example
        print("Aggragated accuracy: ", self.aggregated_accuracy)
        global aggregated_accuracies
        aggregated_accuracies.append(self.aggregated_accuracy)

        if algorithm == "none":
            return None

        curr_state = np.concatenate(
            (
                self.metrics[:, 0],
                self.metrics[:, 1],
                self.metrics[:, 2],
                np.array([self.num_local_update]),
            )
        )
        curr_state = normalize(curr_state.reshape(1, -1))[0]
        local_updates.append(self.num_local_update)
        action = 1
        if algorithm == "sac":
            action = self.calculate_sac(curr_reward, curr_state)
        elif algorithm == "dql_epsilon":
            action = self.calculate_dql_epsilon(curr_reward, curr_state)
        elif algorithm == "dql_ucb1":
            action = self.calculate_dql_ucb1(curr_reward, curr_state)
        elif algorithm == "dql_softmax":
            action = self.calculate_dql_softmax(curr_reward, curr_state)
        elif algorithm == "ddpg_epsilon":
            action = self.calculate_ddpg_epsilon(curr_reward, curr_state)
        elif algorithm == "ddpg_ucb1":
            action = self.calculate_ddpg_ucb1(curr_reward, curr_state)
        else:
            action = self.calculate_ppo(curr_reward, curr_state)
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

        # Declare storage file
        this_dir = Path.cwd()
        algorithm = args.algorithm
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        data_distribution = args.skewness if args.iid == 0 else "iid"
        FILEOUT = (
            f"local-update-{args.num_local_update}_edgeagg-{args.num_edge_aggregation}"
            f"_{data_distribution}_alpha-{self.alpha}_lr-{args.lr}"
        )

        output_dir = this_dir / "runs" / algorithm / f"{FILEOUT}_{current_time}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Store results to files
        training_time_file = str(output_dir) + "/training_time.txt"
        training_time = time.time() - start_time
        with open(training_time_file, "w") as f:
            f.write(f"{training_time}")

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
        global start_time
        start_time = time.time()
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
            self.calculate_local_updates(
                num_edges=args.num_edges,
                algorithm=args.algorithm,
            )
            self.refresh_cloudserver(args)

            # Validate model with server's test dataset
            global_nn.load_state_dict(state_dict=copy.deepcopy(self.shared_state_dict))
            global_nn.eval()
            global_acc = self.fast_all_clients_test(
                self.test_loaders, global_nn, device=DEVICE
            )
            print("Global accuracy: ", global_acc)
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
    if args.algorithm != "none":
        state_dims = 3 * args.num_clients + 1
        if args.algorithm == "sac":
            agent = SAC_Agent(
                state_dims=state_dims,
                action_dims=3,
                hidden_dims=64,
                batch_size=64,
            )
        elif args.algorithm == "dql_epsilon" or args.algorithm == "dql_ucb1":
            agent = DQL_Epsilon_Agent(
                state_size=state_dims,
                action_size=3,
                seed=0,
            )
        elif args.algorithm == "dql_softmax":
            agent = DQL_Softmax_Agent(
                state_size=state_dims,
                action_size=3,
                seed=0,
            )
        elif args.algorithm == "ddpg_epsilon" or args.algorithm == "ddpg_ucb1":
            agent = DDPG_Agent(
                state_dims=state_dims,
                action_dims=1,
            )
        elif args.algorithm == "ppo":
            agent = PPO_Agent(
                state_dims=state_dims,
                action_dims=3,
                hidden_dims=64,
            )
        else:
            raise ValueError(f"Algorithm {args.algorithm} not implemented")
    else:
        agent = None
    print("Algorithm: ", args.algorithm)
    server = Server(agent, args)
    server.start(args)


if __name__ == "__main__":
    main()
