import socket
import threading
import copy
import time
import pickle
import struct
import io
import numpy as np
import torch
from options import args_parser
from datasets.prepare_data import get_dataset
from models.gquic_cnn import Net as GQUICNet
from average import average_weights
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

global_accuracies = []
aggregated_accuracies = []
training_time = []
start_time = None


class Server:
    def __init__(self, args):
        self.host = "127.0.0.1"
        self.port = 40000
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        self.start_training = False
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clients = []
        self.client_conns = {}
        self.received_clients = {}
        self.train_loaders, self.val_loaders, self.test_loaders = get_dataset(args)
        self.socket_volumn = args.socket_volumn
        self.num_local_update = args.num_local_update
        self.aggregated_accuracy = 0.0

    def handle_connection(self, condition, conn, addr, args):
        while True:
            if self.start_training == False:
                msg = conn.recv(1024).decode("utf-8")
                if msg == "DISCONNECT" or msg == "":
                    print(f"{addr} disconnected.")
                    conn.close()
                    break
                if msg == "client":
                    client_id = len(self.clients)
                    self.client_register(conn, addr)
                    print(f"Client {client_id} connected.")
                    conn.send(f"{client_id}".encode("utf-8"))
                    with condition:
                        condition.wait()
                    conn.send("start".encode("utf-8"))
                    while True:
                        with condition:
                            condition.wait()
                        if self.start_training == False:
                            break
                        self.receive_data_from_client(client_id, conn)
                        print(f"Received data from client {client_id}")
                    break

    def client_register(self, conn, addr):
        client_id = len(self.clients)
        self.received_clients[client_id] = 0
        self.clients.append(addr)
        self.client_conns[client_id] = conn
        self.id_registration.append(client_id)
        self.sample_registration[client_id] = len(self.train_loaders[client_id].dataset)
        return None

    def initialize_global_nn(self, args):
        if args.dataset == "gquic":
            global_nn = GQUICNet()
        else:
            raise ValueError(f"Dataset {args.dataset} not implemented")
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
                (
                    client_train_loss,
                    client_accuracy,
                ) = map(float, conn.recv(1024).decode("utf-8").split(" "))
                print(
                    f"client {client_id}: ",
                    f"train_loss {client_train_loss} ",
                    f"accuracy {client_accuracy}.",
                )
                # self.test_losses[client_id] += client_test_loss
                # self.train_losses[client_id] += client_train_loss
                self.aggregated_accuracy += (
                    client_accuracy * self.sample_registration[client_id]
                )
                self.received_clients[client_id] = 1
                break
            except:
                pass
        return None

    def send_data_to_client(self, client_id, conn):
        buffer = io.BytesIO()
        torch.save(self.shared_state_dict, buffer)
        state_dict_bytes = buffer.getvalue()
        size = len(state_dict_bytes)
        conn.sendall(struct.pack("!I", size))
        conn.sendall(state_dict_bytes)

    def aggregate(self):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        for i in self.received_clients.keys():
            self.received_clients[i] = 0
        self.metrics = []
        # del self.id_registration[:]
        # for i in self.sample_registration.keys():
        #     self.sample_registration[i] = 0
        return None

    def saveFile(self, args):
        # Declare storage file
        global global_accuracies
        global training_time
        global aggregated_accuracies

        # Declare storage file
        this_dir = Path.cwd()
        algorithm = "FedAvg"
        data_distribution = args.skewness if args.iid == 0 else "iid"
        FILEOUT = (
            f"local-update-{args.num_local_update}" f"_{data_distribution}_lr-{args.lr}"
        )

        output_dir = this_dir / "runs" / algorithm / f"{FILEOUT}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Store results to files
        training_time_file = str(output_dir) + "/training_time.pkl"
        training_time = np.array(training_time)
        with open(training_time_file, "wb") as f:
            pickle.dump(training_time, f)

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
        print("Server Started. Waiting for clients...")
        DEVICE = torch.device("cuda" if args.cuda else "cpu")
        print("Training on device: ", DEVICE)
        while True:
            if self.start_training == False:
                if threading.active_count() - 1 < args.num_clients:
                    conn, addr = self.server.accept()
                    thread = threading.Thread(
                        target=self.handle_connection,
                        args=(condition, conn, addr, args),
                    )
                    thread.start()
                else:
                    with condition:
                        condition.wait(timeout=10)
                        print("All clients connected. Server ready to start training.")
                        self.start_training = True
                        condition.notify_all()
                    break
        global aggregated_accuracies
        aggregated_accuracies.append(self.aggregated_accuracy)
        global_nn = self.initialize_global_nn(args)
        self.shared_state_dict = global_nn.state_dict()
        global start_time
        global training_time
        start_time = time.time()
        for num_comm in tqdm(range(args.num_communication)):
            print(f"Communication round {num_comm}")
            print("Start sending data to all clients.")
            for client_id in self.id_registration:
                self.send_data_to_client(client_id, self.client_conns[client_id])
                print(f"Sended data to client {client_id}")
            print("Sended data to all clients.")
            with condition:
                condition.notify_all()
            while sum(self.received_clients.values()) < len(self.id_registration):
                pass
            print("All clients have sent their local models.")
            self.aggregate()
            print("Aggregation finished.")

            self.refresh_cloudserver()
            global_nn.load_state_dict(state_dict=copy.deepcopy(self.shared_state_dict))
            global_nn.eval()
            global_acc = self.fast_all_clients_test(
                self.test_loaders, global_nn, device=DEVICE
            )
            print("Global accuracy: ", global_acc)
            training_time.append(time.time() - start_time)
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
