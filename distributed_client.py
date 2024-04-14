import socket
from options import args_parser
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy
import io
import argparse
from datasets.get_data import get_dataloaders, show_distribution
import struct


class Client:
    def __init__(self, args, server_host="127.0.0.1", server_port=40000):
        # connect to the head
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_host, server_port))
        # config training
        self.id = None
        self.train_loader = {}
        self.test_loader = {}
        self.device = args.device
        self.model = initialize_model(args)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        # record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []

    def local_update(self, num_iter):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(self.device)
                labels = Variable(labels).to(self.device)
                loss += self.model.optimize_model(
                    input_batch=inputs, label_batch=labels
                )
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end:
                break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_model(self):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    # send normal message
    def send_msg(self, msg):
        self.client_socket.send(msg.encode("utf-8"))

    # receive normal message
    def receive_msg(self):
        msg = self.client_socket.recv(1024).decode("utf-8")
        return msg

    def connect_to_edge(self, edge_port, host="127.0.0.1"):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, edge_port))

    def send_data_to_edge(self, loss, correct, total):
        # Serialize the state_dict to a byte stream
        buffer = io.BytesIO()
        torch.save(self.model.shared_layers.state_dict(), buffer)

        # Get the byte stream from the buffer
        state_dict_bytes = buffer.getvalue()

        # Send the size of the state_dict_bytes before sending state_dict_bytes
        state_dict_size = len(state_dict_bytes)
        self.client_socket.sendall(struct.pack("!I", state_dict_size))
        self.client_socket.sendall(state_dict_bytes)

        # Send the loss, correct and total to the edge
        self.client_socket.sendall(
            f"{str(loss)} {str(correct)} {str(total)}".encode("utf-8")
        )
        return None

    def sync_with_edge(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

    def receive_data_from_edge(self):
        while True:
            try:
                size = struct.unpack("!I", self.client_socket.recv(4))[0]
                state_dict_bytes = b""
                while len(state_dict_bytes) < size:
                    state_dict_bytes += self.client_socket.recv(1048576)
                    # Load the state_dict from the byte stream
                buffer = io.BytesIO(state_dict_bytes)
                shared_state_dict = torch.load(buffer)
                self.receiver_buffer = shared_state_dict
                break
            except:
                pass
        return None

    def disconnect(self):
        self.send_msg("DISCONNECT")

    def build_dataloaders(self, args):
        (
            train_loaders,
            test_loaders,
            v_train_loader,
            v_test_loader,
        ) = get_dataloaders(args)
        self.train_loader = train_loaders[self.id]
        self.test_loader = test_loaders[self.id]


def main():
    args = args_parser()
    client = Client(args)
    # register the client to the server and get the assigned edge's port
    while True:
        # send message to the server to register the client
        client.send_msg("client")
        received_msg = client.receive_msg()
        print(received_msg)
        if received_msg != "":
            client.id = int(received_msg.split(" ")[0])
            edge_port = int(received_msg.split(" ")[1])
            client.build_dataloaders(args)
            print(f"Redirect to edge {edge_port}")
            client.connect_to_edge(edge_port)
            client.send_msg(f"{client.id} {len(client.train_loader.dataset)}")
            break
    # wait for the server to start the training
    while True:
        received_msg = client.receive_msg()
        if received_msg == "start":
            print("start training")
            break
    # training
    for num_comm in range(args.num_communication):
        for num_edgeagg in range(args.num_edge_aggregation):
            print("Start receiving data from edge")
            client.receive_data_from_edge()
            print("Received data from edge")
            client.sync_with_edge()
            loss = client.local_update(num_iter=args.num_local_update)
            print("Start testing")
            correct, total = client.test_model()
            print("Start sending data to edge")
            client.send_data_to_edge(loss=loss, correct=correct, total=total)
            print("Sended data to edge")

    print("Training finished")
    client.client_socket.close()


if __name__ == "__main__":
    main()
