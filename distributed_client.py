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
    def __init__(self, args, host="127.0.0.1", port=40000):
        # connect to the head
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        # config training
        self.id = args.client_id
        self.train_loader = {}
        self.test_loader = {}
        self.model = initialize_model(args, device="cpu")
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        # record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []

    def local_update(self, num_iter, device="cpu"):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
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

    def test_model(self, device="cpu"):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self):
        # Serialize the state_dict to a byte stream
        buffer = io.BytesIO()
        torch.save(self.model.shared_layers.state_dict(), buffer)

        # Get the byte stream from the buffer
        state_dict_bytes = buffer.getvalue()

        # Send the size of the state_dict_bytes before sending state_dict_bytes
        size = len(state_dict_bytes)
        self.client_socket.sendall(struct.pack("!I", size))
        self.client_socket.sendall(state_dict_bytes)
        return None

    # def receive_from_edgeserver(self, shared_state_dict):
    #     self.receiver_buffer = shared_state_dict
    #     return None

    # def sync_with_edgeserver(self):
    #     """
    #     The global has already been stored in the buffer
    #     :return: None
    #     """
    #     # self.model.shared_layers.load_state_dict(self.receiver_buffer)
    #     self.model.update_model(self.receiver_buffer)
    #     return None

    def connect_to_edge(self, edge_port):
        host = "127.0.0.1"
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, edge_port))

    def send_msg(self, msg):
        self.client_socket.send(msg.encode("utf-8"))

    # receive message from server
    def receive_msg(self):
        msg = self.client_socket.recv(1024).decode("utf-8")
        return msg

    def disconnect(self):
        self.send_msg("DISCONNECT")

    def build_dataloaders(self):
        (
            train_loaders,
            test_loaders,
            v_train_loader,
            v_test_loader,
        ) = get_dataloaders(args)
        self.train_loader = train_loaders[self.id]
        self.test_loader = test_loaders[self.id]


if __name__ == "__main__":
    args = args_parser()
    client = Client(args)
    client.build_dataloaders()
    while True:
        msg = input()
        # if msg == "DISCONNECT":
        #     client.disconnect()
        #     break
        client.send_msg(msg)
        received_msg = client.receive_msg()
        print(received_msg)
        if received_msg != "":
            edge_port = int(received_msg)
            print(f"Redirect to edge {edge_port}")
            client.connect_to_edge(edge_port)
            client.send_msg(str(client.id))
            loss = client.local_update(num_iter=10)
            print(loss)
            client.send_to_edgeserver()
            break

    # while True:
    #     received_msg = client.receive_msg()
    #     if received_msg == "disconnect":
    #         client.disconnect()
    #         break
    #     if received_msg[:1]= "1":
    client.disconnect()
