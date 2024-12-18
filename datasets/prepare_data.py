"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""

import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import (
    DataLoader,
    random_split,
    TensorDataset,
)

# from options import args_parser
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *
import random
from collections import defaultdict

random.seed(111)


def printNumberOfSample(label_set, args):
    if args.is_server == 1:
        sample_record = []
        for client_index, _ in enumerate(label_set):
            client_record = []
            num_labels = {label: 0 for label in range(args.num_classes)}
            for label in label_set[client_index]:
                num_labels[label] += 1
            for _, value in num_labels.items():
                client_record.append(value)
            sample_record.append(client_record)
        sample_record = np.array(sample_record)

        # Print
        clients = [f"Client {i}" for i in range(args.num_clients)]
        classes = [f"Class {i}" for i in range(args.num_classes)]
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each category as a stacked bar
        bottom = np.zeros(args.num_clients)
        colors = ["#bfc66c", "#4dec5b", "#ad2a2c", "#f0a1ce", "#158903"]
        for i in range(args.num_classes):
            ax.bar(
                clients,
                sample_record[:, i],
                bottom=bottom,
                label=f"{classes[i]}",
                color=colors[i],
            )
            bottom += sample_record[:, i]

        # Customize plot
        ax.set_xlabel("Clients")
        ax.set_ylabel("Samples")
        ax.set_title("Client datasets distribution")
        ax.legend(loc="upper right")
        # plt.xticks(rotation=45)
        this_dir = Path.cwd()
        save_dir = this_dir / "datasets/client_sample"
        label = None
        if args.iid == 1:
            label = "iid"
        else:
            if args.skewness == "quantity":
                label = "quantity_skew"
            else:
                label = "label_skew"

        fig.savefig(save_dir / str("gquic256_" + label + ".png"))


def get_dataset(args):
    train_loaders = []
    val_loaders = []
    test_loaders = []
    if args.dataset == "cifar10":
        train_loaders, val_loaders, test_loaders = get_cifar10(args)
    elif args.dataset == "gquic":
        train_loaders, val_loaders, test_loaders = get_gquic(args)
    else:
        raise ValueError("Dataset `{}` not found".format(args.dataset))
    return train_loaders, val_loaders, test_loaders


def crop(x_train, y_train, num_clients):
    length = x_train.shape[0] // num_clients
    subX = []
    subY = []
    start_index = 0
    for _ in range(num_clients):
        subarrayX = x_train[start_index : start_index + length]
        subX.append(subarrayX)
        subarrayY = y_train[start_index : start_index + length]
        subY.append(subarrayY)
        start_index += length
    return subX, subY


def quantity_skew(x_train, y_train, num_clients):
    lengths = quantitySkew(x_train.shape[0], num_clients)
    subX = []
    subY = []
    start_index = 0
    for num in lengths:
        subarrayX = x_train[start_index : start_index + num]
        subX.append(subarrayX)
        subarrayY = y_train[start_index : start_index + num]
        subY.append(subarrayY)
        start_index += num
    return subX, subY


# def label_non_equal_skew(x_train, y_train, num_clients):
#     group_samples_X = [[] for _ in range(num_clients)]
#     group_samples_Y = [[] for _ in range(num_clients)]

#     # Group samples by label for train set
#     for i, label in enumerate(y_train):
#         if random.random() > 0.5:
#             group_id = label % num_clients  # Assign the group based on the label
#         else:
#             group_id = random.randint(0, num_clients - 1)
#         group_samples_X[group_id].append(x_train[i])
#         group_samples_Y[group_id].append(y_train[i])
#     for i in range(num_clients):
#         for j in range(5):
#             # count the number of samples for each label
#             count = group_samples_Y[i].count(j)
#             # if the number of samples  is less than 200, delete all of them from the list
#             if count < 150 and random.random() > 0.7:
#                 for _ in range(count):
#                     # remove the first occurrence of the label and its corresponding sample
#                     index = group_samples_Y[i].index(j)
#                     del group_samples_Y[i][index]
#                     del group_samples_X[i][index]
#     combined = list(zip(group_samples_X, group_samples_Y))
#     random.shuffle(combined)
#     group_samples_X[:], group_samples_Y[:] = zip(*combined)

#     return group_samples_X, group_samples_Y


def label_skew(x_train, y_train, num_clients):
    # Ensure inputs are numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Sort the data by label
    sorted_indices = np.argsort(y_train)
    x_train = x_train[sorted_indices]
    y_train = y_train[sorted_indices]

    # Calculate sizes
    total_samples = len(x_train)
    partition_size = total_samples // num_clients
    num_shards = 2 * num_clients
    shard_size = partition_size // 2

    # Create shards
    x_shards = []
    y_shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < num_shards - 1 else total_samples
        x_shards.append(x_train[start_idx:end_idx])
        y_shards.append(y_train[start_idx:end_idx])

    # Shuffle shard order
    shard_indices = np.arange(num_shards)
    np.random.shuffle(shard_indices)

    # Assign 2 shards to clients
    group_samples_X = [[] for _ in range(num_clients)]
    group_samples_Y = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        shard1_idx = shard_indices[2 * i]
        shard2_idx = shard_indices[2 * i + 1]
        group_samples_X[i] = np.concatenate(
            [x_shards[shard1_idx], x_shards[shard2_idx]]
        )
        group_samples_Y[i] = np.concatenate(
            [y_shards[shard1_idx], y_shards[shard2_idx]]
        )

    return group_samples_X, group_samples_Y


def to_tensor(x_train, y_train):
    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    tensor_y = tensor_y.type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    return my_dataset


def gen_loader(trainset, testset, args):
    np.random.seed(args.seed)
    train_loaders = []
    val_loaders = []
    val_ratio = 0.1
    for dataset in trainset:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(args.seed)
        )
        train_loaders.append(
            DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        )
        val_loaders.append(DataLoader(ds_val, batch_size=args.batch_size))
    test_loaders = DataLoader(testset, batch_size=args.batch_size)
    return train_loaders, val_loaders, test_loaders


def get_cifar10(dataset_root, args):
    if args.model == "cifar10_cnn":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        raise ValueError("this nn for cifar10 not implemented")
    trainset = datasets.CIFAR10(
        os.path.join(dataset_root, "cifar10"),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = datasets.CIFAR10(
        os.path.join(dataset_root, "cifar10"),
        train=False,
        download=True,
        transform=transform_test,
    )
    print(type(trainset))
    return gen_loader("cifar10", trainset, testset, args)


def get_gquic(args):
    train_dir = f"/home/bkcs/HDD/Parallel-HierFL/data/GQUIC_small/Train/GQUIC_train_{args.byte_number}.feather"
    test_dir = f"/home/bkcs/HDD/Parallel-HierFL/data/GQUIC_small/Test/GQUIC_test_{args.byte_number}.feather"
    # train_dir = f"/Users/robert/dev/AINI/HierFL/data/GQUIC_small/Train/GQUIC_train_{args.byte_number}.feather"
    # test_dir = f"/Users/robert/dev/AINI/HierFL/data/GQUIC_small/Test/GQUIC_test_{args.byte_number}.feather"
    train = pd.read_feather(train_dir)
    test = pd.read_feather(test_dir)
    x_train, y_train = load_data_set(train, 2103, args)
    x_test, y_test = load_data_set(test, 33, args)
    if args.iid == 0:
        if args.skewness == "quantity":
            x_train, y_train = quantity_skew(x_train, y_train, args.num_clients)
        elif args.skewness == "label":
            x_train, y_train = label_skew(x_train, y_train, args.num_clients)
    else:
        x_train, y_train = crop(x_train, y_train, args.num_clients)
    printNumberOfSample(y_train, args)
    # covert x_train and y_train to dataset
    trainset = []
    for i in range(args.num_clients):
        trainset.append(to_tensor(x_train[i], y_train[i]))
    testset = to_tensor(x_test, y_test)
    return gen_loader(trainset, testset, args)


def most_frequent(List):
    return max(set(List), key=List.count)


def load_data_set(data, seed, args):
    flows = data.groupby("flow_id")["Label"].apply(list).to_dict()
    true_label = []
    for flow in flows:
        true_label.append(most_frequent(flows[flow]))

    true_label = np.array(true_label)
    true_dataset = data.drop(["Label", "flow_id"], axis=1).to_numpy() / 255
    true_dataset = true_dataset.reshape(-1, args.num_packets, args.num_features)
    true_dataset = np.expand_dims(true_dataset, -1)

    true_set = []
    for i in range(true_dataset.shape[0]):
        true_set.append(true_dataset[i].transpose(2, 0, 1))
    true_set = np.array(true_set)

    idx = np.arange(true_set.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    true_set = true_set[idx]
    true_label = true_label[idx]
    return true_set, true_label


if __name__ == "__main__":
    args = args_parser()
    train_loaders, val_loaders, test_loaders = get_dataset(args)
    print(
        f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way with {args.skewness} skewness"
    )
    # client_labels = []
    # # Calculate the distribution of labels in test set
    # labels = []
    # for _, label in test_loaders:
    #     labels.extend(label.tolist())
    # labels = np.array(labels)
    # unique, counts = np.unique(labels, return_counts=True)
    # print(f"Test set has {len(test_loaders.dataset)} samples")
    # print(dict(zip(unique, counts)))

    # # plot the data distribution of the test set
    # fig, ax = plt.subplots()
    # ax.bar(unique, counts)
    # ax.set_xlabel("Label")
    # ax.set_ylabel("Number of Samples")
    # ax.set_title("Label distribution")
    # plt.savefig("test_label_distribution.png")

    # num_classes = {k: 0 for k in range(5)}
    # for i in range(args.num_clients):
    #     train_loader = train_loaders[i]
    #     # calculate distribution of labels in train_loader
    #     labels = []
    #     for _, label in train_loader:
    #         labels.extend(label.tolist())
    #     labels = np.array(labels)
    #     unique, counts = np.unique(labels, return_counts=True)
    #     for k, v in dict(zip(unique, counts)).items():
    #         num_classes[k] += v
    #     print(f"Client {i} has {len(train_loader.dataset)} samples")
    #     client_labels.append(dict(zip(unique, counts)))

    # print(sum(num_classes.values()))
    # # plot the distribution of labels in the trainset
    # fig, ax = plt.subplots()
    # ax.bar(num_classes.keys(), num_classes.values())
    # ax.set_xlabel("Label")
    # ax.set_ylabel("Number of Samples")
    # ax.set_title("Label distribution")
    # plt.savefig("train_label_distribution.png")

    # # Plot data distribution of clients in a stacked bar diagram
    # fig, ax = plt.subplots()
    # client_names = [f"Client {i}" for i in range(args.num_clients)]
    # labels = list(client_labels[0].keys())
    # # Identify all unique labels across all clients
    # all_labels = set()
    # for client_label in client_labels:
    #     all_labels.update(client_label.keys())
    # all_labels = sorted(list(all_labels))  # Sort for consistent ordering

    # # Ensure each client has a count for each label
    # uniform_client_labels = []
    # for client_label in client_labels:
    #     uniform_label_counts = {
    #         label: client_label.get(label, 0) for label in all_labels
    #     }
    #     uniform_client_labels.append(list(uniform_label_counts.values()))

    # # create the data array with uniform shape
    # data = np.array(uniform_client_labels)

    # ax.bar(client_names, data[:, 0], label=labels[0])
    # for i in range(1, len(labels)):
    #     ax.bar(
    #         client_names,
    #         data[:, i],
    #         bottom=np.sum(data[:, :i], axis=1),
    #         label=labels[i],
    #     )

    # ax.set_xlabel("Clients")
    # ax.set_ylabel("Number of Samples")
    # ax.set_title("Data Distribution of Clients")
    # ax.legend()
    # # save the plot
    # plt.savefig("data_distribution.png")
