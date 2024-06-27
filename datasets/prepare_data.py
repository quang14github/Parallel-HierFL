"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""

import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    random_split,
    TensorDataset,
)
from options import args_parser


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels  # labels are stored in the 'targets' attribute

    def __getitem__(self, index):
        # Return the data and label for the given index
        return self.data[index], self.targets[index]

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)


def get_dataset(dataset_root, dataset, args):
    train_loaders = []
    val_loaders = []
    test_loaders = []
    if dataset == "mnist":
        train_loaders, val_loaders, test_loaders = get_mnist(dataset_root, args)
    elif dataset == "cifar10":
        train_loaders, val_loaders, test_loaders = get_cifar10(dataset_root, args)
    elif dataset == "gquic":
        train_loaders, val_loaders, test_loaders = get_gquic(dataset_root, args)
    else:
        raise ValueError("Dataset `{}` not found".format(dataset))
    return train_loaders, val_loaders, test_loaders


def gen_loader(dataset, trainset, testset, args):
    np.random.seed(args.seed)
    partition_size = int(len(trainset) / args.num_clients)
    lengths = [partition_size] * args.num_clients
    if args.iid:
        datasets = random_split(
            trainset, lengths, torch.Generator().manual_seed(args.seed)
        )
    else:
        if args.balance:
            shard_size = int(partition_size / 2)
            if not isinstance(trainset.targets, torch.Tensor):
                trainset.targets = torch.tensor(trainset.targets)
            idxs = trainset.targets.argsort()
            sorted_data = Subset(trainset, idxs)
            tmp = []
            for idx in range(args.num_clients * 2):
                tmp.append(
                    Subset(
                        sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1))
                    )
                )
            idxs_list = torch.randperm(
                args.num_clients * 2, generator=torch.Generator().manual_seed(args.seed)
            )
            datasets = [
                ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
                for i in range(args.num_clients)
            ]
        else:
            shard_size = int(partition_size / 4)
            if not isinstance(trainset.targets, torch.Tensor):
                trainset.targets = torch.tensor(trainset.targets)
            idxs = trainset.targets.argsort()
            sorted_data = Subset(trainset, idxs)
            tmp = []
            for idx in range(args.num_clients * 4):
                tmp.append(
                    Subset(
                        sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1))
                    )
                )
            idxs_list = torch.randperm(
                args.num_clients * 4, generator=torch.Generator().manual_seed(args.seed)
            ).tolist()
            datasets = []
            for i in range(args.num_clients):
                # get a random number of shards from 1 to 4 for each client, select random shards in the list and remove from the list
                num_shards = np.random.randint(1, 5)
                shards = []
                for _ in range(num_shards):
                    shard_idx = idxs_list.pop()
                    shards.append(tmp[shard_idx])
                datasets.append(ConcatDataset(shards))
            while len(idxs_list) > 0:
                idx = idxs_list.pop()
                rand_client = np.random.randint(0, args.num_clients)
                datasets[rand_client] = ConcatDataset((datasets[rand_client], tmp[idx]))
    train_loaders = []
    val_loaders = []
    val_ratio = 0.1
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(args.seed)
        )
        train_loaders.append(
            DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        )
        val_loaders.append(DataLoader(ds_val, batch_size=args.batch_size))
    test_loaders = DataLoader(testset, batch_size=args.batch_size)
    return train_loaders, val_loaders, test_loaders


def get_mnist(dataset_root, args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainset = datasets.MNIST(
        os.path.join(dataset_root, "mnist"),
        train=True,
        download=True,
        transform=transform,
    )
    testset = datasets.MNIST(
        os.path.join(dataset_root, "mnist"),
        train=False,
        download=True,
        transform=transform,
    )
    return gen_loader("mnist", trainset, testset, args)


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


def get_gquic(dataset_root, args):
    train_dir = os.path.join(
        dataset_root, f"GQUIC_small/Train/GQUIC_train_{args.byte_number}.feather"
    )
    test_dir = os.path.join(
        dataset_root, f"GQUIC_small/Test/GQUIC_test_{args.byte_number}.feather"
    )
    train = pd.read_feather(train_dir)
    test = pd.read_feather(test_dir)
    x_train, y_train = load_data_set(train, 2103, args)
    x_test, y_test = load_data_set(test, 33, args)
    trainset = CustomDataset(x_train, y_train)
    testset = CustomDataset(x_test, y_test)
    return gen_loader("gquic", trainset, testset, args)


def most_frequent(List):
    return max(set(List), key=List.count)


def load_data_set(data, seed, args):
    flows = data.groupby("flow_id")["Label"].apply(list).to_dict()
    true_label = []
    for flow in flows:
        true_label.append(most_frequent(flows[flow]))

    true_label = np.array(true_label)
    true_dataset = data.drop(["Label", "flow_id"], axis=1).to_numpy() / 255
    true_dataset = true_dataset.reshape(-1, args.num_packet, args.num_feature)
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
    train_loaders, val_loaders, test_loaders = get_dataset(
        "/Users/robert/dev/AINI/HierFL/datasets", args.dataset, args
    )
    print(
        f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way"
    )
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        # calculate distribution of labels in train_loader
        labels = []
        for _, label in train_loader:
            labels.extend(label.tolist())
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {i} has {len(train_loader.dataset)} samples")
        print(dict(zip(unique, counts)))
    print(sum(len(train_loader.dataset) for train_loader in train_loaders))
