from datasets.prepare_data import get_dataset


def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ["mnist", "cifar10", "gquic"]:
        train_loaders, val_loaders, test_loaders = get_dataset(
            dataset_root="data", dataset=args.dataset, args=args
        )
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, val_loaders, test_loaders
