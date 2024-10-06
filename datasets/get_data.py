from datasets.prepare_data import get_dataset


def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, valloaders and testloaders
    """
    if args.dataset in ["cifar10", "gquic"]:
        train_loaders, val_loaders, test_loaders = get_dataset(
            dataset_root=args.dataset_root, dataset=args.dataset, args=args
        )
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, val_loaders, test_loaders
