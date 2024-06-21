import argparse
import torch

num_communication = 300
num_edge_aggregation = 2
num_local_update = 15
num_clients = 4


def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        "--dataset",
        type=str,
        default="gquic",
        help="name of the dataset: mnist, cifar10, gquic",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gquic_cnn",
        help="name of model. gquic: gquic_cnn mnist: mnist_cnn; cifar10: cifar10_cnn",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=1,
        help="input channels. mnist:1, cifar10 :3",
    )
    parser.add_argument(
        "--output_channels", type=int, default=10, help="output channels"
    )
    # nn training hyper parameter
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size when trained on client"
    )
    parser.add_argument(
        "--num_communication",
        type=int,
        default=num_communication,
        help="number of communication rounds with the cloud server",
    )
    parser.add_argument(
        "--num_local_update",
        type=int,
        default=num_local_update,
        help="number of local update (tau_1)",
    )
    parser.add_argument(
        "--num_edge_aggregation",
        type=int,
        default=num_edge_aggregation,
        help="number of edge aggregation (tau_2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate of the Adam when trained on client",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="verbose for print progress bar"
    )
    # setting for federeated learning
    parser.add_argument(
        "--iid",
        type=int,
        default=0,
        help="distribution of the data, 1,0, -2(one-class)",
    )
    parser.add_argument(
        "--edgeiid",
        type=int,
        default=1,
        help="distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)",
    )
    parser.add_argument(
        "--frac", type=float, default=1, help="fraction of participated clients"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=num_clients,
        help="number of all available clients",
    )
    parser.add_argument("--num_edges", type=int, default=2, help="number of edges")
    parser.add_argument("--seed", type=int, default=42, help="random seed (defaul: 42)")
    parser.add_argument(
        "--dataset_root", type=str, default="data", help="dataset root folder"
    )
    parser.add_argument(
        "--show_dis", type=int, default=0, help="whether to show distribution"
    )
    parser.add_argument(
        "--classes_per_client",
        type=int,
        default=2,
        help="under artificial non-iid distribution, the classes per client",
    )

    parser.add_argument("--mtl_model", default=0, type=int)
    parser.add_argument("--global_model", default=1, type=int)
    parser.add_argument("--local_model", default=0, type=int)

    parser.add_argument("--edge_port", default=40001, help="edge port number", type=int)
    parser.add_argument(
        "--socket_volumn", default=1048576, help="socket volumn", type=int
    )

    # gquic dataset
    parser.add_argument("--byte_number", default="256", help="byte number", type=str)
    parser.add_argument("--num_packet", default=20, help="number of packets", type=int)
    parser.add_argument(
        "--num_feature", default=256, help="number of feature", type=int
    )
    parser.add_argument("--num_class", default=5, help="number of class", type=int)

    parser.add_argument(
        "--apply_algorithm", default=1, help="apply algorithm", type=int
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
