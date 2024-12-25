import argparse
import torch

num_communication = 300
num_edge_aggregation = 2
num_local_update = 50
num_clients = 6
skewness = "quantity"


def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        "--dataset",
        type=str,
        default="gquic",
        help="name of the dataset: cifar10, gquic",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gquic_cnn",
        help="name of model. gquic: gquic_cnn; cifar10: cifar10_cnn",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="input channels. cifar10 :3",
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
        default=0.0001,
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
        help="distribution of the data among clients, 0 (non-iid), 1 (iid)",
    )
    parser.add_argument(
        "--skewness",
        type=str,
        default=skewness,
        help="type of data skewness: quantity, label, none",
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
    parser.add_argument("--num_edges", type=int, default=3, help="number of edges")
    parser.add_argument("--seed", type=int, default=39, help="random seed (defaul: 42)")
    parser.add_argument(
        "--dataset_root", type=str, default="data", help="dataset root folder"
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
    parser.add_argument("--num_packets", default=20, help="number of packets", type=int)
    parser.add_argument(
        "--num_features", default=256, help="number of features", type=int
    )
    parser.add_argument("--num_classes", default=5, help="number of classes", type=int)

    parser.add_argument(
        "--algorithm",
        default="sac",
        help="type of drl algorithm: sac, dql_epsilon, dql_ucb1, dql_softmax, ddpg_epsilon, ddpg_ucb1, ppo, none",
        type=str,
    )
    parser.add_argument("--alpha", default=0.01, help="alpha", type=float)
    parser.add_argument("--is_server", default=0, help="is server", type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
