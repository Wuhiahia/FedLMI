import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

from flcore.servers.serverFedLMI import FedLMI
from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len = 200

hidden_dim = 32


def run(args):
    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024, dropout_rate=args.dropout_rate).to(args.device)
            elif args.dataset[:5] == "cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, dropout_rate=args.dropout_rate).to(args.device)
            elif args.dataset[:7] == "Fashion":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024, dropout_rate=args.dropout_rate).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816, dropout_rate=args.dropout_rate).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedLMI":
            server = FedLMI(args, i)
        else:
            raise NotImplementedError

        server.train()

        # torch.cuda.empty_cache()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=150)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedLMI")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--dropout_rate", type=float, default=0.1,
                        help="The rate of dropout")

    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")
    parser.add_argument('-hn', "--h5_name", type=str, required=True,
                        help="Prefix for result file names")

    parser.add_argument('-sr', '--sample_ratio', type=float, default=0.9,
                        help='Ratio of training samples to use in each local round (default: 0.9)')
    parser.add_argument('-wdb', "--wd_base", type=float, default=0.01,
                        help="Base weight decay (controls regularization strength, single parameter)")
    parser.add_argument('-dpb', "--dp_base", type=float, default=0.2,
                        help="Base dropout rate (controls dropout strength, single parameter)")
    parser.add_argument('-mt', "--mapping_type", type=str, default="exp", choices=["base", "line", "log", "exp"],
                        help="Type of mapping for regularization parameters: base (no mapping), linear, log, exp (default: linear)")

    parser.add_argument('-dh', '--disable_hwva', type=bool, default=False)
    parser.add_argument('-dd', '--disable_ddwa', type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    # torch.cuda.set_device(int(args.device_id))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    else:
        print("=" * 50 + "device is GPU" + "="*50)

    run(args)
    "python main.py -hn file_name -data Fashion-MNIST_20client_01dir -nb 10 -nc 20 -gr 100"
