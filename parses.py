import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['COCO-Stuff', 'CityScapes-stuff'], default='CityScapes-stuff')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default='true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--lr_RL', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epochs_decay', type=int, default=100)
    # parser.add_argument('--lambdaR', type=float, default=0.001)
    # parser.add_argument('--lambda_Lp', type=float, default=10)
    # parser.add_argument('--lambda_Ls', type=float, default=0.99)
    parser.add_argument('--save_epochs', type=int, default=25)
    parser.add_argument('--experiment_name', type=str, default='cityscapes')
    parser.add_argument('--log_iters', type=int, default=100)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--n_class', type=int, default=19)

    return parser.parse_args()
