import os
import argparse
import facedetection


def dir_path(arg):
    if os.path.isdir(arg):
        return arg
    raise NotADirectoryError(arg)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-frequency', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of worker threads for data loading')
    parser.add_argument('path', type=dir_path,
                        help='path to WIDER Face dataset')
    return parser.parse_args()


def main():
    args = get_args()
    facedetection.train(args)


if __name__ == '__main__':
    main()
