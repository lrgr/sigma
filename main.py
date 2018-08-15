from utils.execute_command import main
import argparse, sys
import time


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--threshold', type=int, required=True)
    parser.add_argument('-b', '--batch', type=int, required=False, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=560)
    parser.add_argument('-iter', '--max_iter', type=int, required=False, default=None)
    parser.add_argument('-eps', '--epsilon', type=int, required=False, default=None)
    parser.add_argument('-rand', '--random_state', type=int, required=False, default=5733)
    parser.add_argument('-o', '--out_dir', type=str, required=False, default='results')
    return parser


if __name__ == '__main__':
    tic = time.clock()
    args = get_parser().parse_args(sys.argv[1:])
    command = args.command
    model = args.model
    threshold = args.threshold
    batch = args.batch
    batch_size = args.batch_size
    max_iterations = args.max_iter
    epsilon = args.epsilon
    random_state = args.random_state
    out_dir = args.out_dir
    main(command, model, threshold, batch, batch_size, max_iterations, epsilon, random_state, out_dir)
    print(time.clock() - tic)
