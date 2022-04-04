import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(description='sound2image arguments')
    parser.add_argument('--use_cpu', action='store_true')

    args = parser.parse_args()
    args.use_gpu = True
    if args.use_cpu:
        args.use_gpu = False

    return args