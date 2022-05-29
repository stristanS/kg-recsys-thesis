import argparse
from PinSage.process_movielens1m import process_movieslens
from IGCN.process_dataset import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pinsage')
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path


    if args.model == 'pinsage':
        process_movieslens(directory, output_path)
    elif args.model == 'igcn':
        main()