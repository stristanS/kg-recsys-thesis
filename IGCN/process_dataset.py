from IGCN.dataset import get_dataset
from IGCN.utils import set_seed
from IGCN import dataset as data_module
from IGCN import model as model_module
from IGCN import trainer as trainer_module
import sys


def process_dataset(name):
    sys.modules['dataset'] = data_module
    sys.modules['model'] = model_module
    sys.modules['trainer'] = trainer_module

    # os.chdir("/content/gdrive/MyDrive/Graph_RecSys_repo/IGCN")
    dataset_config = {'name': name + 'Dataset', 'path': './IGCN/data/Movielens/time/',
                      'device': 'cpu', 'split_ratio': [0.7, 0.1, 0.2], 'min_inter': 10}
    dataset = get_dataset(dataset_config)
    dataset.output_dataset('./IGCN/data/' + name + '/time')
    for i in range(5):
        set_seed(2021 + 2 ** i)
        dataset.shuffle = True
        dataset.generate_data()
        dataset.output_dataset('./IGCN/data/' + name + '/' + str(i))


def main():
    process_dataset('Movielens')

