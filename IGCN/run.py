from IGCN.dataset import get_dataset
from IGCN.model import get_model
from IGCN.trainer import get_trainer
import torch
from IGCN.utils import init_run
from tensorboardX import SummaryWriter
from IGCN.config import get_gowalla_config, get_yelp_config, get_amazon_config, get_movielens_config
from IGCN import dataset as data_module
from IGCN import model as model_module
from IGCN import trainer as trainer_module
import sys

class IGCNModel:
    def __init__(self):
        self.trainer = None
    def main(self):
        # log_path = '/content/gdrive/MyDrive/Graph_RecSys_repo/IGCN/logs'
        sys.modules['dataset'] = data_module
        sys.modules['model'] = model_module
        sys.modules['trainer'] = trainer_module

        log_path = '/content/kg-recsys/IGCN/logs'
        init_run(log_path, 2021)

        device = torch.device('cpu')
        config = get_movielens_config(device)
        dataset_config, model_config, trainer_config = config[2]
        dataset_config['path'] = dataset_config['path'][:-4] + str('time')

        writer = SummaryWriter(log_path)
        dataset = get_dataset(dataset_config)
        model = get_model(model_config, dataset)
        trainer = get_trainer(trainer_config, dataset, model)
        trainer.train(verbose=True, writer=writer)
        self.trainer = trainer
        writer.close()
        # results, _ = trainer.eval('test')
        # print('Test result. {:s}'.format(results))

    def eval(self):
        results = self.trainer.eval('test')
        return results
        # print('Test result. {:s}'.format(results))


