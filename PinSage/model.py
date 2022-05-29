import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import tqdm

import PinSage.layers as layers
import PinSage.sampler as sampler_module
import PinSage.evaluation as evaluation


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

class ModelClass():
    def __init__(self):
        self.dataset = None
        self.batch_size = None
        self.dataloader_test = None
        self.model = None
        self.device = None

    def train(self, dataset, args):
        self.dataset = dataset
        self.batch_size = args['batch_size']
        g = dataset['train-graph']
        val_matrix = dataset['val-matrix'].tocsr()
        test_matrix = dataset['test-matrix'].tocsr()
        item_texts = dataset['item-texts']
        user_ntype = dataset['user-type']
        item_ntype = dataset['item-type']
        user_to_item_etype = dataset['user-to-item-type']
        timestamp = dataset['timestamp-edge-column']

        device = torch.device(args['device'])
        self.device = device

        # Assign user and movie IDs and use them as features (to learn an individual trainable
        # embedding for each entity)
        g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
        g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))

        # Prepare torchtext dataset and vocabulary
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = torchtext.legacy.data.Field(include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.legacy.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)
        textset = torchtext.legacy.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')

        # Sampler
        batch_sampler = sampler_module.ItemToItemBatchSampler(
            g, user_ntype, item_ntype, args['batch_size'])
        neighbor_sampler = sampler_module.NeighborSampler(
            g, user_ntype, item_ntype, args['random_walk_length'],
            args['random_walk_restart_prob'], args['num_random_walks'], args['num_neighbors'],
            args['num_layers'])
        collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
        dataloader = DataLoader(
            batch_sampler,
            collate_fn=collator.collate_train,
            num_workers=args['num_workers'])
        dataloader_test = DataLoader(
            torch.arange(g.number_of_nodes(item_ntype)),
            batch_size=args['batch_size'],
            collate_fn=collator.collate_test,
            num_workers=args['num_workers'])
        self.dataloader_test = dataloader_test
        dataloader_it = iter(dataloader)

        # Model
        model = PinSAGEModel(g, item_ntype, textset, args['hidden_dims'], args['num_layers']).to(device)
        # Optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args['lr'])

        # For each batch of head-tail-negative triplets...
        for epoch_id in range(args['num_epochs']):
            model.train()
            for batch_id in tqdm.trange(args['batches_per_epoch']):
                pos_graph, neg_graph, blocks = next(dataloader_it)
                # Copy to GPU
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)

                loss = model(pos_graph, neg_graph, blocks).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Evaluate
            self.model = model
            model.eval()
            with torch.no_grad():
                item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args['batch_size'])
                h_item_batches = []
                for blocks in dataloader_test:
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)

                    h_item_batches.append(model.get_repr(blocks))
                h_item = torch.cat(h_item_batches, 0)

                print(evaluation.evaluate_nn(dataset, h_item, args['k'], args['batch_size']))

    def eval(self, k):
        g = self.dataset['train-graph']
        user_ntype = self.dataset['user-type']
        item_ntype = self.dataset['item-type']
        user_to_item_etype = self.dataset['user-to-item-type']
        timestamp = self.dataset['timestamp-edge-column']

        self.model.eval()
        with torch.no_grad():
            h_item_batches = []
            for blocks in self.dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
                h_item_batches.append(self.model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

        rec_engine = evaluation.LatestNNRecommender(
            user_ntype, item_ntype, user_to_item_etype, timestamp, self.batch_size)

        recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
        return recommendations


