"""
Script that reads from raw MovieLens-1M data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from PinSage.builder import PandasGraphBuilder
from PinSage.data_utils import *

def process_amazon(directory, output_path):
    data = pd.read_csv(directory, index_col = 0)
    data = data.iloc[:, :4]
    data.columns = ['user_id', 'book_id', 'rating', 'read_at']
    data = data.dropna()

    users = data[['user_id']].drop_duplicates()
    tracks = data[['book_id']].drop_duplicates()
    assert tracks['book_id'].value_counts().max() == 1
    # tracks = tracks.astype({'mode': 'int64', 'key': 'int64', 'artist_id': 'category'})
    events = data[['user_id', 'book_id', 'read_at']]
    events['read_at'] = events['read_at'].values.astype('datetime64[s]').astype('int64')

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(tracks, 'book_id', 'book')
    graph_builder.add_binary_relations(events, 'user_id', 'book_id', 'read')
    graph_builder.add_binary_relations(events, 'book_id', 'user_id', 'read-by')

    g = graph_builder.build()
    # print('!!!!!!!!!mapping!!!!!!!!!',mapping)

    float_cols = []
    for col in tracks.columns:
        if col == 'book_id':
            continue
        elif col == 'artist_id':
            g.nodes['book'].data[col] = torch.LongTensor(tracks[col].cat.codes.values)
        elif tracks.dtypes[col] == 'float64':
            float_cols.append(col)
        else:
            g.nodes['book'].data[col] = torch.LongTensor(tracks[col].values)
    # g.nodes['track'].data['song_features'] = torch.FloatTensor(linear_normalize(tracks[float_cols].values))
    g.edges['read'].data['read_at'] = torch.LongTensor(events['read_at'].values)
    g.edges['read-by'].data['read_at'] = torch.LongTensor(events['read_at'].values)

    n_edges = g.number_of_edges('read')
    train_indices, val_indices, test_indices = train_test_split_by_time(events, 'read_at', 'user_id')
    train_g = build_train_graph(g, train_indices, 'user', 'book', 'read', 'read-by')
    assert train_g.out_degrees(etype='read').min() > 0
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, 'user', 'book', 'read')

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': {},
        'item-images': None,
        'user-type': 'user',
        'item-type': 'book',
        'user-to-item-type': 'read',
        'item-to-user-type': 'read-by',
        'timestamp-edge-column': 'read_at',
        }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

