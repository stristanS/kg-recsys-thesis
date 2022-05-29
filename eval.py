import numpy as np
import torch
import pickle
import dgl


def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        items = line.split(' ')[1:]
        items = [int(item) for item in items]
        # if items:
        #     self.n_items = max(self.n_items, max(items) + 1)
        data.append(items)
    return data

def calculate_metrics(eval_data, rec_items, topks):
    results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
    hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
    for user in range(rec_items.shape[0]):
        for item_idx in range(rec_items.shape[1]):
            if rec_items[user, item_idx] in eval_data[user]:
                hit_matrix[user, item_idx] = 1.
    eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

    for k in topks:
        hit_num = np.sum(hit_matrix[:, :k], axis=1)
        precisions = hit_num / k
        with np.errstate(invalid='ignore'):
            recalls = hit_num / eval_data_len

        max_hit_num = np.minimum(eval_data_len, k)
        max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
        for user, num in enumerate(max_hit_num):
            max_hit_matrix[user, :num] = 1.
        denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
        dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
        idcgs = np.sum(max_hit_matrix / denominator, axis=1)
        with np.errstate(invalid='ignore'):
            ndcgs = dcgs / idcgs

        user_masks = (max_hit_num > 0)
        results['Precision'][k] = precisions[user_masks].mean()
        results['Recall'][k] = recalls[user_masks].mean()
        results['NDCG'][k] = ndcgs[user_masks].mean()
    return results


def eval(recommendations, groud_truth_path, top_k):
        eval_data = read_data(groud_truth_path)
        metrics = calculate_metrics(eval_data, recommendations, top_k)
        precison = ''
        recall = ''
        ndcg = ''
        for k in top_k:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics




