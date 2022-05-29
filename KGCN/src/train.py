import tensorflow as tf
import numpy as np
from model import KGCN
from tqdm import tqdm
import pickle


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, eval_data, n_item)
    user_list_test, train_record_test, test_record_test, item_set_test, k_list_test = topk_settings(show_topk,
                                                                                                    train_data,
                                                                                                    test_data, n_item)
    best_recall = -np.inf
    max_patience = 5
    patience = max_patience
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            print('train_data.shape[0]', train_data.shape[0])
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')
            if recall[-1] > best_recall:
                best_recall = recall[-1]
                patience = max_patience
                print('Recall@20 {}'.format(recall[-1]))
            else:
                print('Recall@20 {}'.format(recall))
                patience -= 1
                if patience <= 0:
                    print('Early stopping!')
                    break
        predictions = predict(
            sess, model, user_list_test, train_record_test, test_record_test, item_set_test, args.batch_size, k=20)


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 6040
        k_list = [1, 2, 5, 10, 20]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    # print('model.user_indices', data[start:end, 0])
    # print('model.item_indices', data[start:end, 1])
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        try:
            auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
            auc_list.append(auc)
            f1_list.append(f1)
            start += batch_size
        except ValueError:
            pass
        return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    # user_list.sort()

    for user in tqdm(user_list):
        # print('user', user)
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        # while start + batch_size <= len(test_item_list):
        # items, scores = model.get_scores(sess, {model.user_indices: [user],
        #                                         model.item_indices: item_set})
        # for item, score in zip(items, scores):
        #     item_score_map[item] = score
        # start += batch_size
        # print('len', len(test_item_list))
        while start + batch_size <= len(test_item_list):
            # print(1)
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            # print('len(items), len(scores)', len(set(items)), len(set(scores)))
            # print()
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        # print('item_sorted', item_sorted)
        # print('user', user)
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def predict(sess, model, user_list, train_record, test_record, item_set, batch_size, k=20):
    user_list.sort()
    predictions = []

    for user in tqdm(user_list):
        # print('user', user)
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        # while start + batch_size <= len(test_item_list):
        # items, scores = model.get_scores(sess, {model.user_indices: [user],
        #                                         model.item_indices: item_set})
        # for item, score in zip(items, scores):
        #     item_score_map[item] = score
        # start += batch_size
        # print('len', len(test_item_list))
        while start + batch_size <= len(test_item_list):
            # print(1)
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            # print('len(items), len(scores)', len(set(items)), len(set(scores)))
            # print()
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        pred_top_k = item_sorted[:k]
        with open('/content/KGCN/data/movie/ml_kgcn_mapping.pkl_', 'rb') as f:
            mapping_kgcn_2_old = pickle.load(f)
        with open('/content/KGCN/data/movie/ml_mapping_for_all.pkl_', 'rb') as f:
            mapping_old_2_general_val = pickle.load(f)

        mapping_kgcn_2_old2 = {y: x for x, y in mapping_kgcn_2_old.items()}
        pred_old_id = [int(mapping_kgcn_2_old2[k]) for k in pred_top_k]

        # mapping_old_2_general_val2 = {y: x for x, y in mapping_old_2_general_val['movie'].items()}

        movie_map = dict(enumerate(mapping_old_2_general_val['movie'].cat.categories))
        movie_map = {value: key for (key, value) in movie_map.items()}

        pred_general_val_id = [int(movie_map[k]) for k in pred_old_id]

        # if user == 0 or user == 57 or user == 2468:
        #     test = test_record[user]
        #     test_old_id = [int(mapping_kgcn_2_old2[k]) for k in test]
        #     test_general_val_id = [int(movie_map[k]) for k in test_old_id]
        #     print('DOUBLE CHECK', user)
        #     print('kg id', test)
        #     print('orig id', test_old_id)
        #     print('pinsage id', test_general_val_id)

        predictions.append(pred_general_val_id)
        # print('item_sorted', item_sorted)
        # print('user', user)
    with open('/content/KGCN/src/KGCN_preds_for_ml.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    return predictions

