# cluster from AttentionXML

import os
import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    sparse_labels = [i.replace('\n', '').split() for i in open(label_file)]
    return normalize(sparse_x), np.array(sparse_labels)

def build_tree_by_level(sparse_data_x, sparse_data_y, eps: float, max_leaf: int, levels: list, groups_path):
    print('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_y = mlb.fit_transform(sparse_labels)
    joblib.dump(mlb, groups_path+'mlb')
    print('Getting Labels Feature')
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))
    print(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            print(F'{groups_path}-Level-{i}.npy')
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    while q:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            print(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
            else:
                np.save(F'{groups_path}-last.npy', np.asarray(labels_list))
        q = next_q
    print('Finish Clustering')
    return mlb


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--tree', action='store_true')
parser.add_argument('--id', type=str, required=False, default='0')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    if dataset == '670k':
        mlb = build_tree_by_level('./data/Amazon-670K/train_v1.txt', 
                                  './data/Amazon-670K/train_labels.txt',
                                  1e-4, 100, [], './data/Amazon-670K/label_group'+args.id)
        groups = np.load(f'./data/Amazon-670K/label_group{args.id}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'./data/Amazon-670K/label_group{args.id}.npy', np.array(new_group))
    elif dataset == '500k':
        mlb = build_tree_by_level('./data/Wiki-500K/train.txt', 
                                  './data/Wiki-500K/train_labels.txt',
                                  1e-4, 100, [], './data/Wiki-500K/groups')
        groups = np.load(f'./data/Wiki-500K/groups-last{args.id}.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'./data/Wiki-500K/label_group{args.id}.npy', np.array(new_group))
