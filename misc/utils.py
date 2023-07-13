import os
import copy
import torch
import random
import numpy as np
import scipy.sparse
import scanpy as sc
from sklearn import metrics
from munkres import Munkres
import logging
import sys
import torch.nn.functional as F

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def filter_genes_cells(adata):
    """Remove empty cells and genes."""
    
    if "var_names_all" not in adata.uns:
        # fill in original var names before filtering
        adata.uns["var_names_all"] = adata.var.index.to_numpy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_counts=2)


def drop_data(adata, rate, datatype='real'):
    
    X = adata.X

    if scipy.sparse.issparse(X):
        X = np.array(X.todense())

    if datatype == 'real':
        X_train = np.copy(X)
        i, j = np.nonzero(X)

        ix = np.random.choice(range(len(i)), int(
            np.floor(rate * len(i))), replace=False)
        X_train[i[ix], j[ix]] = 0.0

        drop_index = {'i':i, 'j':j, 'ix':ix}
        adata.uns['drop_index'] = drop_index        
        adata.obsm["train"] = X_train
        adata.obsm["test"] = X

        # for training
        adata.raw.X[i[ix],j[ix]] = 0.0

    elif datatype == 'simul':
        adata.obsm["train"] = X

    return adata


def cosine_similarity(x,y):
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    cos_sim = torch.sum(torch.mul(x,y),1)
    return cos_sim

def cos_sim(x,y):
    sim = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return sim

def imputation_error(X_hat, X, drop_index):
    
    i, j, ix = drop_index['i'], drop_index['j'], drop_index['ix']
    
    all_index = i[ix], j[ix]
    x, y = X_hat[all_index], X[all_index]

    squared_error = (x-y)**2
    absolute_error = np.abs(x - y)

    rmse = np.mean(np.sqrt(squared_error))
    median_l1_distance = np.median(absolute_error)
    cosine_similarity = cos_sim(x,y)
    
    return rmse, median_l1_distance, cosine_similarity


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def cluster_acc(y_true, y_pred):

        #######
        y_true = y_true.astype(int)
        #######

        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)

        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1

        l2 = list(set(y_pred))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print('n_cluster is not valid')
            return

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
                cost[i][j] = len(mps_d)

        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        new_predict = np.zeros(len(y_pred))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(y_true, new_predict)
        # y_true：Like 1d array or label indicator array/sparse matrix (correct) label
        # y_pred：Like a one-dimensional array or label indicator array/sparse matrix predicted labels, returned by the classifier
        
        f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
        f1_micro = metrics.f1_score(y_true, new_predict, average='micro')

        return acc, f1_macro, f1_micro



def setup_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")

    return logger

def set_filename(args):
    # runs = '_n_runs_10' if args.n_runs == 10 else ''
    runs = f'n_runs_{args.n_runs}'
    if args.drop_rate > 0.0:
        logs_path = f'logs_{runs}/imputation/{args.name}'
    else:
        logs_path = f'logs_{runs}/clustering/{args.name}'

    os.makedirs(logs_path, exist_ok=True)
    
    file = f'{logs_path}/scFP.txt'

    return file

def get_gene(x):
    if 'symbol' in x.keys():
        return x['symbol']
    return ''

