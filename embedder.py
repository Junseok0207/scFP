import os
import torch
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
# from metrics import mse, poisson
from argument import printConfig, config2string
from misc.utils import drop_data
# from data.utils import write_h5ad
from sklearn.metrics import mean_squared_error
import scipy

from sklearn.cluster import KMeans
from misc.utils import imputation_error, cluster_acc
from sklearn.metrics.cluster import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


from sklearn.preprocessing import LabelEncoder

class embedder:
    def __init__(self, args):
        self.args = args
        printConfig(args)
        self.config_str = config2string(args)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

        self.data_path = f'dataset/{self.args.name}.h5ad'
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        self.result_path = f'result/{self.args.name}.txt'
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

        self._init_dataset()

    def _init_dataset(self):

        self.adata = sc.read(self.data_path)
        if self.adata.obs['celltype'].dtype != int:
            self.label_encoding()

        self.preprocess(HVG=self.args.HVG, size_factors=self.args.sf, logtrans_input=self.args.log, normalize_input=self.args.normal)
        self.adata = drop_data(self.adata, rate=self.args.drop_rate)

    def label_encoding(self):
        label_encoder = LabelEncoder()
        celltype = self.adata.obs['celltype']
        celltype = label_encoder.fit_transform(celltype)
        self.adata.obs['celltype'] = celltype

    def preprocess(self, HVG=2000, size_factors=True, logtrans_input=True, normalize_input=False):

        sc.pp.filter_cells(self.adata, min_counts=1)
        sc.pp.filter_genes(self.adata, min_counts=1)

        variance = np.array(self.adata.X.todense().var(axis=0))[0]
        hvg_gene_idx = np.argsort(variance)[-int(HVG):]
        self.adata = self.adata[:,hvg_gene_idx]

        self.adata.raw = self.adata.copy()

        if size_factors:
            sc.pp.normalize_per_cell(self.adata)
            self.adata.obs['size_factors'] = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
        else:
            self.adata.obs['size_factors'] = 1.0

        if logtrans_input:
            sc.pp.log1p(self.adata)

        if normalize_input:
            sc.pp.scale(self.adata)


    def evaluate(self):
        
        X_imputed = self.adata.obsm['denoised']
        if self.args.drop_rate != 0.0:
            X_test = self.adata.obsm["test"]
            drop_index = self.adata.uns['drop_index']

            rmse, median_l1_distance, cosine_similarity = imputation_error(X_imputed, X_test, drop_index)

        # clustering
        celltype = self.adata.obs['celltype'].values
        n_cluster = np.unique(celltype).shape[0]

        ### Imputed
        kmeans = KMeans(n_cluster, n_init=20, random_state=self.args.seed)
        y_pred = kmeans.fit_predict(X_imputed)

        imputed_silhouette = silhouette_score(X_imputed, y_pred)
        imputed_ari = adjusted_rand_score(celltype, y_pred)
        imputed_nmi = normalized_mutual_info_score(celltype, y_pred)
        imputed_ca, imputed_ma_f1, imputed_mi_f1 = cluster_acc(celltype, y_pred)

        ### Reduced
        reduced = self.adata.obsm['reduced']
        kmeans = KMeans(n_cluster, n_init=20, random_state=self.args.seed)
        y_pred = kmeans.fit_predict(reduced)

        reduced_silhouette = silhouette_score(reduced, y_pred)
        reduced_ari = adjusted_rand_score(celltype, y_pred)
        reduced_nmi = normalized_mutual_info_score(celltype, y_pred)
        reduced_ca, reduced_ma_f1, reduced_mi_f1 = cluster_acc(celltype, y_pred)
        

        print(f"Dataset: {self.args.name}, Drop: {self.args.drop_rate}, Alpha: {self.args.alpha}")
        print()
        if self.args.drop_rate != 0.0:
            print("RMSE : {:.4f} / Median L1 Dist : {:.4f} / Cos-Sim : {:.4f}".format(rmse, median_l1_distance, cosine_similarity))
        
        print("Imputed --> ARI : {:.4f} / NMI : {:.4f} / ca : {:.4f}\n".format(imputed_ari, imputed_nmi, imputed_ca))
        print("Reduced --> ARI : {:.4f} / NMI : {:.4f} / ca : {:.4f}\n".format(reduced_ari, reduced_nmi, reduced_ca))

        with open(self.result_path, 'a+') as f:
            f.write("{}\n".format(self.config_str))
            if self.args.drop_rate != 0.0:
                f.write("Rate {} -> RMSE : {:.4f} / Median L1 Dist : {:.4f} / Cos-Sim : {:.4f}\n".format(self.args.drop_rate, rmse, median_l1_distance, cosine_similarity))
            f.write("(Imputed) Rate {} -> ARI : {:.4f} / NMI : {:.4f} / Silhouette : {:.4f} / ca : {:.4f} / ma-f1 : {:.4f} / mi-f1 : {:.4f}\n".format(self.args.drop_rate, imputed_ari, imputed_nmi, imputed_silhouette, imputed_ca, imputed_ma_f1, imputed_mi_f1))            
            f.write("(Reduced) Rate {} -> ARI : {:.4f} / NMI : {:.4f} / Silhouette : {:.4f} / ca : {:.4f} / ma-f1 : {:.4f} / mi-f1 : {:.4f}\n".format(self.args.drop_rate, reduced_ari, reduced_nmi, reduced_silhouette, reduced_ca, reduced_ma_f1, reduced_mi_f1))            
            f.write("\n")
        
        if self.args.drop_rate != 0.0:
            return [rmse, median_l1_distance, cosine_similarity, imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca]
        else:
            return [imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca]
