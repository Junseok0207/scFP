import torch
import copy
from embedder import embedder
from misc.graph_construction import knn_graph
import numpy as np
from sklearn.decomposition import PCA


class scFP_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.args.n_nodes = self.adata.obsm["train"].shape[0]
        self.args.n_feat = self.adata.obsm["train"].shape[1]

    def train(self):
        cell_data = torch.Tensor(self.adata.obsm["train"]).to(self.args.device)

        #  Hard FP
        print('Start Hard Feature Propagation ...!')
        edge_index, edge_weight = knn_graph(cell_data, self.args.k)
        self.model = FeaturePropagation(num_iterations=self.args.iter, mask=True, alpha=0.0)
        self.model = self.model.to(self.device)

        denoised_matrix = self.model(cell_data, edge_index, edge_weight)

        # Soft FP
        print('Start Soft Feature Propagation ...!')
        edge_index_new, edge_weight_new = knn_graph(denoised_matrix, self.args.k)
        self.model = FeaturePropagation(num_iterations=self.args.iter, mask=False, alpha=self.args.alpha)        
        self.model = self.model.to(self.device)

        denoised_matrix = self.model(denoised_matrix, edge_index_new, edge_weight_new)

        # reduced
        pca = PCA(n_components = 32)
        denoised_matrix = denoised_matrix.detach().cpu().numpy()
        reduced = pca.fit_transform(denoised_matrix)
            
        self.adata.obsm['denoised'] = denoised_matrix
        self.adata.obsm['reduced'] = reduced
        
        return self.evaluate()
        
class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations, mask, alpha=0.0):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        original_x = copy.copy(x)
        nonzero_idx = torch.nonzero(x)
        nonzero_i, nonzero_j = nonzero_idx.t()

        out = x
        n_nodes = x.shape[0]
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
        adj = adj.float()
        
    
        res = (1-self.alpha) * out
        for _ in range(self.num_iterations):
            out = torch.sparse.mm(adj, out)
            if self.mask:
                out[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
            else:
                out.mul_(self.alpha).add_(res) 
                
        return out

