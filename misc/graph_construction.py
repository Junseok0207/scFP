import torch
import torch.nn.functional as F


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).to(raw_graph.device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def post_processing(cur_raw_adj, add_self_loop=True, sym=True, gcn_norm=False):

    if add_self_loop:
        num_nodes = cur_raw_adj.size(0)
        cur_raw_adj = cur_raw_adj + torch.diag(torch.ones(num_nodes)).to(cur_raw_adj.device)
    
    if sym:
        cur_raw_adj = cur_raw_adj + cur_raw_adj.t()
        cur_raw_adj /= 2

    deg = cur_raw_adj.sum(1)

    if gcn_norm:

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        cur_adj = torch.mm(deg_inv_sqrt, cur_raw_adj)
        cur_adj = torch.mm(cur_adj, deg_inv_sqrt)

    else:

        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        cur_adj = torch.mm(deg_inv_sqrt, cur_raw_adj)
    
    return cur_adj

def knn_graph(embeddings, k, gcn_norm=False, sym=True):

    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=1, p=2)
    similarity_graph = torch.mm(embeddings, embeddings.t())
    
    X = top_k(similarity_graph.to(device), k + 1)
    similarity_graph = F.relu(X)

    cur_adj = post_processing(similarity_graph, gcn_norm=gcn_norm, sym=sym)

    sparse_adj = cur_adj.to_sparse()
    edge_index = sparse_adj.indices().detach()
    edge_weight = sparse_adj.values()

    return edge_index, edge_weight
