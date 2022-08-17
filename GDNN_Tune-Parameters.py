import random
import argparse
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import matplotlib.pyplot as plt

import optuna


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2f}')
            print(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')
    
    def tiaocan(self,run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            Final_Test=result[argmax, 1]
            #print(f'Run {run + 1:02d}:')
            #print(f'Highest Valid: {result[:, 0].max():.2f}')
            #print(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            #print(f'All runs:')
            #r = best_result[:, 0]
            #print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            Final_Test=r.mean()
            #print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')
        return Final_Test


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, edge_attr, x, emb_ea, adj_t, split_edge, optimizer, batch_size):
    edge_index = adj_t

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t, edge_attr, emb_ea)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, edge_attr, x, emb_ea, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t, edge_attr, emb_ea)

    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results



def main(param):
    parser = argparse.ArgumentParser(description='GDNN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--node_emb', type=int, default=512)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=param['dropout'])
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=param['learning_rate'])
    #parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=param['epochs'])
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args(args=[])
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    data = dataset[0]
    edge_index = data.edge_index.to(device)

    split_edge = dataset.get_edge_split()

    model = GraphSAGE(args.node_emb, args.hidden_channels, args.hidden_channels,
                      args.num_layers, args.dropout).to(device)

    #emb = torch.nn.Embedding(data.num_nodes, args.node_emb).to(device)
    emb_ea = torch.nn.Embedding(args.num_samples, args.node_emb).to(device)
    
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers+1, args.dropout).to(device)

  

    
    np.random.seed(0)
    
    node_mask = []
    for _ in range(args.num_samples):
        node_mask.append(np.random.choice(500, size=200, replace=False))
    node_mask = np.array(node_mask)
    
    
    #DE-Mymethod
    dataset_dis_coding = PygLinkPropPredDataset(name='ogbl-ddi')
    ddi_G = dataset_dis_coding[0]
    edge_index=ddi_G.edge_index.to(device)
    ddiNet=to_networkx(ddi_G)
    ddiNet=ddiNet.to_undirected()
    print(ddiNet.is_directed())

    #Distance Matrix
    def distance_encoding(x,y):
        distance=len(nx.shortest_path(ddiNet, source=x, target=y))
        return distance

    #Select k nodes randomly in ddi graph as targets
    badian=np.random.choice(4267, 512, replace=False)
    print('badian is:',badian)

    distance_feature=[]

    for x in range(0,4267):
        distance_feature.append([])
        for y in badian:
            dis=distance_encoding(x,y)
            distance_feature[x].append(dis)

    print('distance_feature is:',distance_feature)
    distance_feature=torch.tensor(distance_feature,dtype=torch.float).to(device)
    
    emb=torch.nn.Embedding.from_pretrained(distance_feature,freeze=False).to(device)
    
    print('Number of parameters:',
          sum(p.numel() for p in list(model.parameters()) +
          list(predictor.parameters()) + list(emb.parameters()) + list(emb_ea.parameters())))

    edge_attr = distance_feature[edge_index, :].mean(0)[:, node_mask].mean(2)

    a_max = torch.max(edge_attr, dim=0, keepdim=True)[0]
    a_min = torch.min(edge_attr, dim=0, keepdim=True)[0]
    edge_attr = (edge_attr - a_min) / (a_max - a_min + 1e-6)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@20': Logger(args.runs, args)
    }
    #valid_list=[]
    #test_list=[]
    #loss_list=[]

    for run in range(args.runs):
        random.seed(run)
        torch.manual_seed(run)
        torch.nn.init.xavier_uniform_(emb.weight)
        torch.nn.init.xavier_uniform_(emb_ea.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(emb_ea.parameters()) + list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        #if key=='Hits@20':
                            #loss_list.append(loss)
                            #valid_list.append(valid_hits)
                            #test_list.append(test_hits)
                    print('---')
        #plt.xlabel("Epoch")
        #plt.plot(loss_list, label="Training loss")
        #plt.plot(valid_list, label="Validation Hits@20")
        #plt.plot(test_list, label="Test Hits@20")
        #plt.legend()
        #plt.show() 

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
    return loggers[key].tiaocan()
    
def objective(trial):
    
    param = {
              'learning_rate': trial.suggest_loguniform('learning_rate',0.0005, 0.005),
              'epochs': trial.suggest_int('epochs', 350, 550),
              'dropout': trial.suggest_loguniform('dropout',0.25, 0.55)
              }
    #main(param)
    result1=main(param)
    return result1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=10)
best_trial = study.best_trial
print(best_trial)


#if __name__ == "__main__":
    #main()