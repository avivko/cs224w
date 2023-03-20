"""EGNN Implementation from Satorras et al. https://github.com/vgsatorras/egnn"""
from feat_processing import process_features
import pytorch_lightning as pl
import torch
import torch.nn as nn
import itertools
import torchmetrics
from torch_geometric.data import Data
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import global_add_pool

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf)]
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i, j in itertools.product(range(n_nodes), range(n_nodes)):
        if i != j:
            rows.append(i)
            cols.append(j)

    return [rows, cols]


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


class SimpleEGNN(pl.LightningModule):
    def __init__(self,
                 n_feats=20,
                 hidden_dim=32,
                 out_feats=32,
                 edge_feats=0,
                 n_layers=2,
                 num_classes=6,
                 batch_size=16,
                 loss_fn=CrossEntropyLoss):
        super().__init__()
        self.num_classes = num_classes
        self.model = EGNN(
            in_node_nf=n_feats,
            out_node_nf=out_feats,
            in_edge_nf=edge_feats,
            hidden_nf=hidden_dim,
            n_layers=n_layers,
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, num_classes),
        )
        self.loss = loss_fn()
        self.batch_size = batch_size
        self.training_step_outputs = []
        self.training_step_labels = []
        self.valid_step_outputs = []
        self.valid_step_labels = []
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.micro_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro')
        self.macro_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro')

    def log_metrics(self, y, preds, mode='train'):
        '''
        y:      True data labels
        preds:  Predicted data labels
        mode:   Either 'train' or 'test'. For logging
        '''
        micro_f1 = self.micro_f1(preds, y)
        self.log(f'f1/{mode}/micro_f1', micro_f1)
        macro_f1 = self.macro_f1(preds, y)
        self.log(f'f1/{mode}/macro_f1', macro_f1)

        accuracy = self.accuracy(torch.argmax(preds, dim=1), y)
        self.log(f'accuracy/{mode}', accuracy)
        for label in torch.unique(y):
            y_sub = y[torch.where(y == label)]
            pred_sub = preds[torch.where(y == label)]
            class_acc = self.accuracy(torch.argmax(pred_sub, dim=1), y_sub)
            self.log(f'class_acc/{mode}/accuracy_{label}', class_acc)

    def configure_loss(self, name: str):
        """Return the loss function based on the config."""
        return self.loss

    # --- Forward pass
    def forward(self, x):
        '''
        x.aa = torch.cat([torch.tensor(a) for a in x.amino_acid_one_hot]).float().cuda()
        x.c = torch.cat([torch.tensor(a).squeeze(0) for a in x.coords]).float().cuda()
        feats, coords = self.model(
            h=x.aa,
            x=x.c,
            edges=x.edge_index,
            edge_attr=None,
        )
        '''
        for att in x.__dict__.keys():
            att_val = getattr(x, att)
            if type(att_val) == list:
                setattr(x, att, torch.tensor(*att_val).float().cuda())
        print('coords:', x.coords)
        print('edgeindex:', x.edge_index)
        # CALL PROCESSING FUNCTION THAT RETURNS H
        feats, coords = self.model(
            h=process_features(x, just_onehot=False),
            x=x.coords.float(),
            edges=x.edge_index,
            edge_attr=None,
        )

        feats = global_add_pool(feats, x.batch)
        return self.decoder(feats)

    def training_step(self, batch: Data, batch_idx) -> torch.Tensor:
        x = batch
        # y = batch.graph_y.unsqueeze(1).float()
        y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes))

        y_hat = self(x)
        # print(y_hat.dtype)
        _, y = y.max(dim=1)
        # print(y.shape)
        # print(y.dtype)
        # self.log_accuracy(y, y_hat, 'train')

        self.training_step_outputs.append(y_hat)
        self.training_step_labels.append(y)

        loss = self.loss(y_hat, y)
        # self.log('loss/train_loss', loss)

        return loss

    def on_train_epoch_end(self):
        all_preds = torch.concatenate(self.training_step_outputs, dim=0)
        all_labels = torch.concatenate(self.training_step_labels, dim=0)

        loss = self.loss(all_preds, all_labels)
        self.log('loss/train_loss', loss)
        self.log_metrics(all_labels, all_preds, 'train')

        self.training_step_outputs.clear()
        self.training_step_labels.clear()

    def validation_step(self, batch, batch_idx):
        x = batch
        # y = batch.graph_y.unsqueeze(1).float()
        y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes))

        y_hat = self(batch)
        self.validation_step_trainpend(y_hat)
        self.validation_step_labels.append(y)
        # self.log_metrics(y, y_hat, 'val')
        loss = self.loss(y_hat, y)
        # self.log('loss/val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.concatenate(self.valid_step_outputs, dim=0)
        all_labels = torch.concatenate(self.valid_step_labels, dim=0)

        loss = self.loss(all_preds, all_labels)
        self.log('loss/valid_loss', loss)
        self.log_metrics(all_labels, all_preds, 'valid')

        self.valid_step_outputs.clear()
        self.valid_step_labels.clear()

    def test_step(self, batch, batch_idx):
        x = batch
        # y = batch.graph_y.unsqueeze(1).float()
        y = batch.graph_y.reshape((int(x.graph_y.shape[0] / self.num_classes), self.num_classes))

        y_hat = self(x)
        _, y = y.max(dim=1)

        loss = self.loss(y_hat, y)

        y_pred_softmax = torch.log_softmax(y_hat, dim=1)
        y_pred_tags = torch.argmax(y_pred_softmax, dim=1)
        self.log("test_loss", loss, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.parameters(), lr=0.001)