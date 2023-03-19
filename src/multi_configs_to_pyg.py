#!/usr/bin/python
from egnns import SimpleEGNN
import pandas as pd
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from functools import partial
import torch
import pytorch_lightning as pl
import torch.nn as nn
import itertools
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
import graphein.protein as gp
from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphDataset
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )

# global params
HOME =  '~/cs224w'
DATA = HOME + '/data/'
TORCH_HOME = HOME + '/torch_home/'
CONFIG = sys.argv[1]


p_ds = os.path.join(os.path.dirname(__file__), '../structural_rearrangement_data.csv')
df = pd.read_csv(p_ds)

# let's get rid of  duplicates
df = df.loc[df["motion_type"] != "other_motion"]


# split data to train/valid/test
seed=1
np.random.seed(seed)
labels_onehot = pd.get_dummies(df.motion_type)
train_val, test, y_train_val, y_test = train_test_split(df["Free PDB"], labels_onehot, random_state=seed, stratify=labels_onehot, test_size=0.2)
train, valid = train_test_split(train_val, random_state=seed, stratify=y_train_val, test_size=0.2)
train = train.sort_index(); valid = valid.sort_index(); test = test.sort_index()

# make label one-hots and label map
def make_label_map(pdbs, onehots):
  label_map = {}
  for idx, pdb in enumerate(onehots):
    label_map[pdbs.iloc[idx]] = torch.tensor(onehots[idx])
  return label_map

train_labels_onehot = labels_onehot[labels_onehot.index.isin(train.index)].values.tolist()
train_label_map = make_label_map(train, train_labels_onehot)

valid_labels_onehot = labels_onehot[labels_onehot.index.isin(valid.index)].values.tolist()
valid_label_map = make_label_map(valid, valid_labels_onehot)

test_labels_onehot = labels_onehot[labels_onehot.index.isin(test.index)].values.tolist()
test_label_map = make_label_map(test, test_labels_onehot)

# CONFIGS
def get_esm_funcs(get_seq_esm=True, get_res_esm=True, esm_model="esm2_t33_650M_UR50D"):
  # this model is a bit heavy, will take you a moment the first time
  # Even deeper and much heavier, and colab will run out of memory
  #esm_model = "esm2_t36_3B_UR50D"
  #esm_model = "esm2_t48_15B_UR50D"

  os.environ['TORCH_HOME'] = TORCH_HOME
  torch.hub.load("facebookresearch/esm", esm_model)
  output_layer = int(esm_model.split('_t')[1].split('_')[0]) # e.g. get 33 for esm2_t33_650M_UR50D

  graph_metadata_functions = []

  if get_seq_esm:
    func_seq_esm = partial(gp.compute_esm_embedding, representation="sequence", model_name=esm_model, output_layer=output_layer)
    seq_esm = partial(gp.compute_feature_over_chains, func=func_seq_esm, feature_name="esm_embedding")
    graph_metadata_functions.append(seq_esm)

  if get_res_esm:
    res_esm = partial(gp.esm_residue_embedding, model_name=esm_model, output_layer=output_layer)
    graph_metadata_functions.append(res_esm)

  return {"graph_metadata_functions": graph_metadata_functions}

# 1:
# we chose 5A cutoff because of this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5460117/
dist_edge_func = {"edge_construction_functions": [partial(gp.add_distance_threshold, threshold=5, long_interaction_threshold=0)]}

# 2:
select_edge_func = {"edge_construction_functions": [add_peptide_bonds,
                                                  add_hydrogen_bond_interactions,
                                                  add_disulfide_interactions,
                                                  add_ionic_interactions,
                                                  gp.add_salt_bridges
                                                  ]}
# 3:
all_edge_func = {"edge_construction_functions": [add_peptide_bonds,
                                                  add_aromatic_interactions,
                                                  add_hydrogen_bond_interactions,
                                                  add_disulfide_interactions,
                                                  add_ionic_interactions,
                                                  add_aromatic_sulphur_interactions,
                                                  add_cation_pi_interactions,
                                                  gp.add_hydrophobic_interactions,
                                                  gp.add_vdw_interactions,
                                                  gp.add_backbone_carbonyl_carbonyl_interactions,
                                                  gp.add_salt_bridges
                                                  ]}

# A:
one_hot = {"node_metadata_functions" : [gp.amino_acid_one_hot]}

# B:
esm_func = get_esm_funcs(get_seq_esm=True, get_res_esm=True)

# C:
all_graph_metadata = {"graph_metadata_functions" : [gp.rsa,
                                                    gp.secondary_structure
                                                    ] + esm_func['graph_metadata_functions']}
all_node_metadata = {"node_metadata_functions" : [gp.amino_acid_one_hot,
                                                  gp.meiler_embedding,
                                                  partial(gp.expasy_protein_scale, add_separate=True)],
                     "dssp_config": gp.DSSPConfig()}

# ['amino_acid_one_hot', 'esm', 'rsa', 'meiler', 'molecularweight', 'ss', 'bulkiness', 'seq_esm']


config_1A = gp.ProteinGraphConfig(**{**dist_edge_func, **one_hot})
config_1B = gp.ProteinGraphConfig(**{**dist_edge_func, **esm_func})
config_1C = gp.ProteinGraphConfig(**{**dist_edge_func, **all_graph_metadata, **all_node_metadata})

config_2A = gp.ProteinGraphConfig(**{**select_edge_func, **one_hot})
config_2B = gp.ProteinGraphConfig(**{**select_edge_func, **esm_func})
config_2C = gp.ProteinGraphConfig(**{**select_edge_func, **all_graph_metadata, **all_node_metadata})

config_3A = gp.ProteinGraphConfig(**{**all_edge_func, **one_hot})
config_3B = gp.ProteinGraphConfig(**{**all_edge_func, **esm_func})
config_3C = gp.ProteinGraphConfig(**{**all_edge_func, **all_graph_metadata, **all_node_metadata})

configs_dict = {
    "config_1A": config_1A,
    "config_1B": config_1B,
    "config_1C": config_1C,
    "config_2A": config_2A,
    "config_2B": config_2B,
    "config_2C": config_2C,
    "config_3A": config_3A,
    "config_3B": config_3B,
    "config_3C": config_3C
}


# CHOOSE CONFIG FILE:
config = configs_dict[CONFIG]
convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=["coords", "edge_index",
                                                                             "amino_acid_one_hot", "bulkiness",
                                                                             "meiler", "molecularweight", "rsa",
                                                                             "esm_embedding_A", "esm_embedding",
                                                                             "ss"])


train_ds = InMemoryProteinGraphDataset(
    root=DATA,
    name="train",
    pdb_codes=train[:1],
    graph_label_map=train_label_map,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[],
    num_cores=1
    )

valid_ds = InMemoryProteinGraphDataset(
    root=DATA,
    name="valid",
    pdb_codes=valid,
    graph_label_map=valid_label_map,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[]
    )

test_ds = InMemoryProteinGraphDataset(
    root=DATA,
    graph_label_map=test_label_map,
    name="test",
    pdb_codes=test,
    graphein_config=config,
    graph_format_convertor=convertor,
    graph_transformation_funcs=[]
    )


# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=16, drop_last=True)


def process_features(batch, just_onehot: bool, also_concat_coords=False):
    if just_onehot:
        h = batch.amino_acid_one_hot.float()
        return h

    all_feats = ['amino_acid_one_hot', 'esm_embedding', 'rsa', 'meiler',
                 'molecularweight', 'ss', 'bulkiness']  # phi and psi?

    dssp_ss = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
    lb = LabelBinarizer().fit(dssp_ss)

    h_list = []
    for attrname in all_feats:
        try:
            attr = getattr(batch, attrname)
        except:
            # print(f'Missing attribute: {attrname}')
            continue

        if type(attr[0]) == pd.Series:  # converting pd.Series to numpy
            for i in range(len(attr)):
                attr[i] = attr[i].to_numpy()
            attr_tens = torch.concat(attr)
            h_list.append(attr_tens)

        if attrname == 'ss':  # onehotting dssp , in case it is a string array
            att_onehot = torch.FloatTensor(lb.transform(attr[0]))
            h_list.append(att_onehot)
        else:
            if len(attr.shape) == 1:
                attr = attr.unsqueeze(1)
            h_list.append(attr)

    h = torch.concatenate(h_list, dim=1)
    return h.to(torch.float32)


def calc_num_feats(loader):
  num_feats = 0
  for batch in loader:
    processed = process_features(batch, just_onehot=False)
    num_feats = processed.shape[1]
    break
  return num_feats


trainer = pl.Trainer(
    # strategy=None,
    accelerator='gpu',
    devices=1,
    benchmark=True,
    deterministic=False,
    num_sanity_val_steps=0,
    max_epochs=50,
    log_every_n_steps=1
)

model = SimpleEGNN(n_feats=calc_num_feats(train_loader),
                  hidden_dim=32,
                  out_feats=32,
                  edge_feats=0,
                  n_layers=2,
                  num_classes=6,
                  batch_size=16,
                  loss_fn = CrossEntropyLoss)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model, test_loader)