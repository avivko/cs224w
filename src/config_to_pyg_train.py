#!/usr/bin/python
from egnns import SimpleEGNN
from feat_processing import process_features
import pandas as pd
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
import graphein.protein as gp
from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphDataset
from graphein.ml.conversion import GraphFormatConvertor

# global params
CONFIG = sys.argv[1]
HOME =  '~/cs224w'
DATA = HOME + '/data/'
LIGHTNING_LOGS =  HOME + f'/lightning_logs/{CONFIG}_latest/'
TORCH_HOME = HOME + '/torch_home/'

def main():

    p_ds = os.path.join(os.path.dirname(__file__), '../structural_rearrangement_data.csv')
    df = pd.read_csv(p_ds)

    # get rid of bad pdbs (result in broken dssp features)

    train_bad_pdbs = ['1u6e', '3b7y', '1fdp', '2vj2', '1n2o', '1nxu', '1q18', '1pn2', '1vh3', '1xk7', '2q5r', '2q50',
                      '2olu', '1w2f', '1wxx', '2zc2', '3bwb', '2qt8', '2amj', '1h7s', '1zvw', '2f9t', '2f4b', '1jfv',
                      '1ujm', '2b4v', '2etv', '3c8v', '1tr9', '2imo', '3bgu', '2pyq', '1upl', '1x2g', '2gzr', '1od4',
                      '2qvl', '2i7h', '2dg0', '1evq', '2oyc', '2fyo', '1kko', '2a0k', '1uuj', '1u8t', '2bl7', '2c3v',
                      '1p5q', '2gvl', '2hvf', '2rad', '3c8n', '2z2n', '2a6c', '1b6w', '1u24', '2nyh', '1k75', '2r7f',
                      '2raf', '2qgq', '1ww8', '3c8h', '2eh1', '3db4', '2q4o', '3eo4', '2hzr', '2ghr', '1zs7', '1g5r',
                      '2r11', '1yqg', '3eua', '2dg0', '1evq', '2oyc', '2fyo', '2a0k', '1uj4', '1g7r', '1ynf', '1wpo',
                      '2qm8', '1vk3', '2cn2', '2vcl', '2rkt', '1l7o', '2r4i', '2it9', '1n4d', '2g6v', '1i3c', '1ox8',
                      '1fbt', '1n57', '2rem', '1xxl', '2v3j', '2g0a', '1q6h', '2q22'] \
                     + ['1lfh', '1gqz', '1yz5', '1oid', '1z15', '1l5t', '2cgk', '1sbq', '1vgt', '2ps3', '1bet', '1r5d',
                        '1eut', '1qmt', '1ogb', '2i78', '3b8s', '1uln', '1so7', '1kn9', '1pu5', '1rlr', '1sgz', '1k5h',
                        '1kx9'] \
                     + ['1tib', '1bsq', '1r1c', '1fo8', '1s0g', '1goh', '1m47', '1xyf', '1lyy', '2ofj', '1uzq', '2qfd',
                        '1br8', '1h8n', '1o7v', '1pz9', '1a8d', '1yjl', '2uyr', '2plf', '2rck', '1pbg', '1pgs'] \
                     + ['2i2w', '1beo', '1hxj', '1gy0', '2qzp', '2c7i', '2box', '1psr', '2yrf', '1mzl', '2fp8', '1arl',
                        '2yyv', '1iad', '2qev', '2veo', '1l5n', '1g24', '1w90', '1u4r', '1xze', '1z52', '1jwf', '2dcz',
                        '1slc'] \
                     + ['2d1z', '1oee', '4kiv', '2cbm', '1glh', '1qsa', '1bk7', '1qjv', '1e9n', '1vz2', '1czt', '3bi9',
                        '2e4p', '1o9z', '1l0x', '1qba', '2jcp', '1sgk', '2jh1', '2e1u', '1smn', '2fma', '2g7e', '2vl3',
                        '1g0z', '1uc8']

    valid_bad_pdbs = ['1vh3', '1x0v', '2b4v', '1tr9', '3bgu', '3bu9', '1uj4', '3c8n', '2a6c', '2r7f', '2hzb', '2qgq'] \
                     + ['2e2n', '2qrj', '1r0s', '1lr9', '1dcl', '3ezm', '3c3b', '2ddb', '2qpq', '1aja', '1pzt', '3seb',
                        '2p52', '2ze4', '1rn4', '2bce', '2fjy', '1m0z', '1fcq', '2f6l', '2ok3', '2bis', '1ppn', '1h8u',
                        '1l8f', '2uy2', '2egt', '1plu', '1dqg', '2j0a', '3cj1', '153l']

    test_bad_pdbs = ['3cze', '2q50', '2olu', '2qza', '2zc2', '2f9t', '2vrk', '2f6p', '1xe8', '2pgx', '1x2g', '2qvl',
                     '2i7h', '1evq', '1sza', '2q04', '1u8t', '2c3v', '2hvf', '1wgc', '1u24', '1k75', '1twd', '3c8h',
                     '2ghr', '1zs7', '1yqg'] \
                    + ['1zty', '1rki', '3bqh', '2orx', '2fz6', '1dkl', '2f08', '1ozt', '2rca', '1avk', '1vf8', '1bt2',
                       '1esc', '1wns', '2h74', '1nk1', '1xqv', '1lwb', '1pp3', '1knl', '2zg2', '2atb', '2d05', '1ogm',
                       '1kuf']

    all_bad = train_bad_pdbs + valid_bad_pdbs + test_bad_pdbs
    df = df.loc[~df['Free PDB'].isin(all_bad)]

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
    select_edge_func = {"edge_construction_functions": [gp.add_peptide_bonds,
                                                      gp.add_hydrogen_bond_interactions,
                                                      gp.add_disulfide_interactions,
                                                      gp.add_ionic_interactions,
                                                      gp.add_salt_bridges
                                                      ]}
    # 3:
    all_edge_func = {"edge_construction_functions": [gp.add_peptide_bonds,
                                                      gp.add_aromatic_interactions,
                                                      gp.add_hydrogen_bond_interactions,
                                                      gp.add_disulfide_interactions,
                                                      gp.add_ionic_interactions,
                                                      gp.add_aromatic_sulphur_interactions,
                                                      gp.add_cation_pi_interactions,
                                                      gp.add_hydrophobic_interactions,
                                                      gp.add_vdw_interactions,
                                                      gp.add_backbone_carbonyl_carbonyl_interactions,
                                                      gp.add_salt_bridges
                                                      ]}

    # A:
    one_hot = {"node_metadata_functions" : [gp.amino_acid_one_hot]}

    # B:
    #esm_func = get_esm_funcs(get_seq_esm=True, get_res_esm=False)

    # C:
    all_graph_metadata = {"graph_metadata_functions" : [gp.rsa #,
                                                        #gp.secondary_structure
                                                        ] }#+ esm_func['graph_metadata_functions']}
    all_node_metadata = {"node_metadata_functions" : [gp.amino_acid_one_hot,
                                                      gp.meiler_embedding,
                                                      partial(gp.expasy_protein_scale, add_separate=True)],
                         "dssp_config": gp.DSSPConfig()}

    # ['amino_acid_one_hot', 'esm', 'rsa', 'meiler', 'molecularweight', 'ss', 'bulkiness', 'seq_esm']


    config_1A = gp.ProteinGraphConfig(**{**dist_edge_func, **one_hot})
    #config_1B = gp.ProteinGraphConfig(**{**dist_edge_func, **esm_func})
    config_1C = gp.ProteinGraphConfig(**{**dist_edge_func, **all_graph_metadata, **all_node_metadata})

    config_2A = gp.ProteinGraphConfig(**{**select_edge_func, **one_hot})
    #config_2B = gp.ProteinGraphConfig(**{**select_edge_func, **esm_func})
    config_2C = gp.ProteinGraphConfig(**{**select_edge_func, **all_graph_metadata, **all_node_metadata})

    config_3A = gp.ProteinGraphConfig(**{**all_edge_func, **one_hot})
    #config_3B = gp.ProteinGraphConfig(**{**all_edge_func, **esm_func})
    config_3C = gp.ProteinGraphConfig(**{**all_edge_func, **all_graph_metadata, **all_node_metadata})

    configs_dict = {
        "config_1A": config_1A,
       # "config_1B": config_1B,
        "config_1C": config_1C,
        "config_2A": config_2A,
       # "config_2B": config_2B,
        "config_2C": config_2C,
        "config_3A": config_3A,
      #  "config_3B": config_3B,
        "config_3C": config_3C
    }


    # CHOOSE CONFIG FILE:
    curr_config = configs_dict[CONFIG]
    convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=["coords", "edge_index",
                                                                             "amino_acid_one_hot", "bulkiness",
                                                                             "meiler", "rsa",
                                                                             "esm_embedding_A", "esm_embedding",
                                                                             "pka_rgroup", "isoelectric_points",
                                                                             "polaritygrantham", "hphob_black",
                                                                             "transmembranetendency"
                                                                              ])


    train_ds = InMemoryProteinGraphDataset(
        root=DATA,
        name=f"{CONFIG}_train",
        pdb_codes=train,
        graph_label_map=train_label_map,
        graphein_config=curr_config,
        graph_format_convertor=convertor,
        graph_transformation_funcs=[],
        num_cores=16
        )

    valid_ds = InMemoryProteinGraphDataset(
        root=DATA,
        name=f"{CONFIG}_valid",
        pdb_codes=valid,
        graph_label_map=valid_label_map,
        graphein_config=curr_config,
        graph_format_convertor=convertor,
        graph_transformation_funcs=[],
        num_cores=16
        )

    test_ds = InMemoryProteinGraphDataset(
        root=DATA,
        graph_label_map=test_label_map,
        name=f"{CONFIG}_test",
        pdb_codes=test,
        graphein_config=curr_config,
        graph_format_convertor=convertor,
        graph_transformation_funcs=[],
        num_cores=16
        )


    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=15)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=15)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=15)


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
        max_epochs=500,
        log_every_n_steps=1,
        default_root_dir=LIGHTNING_LOGS
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

if __name__ == '__main__':
    main()