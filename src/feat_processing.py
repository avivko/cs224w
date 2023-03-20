import torch
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
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

        if type(attr) == list: # in case is list and not tensor
            attr = torch.tensor(*attr)

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

    h = torch.concat(h_list, dim=1)
    return h.float().cuda()