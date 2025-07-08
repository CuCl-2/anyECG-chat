import os
import wfdb
import ast
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
import torch
from torch.utils.data import Dataset
from anyecg.utils import _labels_, _diag_labels_, _form_labels_, _rhythm_labels_, _sub_diag_labels_, _super_diag_labels_, _sub_diag_text_, _super_diag_text_
from anyecg.utils import _ludb_labels_, _csn_labels_, _cpsc_labels_
from anyecg.utils import _ptbxl_dir_, _csn_dir_, _cpsc_dir_

def get_ecg_from_path(path, sampling_freq):
    ecg, meta_info = wfdb.rdsamp(path)
    ecg = ecg[:5000] # (5000, 12)
    if sampling_freq != meta_info['fs']:
        ecg = resample(ecg, int(5000 * sampling_freq / meta_info['fs']), axis=0)

    leads_order_ref = [item.lower() for item in ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    leads_order = [item.lower() for item in meta_info['sig_name']]
    if leads_order != leads_order_ref:
        ecg = ecg[:, [leads_order.index(item) for item in leads_order_ref]]
    return ecg
            
class FinetuningDataset(Dataset):
    def __init__(self, dataset, dataset_subtype, ecg_transform, sampling_freq, split_fold, proportion = 1.0):
        self.dataset, self.dataset_subtype = dataset, dataset_subtype
        self.ecg_transform = ecg_transform
        self.sampling_freq = sampling_freq

        filename = f'/mnt/sda1/xxxx/datasets/ECG/clip_data/data/{dataset}.csv'
        df = pd.read_csv(filename)

        df = df[df['split_fold']==split_fold]
        # if split_fold == 'train' and proportion != 1.0:
        if proportion != 1.0:
            df = df.sample(frac=proportion)
        # get label columns name
        self._label_test_, self._text_test_, self._ecg_dir_ = self._get_label_text()

        # get label for sub and super diag
        if dataset == 'ptbxl' and dataset_subtype in ['sub-diag', 'super-diag']:
            label_column = 'sub_diag_labels' if dataset_subtype == 'sub-diag' else 'super_diag_labels'
            df[label_column] = df[label_column].apply(lambda x: ast.literal_eval(x))
            for label in self._label_test_:
                df[label] = df[label_column].apply(lambda x: 1 if label in x else 0)
        # for ptbxl subsets, only keep samples with at least one label
        if dataset == 'ptbxl': 
            df['label_len'] = df[self._label_test_].sum(axis=1)
            df = df[df['label_len']>0]
        self.df = df
            
    def _get_label_text(self):
        if self.dataset == 'ptbxl':
            _label_test_ = {'all': _labels_,
                            'diag': _diag_labels_,
                            'form': _form_labels_,
                            'rhythm': _rhythm_labels_,
                            'sub-diag': _sub_diag_labels_,
                            'super-diag': _super_diag_labels_}[self.dataset_subtype]
        elif self.dataset == 'ludb':
            _label_test_ = _ludb_labels_
        elif self.dataset == 'csn':
            _label_test_ = _csn_labels_
        elif self.dataset == 'cpsc':
            _label_test_ = _cpsc_labels_
        else:
            raise NotImplementedError
        
        if self.dataset == 'ptbxl' and self.dataset_subtype in ['sub-diag', 'super-diag']:
            _text_test_ = {
                'sub-diag': _sub_diag_text_,
                'super-diag': _super_diag_text_
            }[self.dataset_subtype]
        else:
            _text_test_ = _label_test_

        _ecg_dir_ = {
            'ptbxl': _ptbxl_dir_,
            'csn': _csn_dir_,
            'cpsc': _cpsc_dir_
        }[self.dataset]
        return _label_test_, _text_test_, _ecg_dir_
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        ecg_path = os.path.join(self._ecg_dir_, sample['path'])
        ecg = get_ecg_from_path(ecg_path, self.sampling_freq)
        ecg = self.ecg_transform(ecg)
        label = sample[self._label_test_].values
        return ecg, label
    
class FinetuningCollator:
    def __call__(self, batch):
        ecgs = [item[0] for item in batch]
        ecgs = np.array(ecgs, dtype=np.float32)
        ecgs = torch.as_tensor(ecgs).permute(0, 2, 1)
        labels = np.array([item[1] for item in batch], dtype=np.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return ecgs, labels