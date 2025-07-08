import os
import random
import wfdb
import numpy as np
import torch
from scipy.signal import resample
import logging

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

def setup_logger(log_file='logfile.log'):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# Data
def get_ecg_from_path(path, sampling_freq, ecg_start=None, ecg_end=None):
    ecg, meta_info = wfdb.rdsamp(path)
    if ecg_start is not None and ecg_end is not None: # for location task
        ecg = ecg[ecg_start:ecg_end]
    else:
        ecg = ecg[:5000] # (5000, 12) mimic ptbxl 10s*500hz
    if sampling_freq != meta_info['fs']:
        ecg = resample(ecg, int(ecg.shape[0] * sampling_freq / meta_info['fs']), axis=0)

    leads_order_ref = [item.lower() for item in ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    leads_order = [item.lower() for item in meta_info['sig_name']]
    if meta_info['n_sig'] < 12: # for location task
        if leads_order == ['ecg', 'ecg']: # for mit_bih_st
            leads_order = ['mlii', 'v1']
        if leads_order == ['ecg']:
            leads_order = ['i']
        leads_order = [item.replace('ml', '').replace('d3', 'iii') for item in leads_order]
        for lead in leads_order_ref:
            if lead not in leads_order:
                leads_order.append(lead)
        ecg_zero = np.zeros((ecg.shape[0], 12-meta_info['n_sig']))
        ecg = np.concatenate([ecg, ecg_zero], axis=1)
    if leads_order != leads_order_ref:
        ecg = ecg[:, [leads_order.index(item) for item in leads_order_ref]]
    return ecg

def ecg_transform(ecg):
    '''
    transform ecg to [-1, 1]
    ecg: numpy array of shape (times * frequency, leads)
    '''
    min = np.min(ecg, axis=0, keepdims=True)
    max = np.max(ecg, axis=0, keepdims=True)
    ecg = (ecg - min) / (max - min) * 2 - 1
    ecg = np.nan_to_num(ecg, nan=0)
    return ecg
    
class Collate_Fn:
    def __init__(self, return_abnormal_type=False, return_question_type=False):
        self.return_abnormal_type = return_abnormal_type # for localization inference
        self.return_question_type = return_question_type # for ecgqa inference
    def __call__(self, batch):
        ecgs, messages = [], []
        if self.return_abnormal_type:
            abnormal_types = []
        if self.return_question_type:
            question_types = []
        for example in batch:
            ecg_path, dataset, task = example['ecg_path'], example['dataset'], example['task']
            if self.return_abnormal_type:
                abnormal_types.append(example['abnormal_type'])
            if self.return_question_type:
                question_types.append(example['question_type'])
            if task in ['location', 'location_long']:
                ecg_start, ecg_end = example['ecg_start'], example['ecg_end']
            else:
                ecg_start, ecg_end = None, None

            ecg = get_ecg_from_path(os.path.join(ecg_dir[dataset], ecg_path), 100, ecg_start, ecg_end)
            ecg = ecg_transform(ecg)
            ecg = torch.tensor(ecg).permute(1, 0)
            ecgs.append(ecg)
            messages.append([
                {"role": "user", "content": example['question']},
                {"role": "assistant", "content": example['answer']},
            ])
        if self.return_abnormal_type:
            return ecgs, messages, abnormal_types
        if self.return_question_type:
            return ecgs, messages, question_types
        else:
            return ecgs, messages

def collate_fn_infer(batch):
    ecgs, messages = [], []
    abnormal_types = [] # added
    for example in batch:
        abnormal_types.append(example['abnormal_type']) # added
        ecg_path, dataset, task = example['ecg_path'], example['dataset'], example['task']
        if task in ['location', 'location_long']:
            ecg_start, ecg_end = example['ecg_start'], example['ecg_end']
        else:
            ecg_start, ecg_end = None, None
        ecg = get_ecg_from_path(os.path.join(ecg_dir[dataset], ecg_path), 100, ecg_start, ecg_end)
        ecg = ecg_transform(ecg)
        ecg = torch.tensor(ecg).permute(1, 0)
        ecgs.append(ecg)
        messages.append([
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": example['answer']},
        ])
    return ecgs, messages, abnormal_types # added abnormal_types

# Training
def get_label_from_input_ids(input_ids):
    labels_all = []
    for i in range(input_ids.shape[0]):
        lst = input_ids[i].cpu().numpy().tolist()
        start_pattern = [128006, 78191, 128007, 271]
        end_marker = 128009
        segments = []
        n = len(lst)
        pattern_len = len(start_pattern)

        for i in range(n - pattern_len + 1):
            if lst[i:i+pattern_len] == start_pattern:
                try:
                    j = lst.index(end_marker, i + pattern_len)
                    segments.append((i+pattern_len, j+1))
                except ValueError:
                    pass
        labels = [-100] * n
        for start, end in segments:
            labels[start:end] = lst[start:end]
        labels = torch.tensor(labels).cuda().unsqueeze(0)
        labels_all.append(labels)
    labels_all = torch.cat(labels_all, dim=0)
    return labels_all

# Inference
def parse_intervals(interval_str):
    interval_str = interval_str.replace('Duration: ', '')
    intervals = []
    parts = interval_str.split(', ')
    for part in parts:
        start, end = part.split('-')
        start = float(start[:-1])
        end = float(end[:-1])
        intervals.append((start, end))
    return intervals

def compute_iou(interval_a, interval_b):
    intervals_a = parse_intervals(interval_a)
    intervals_b = parse_intervals(interval_b)

    total_intersection = 0
    for a in intervals_a:
        for b in intervals_b:
            total_intersection += max(0, min(a[1], b[1]) - max(a[0], b[0]))

    all_intervals = intervals_a + intervals_b
    all_intervals.sort()
    merged_intervals = []
    for current in all_intervals:
        if not merged_intervals or merged_intervals[-1][1] < current[0]:
            merged_intervals.append(current)
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], current[1]))

    total_union = sum(end - start for start, end in merged_intervals)
    return total_intersection / total_union if total_union > 0 else 0

_labels_ = ['non-diagnostic T abnormalities', 'non-specific ST changes', 'digitalis-effect', 'long QT-interval', 
            'normal ECG', 'inferior myocardial infarction', 'anteroseptal myocardial infarction', 'left ventricular hypertrophy', 
            'left anterior fascicular block', 'non-specific ischemic', 'incomplete right bundle branch block', 'first degree AV block', 
            'non-specific intraventricular conduction disturbance (block)', 'ischemic in anterolateral leads', 'complete right bundle branch block', 
            'complete left bundle branch block', 'inferolateral myocardial infarction', 'left atrial overload/enlargement', 'anterior myocardial infarction', 
            'anterolateral myocardial infarction', 'ischemic in inferior leads', 'subendocardial injury in anteroseptal leads', 'lateral myocardial infarction', 
            'ischemic in inferolateral leads', 'left posterior fascicular block', 'ischemic in anteroseptal leads', 'subendocardial injury in anterolateral leads',
            'ischemic in lateral leads', 'right ventricular hypertrophy', 'ST-T changes compatible with ventricular aneurysm', 'right atrial overload/enlargement', 
            'electrolytic disturbance or drug (former EDIS)', 'Wolf-Parkinson-White syndrome', 'incomplete left bundle branch block', 'inferoposterolateral myocardial infarction', 
            'ischemic in anterior leads', 'inferoposterior myocardial infarction', 'septal hypertrophy', 'subendocardial injury in inferior leads', 'subendocardial injury in lateral leads',
            'posterior myocardial infarction', 'third degree AV block', 'subendocardial injury in inferolateral leads', 'second degree AV block', 'abnormal QRS', 
            'ventricular premature complex', 'non-specific ST depression', 'voltage criteria (QRS) for left ventricular hypertrophy', 'Q waves present', 'low amplitude T-waves',
            'non-specific T-wave changes', 'atrial premature complex', 'prolonged PR interval', 'inverted T-waves', 'low QRS voltages in the frontal and horizontal leads', 
            'high QRS voltage', 'T-wave abnormality', 'non-specific ST elevation', 'premature complex(es)', 'sinus rhythm', 'atrial fibrillation', 'sinus tachycardia', 
            'sinus arrhythmia', 'sinus bradycardia', 'normal functioning artificial pacemaker', 'supraventricular arrhythmia', 'bigeminal pattern (unknown origin, SV or Ventricular)',
            'atrial flutter', 'supraventricular tachycardia', 'paroxysmal supraventricular tachycardia', 'trigeminal pattern (unknown origin, SV or Ventricular)']

_diag_labels_ = ['non-diagnostic T abnormalities', 'non-specific ST changes', 'digitalis-effect', 'long QT-interval', 
                 'normal ECG', 'inferior myocardial infarction', 'anteroseptal myocardial infarction', 'left ventricular hypertrophy', 
                 'left anterior fascicular block', 'non-specific ischemic', 'incomplete right bundle branch block', 'first degree AV block', 
                 'non-specific intraventricular conduction disturbance (block)', 'ischemic in anterolateral leads', 'complete right bundle branch block',
                 'complete left bundle branch block', 'inferolateral myocardial infarction', 'left atrial overload/enlargement', 'anterior myocardial infarction', 
                 'anterolateral myocardial infarction', 'ischemic in inferior leads', 'subendocardial injury in anteroseptal leads', 'lateral myocardial infarction', 
                 'ischemic in inferolateral leads', 'left posterior fascicular block', 'ischemic in anteroseptal leads', 'subendocardial injury in anterolateral leads', 
                 'ischemic in lateral leads', 'right ventricular hypertrophy', 'ST-T changes compatible with ventricular aneurysm', 'right atrial overload/enlargement', 
                 'electrolytic disturbance or drug (former EDIS)', 'Wolf-Parkinson-White syndrome', 'incomplete left bundle branch block', 'inferoposterolateral myocardial infarction', 
                 'ischemic in anterior leads', 'inferoposterior myocardial infarction', 'septal hypertrophy', 'subendocardial injury in inferior leads', 'subendocardial injury in lateral leads', 
                 'posterior myocardial infarction', 'third degree AV block', 'subendocardial injury in inferolateral leads', 'second degree AV block']

_form_labels_ = ['non-diagnostic T abnormalities', 'non-specific ST changes', 'digitalis-effect', 'long QT-interval', 'abnormal QRS', 'ventricular premature complex',
                 'non-specific ST depression', 'voltage criteria (QRS) for left ventricular hypertrophy', 'Q waves present', 'low amplitude T-waves', 
                 'non-specific T-wave changes', 'atrial premature complex', 'prolonged PR interval', 'inverted T-waves', 'low QRS voltages in the frontal and horizontal leads',
                 'high QRS voltage', 'T-wave abnormality', 'non-specific ST elevation', 'premature complex(es)']

_rhythm_labels_ = ['sinus rhythm', 'atrial fibrillation', 'sinus tachycardia', 'sinus arrhythmia', 'sinus bradycardia', 'normal functioning artificial pacemaker', 
                   'supraventricular arrhythmia', 'bigeminal pattern (unknown origin, SV or Ventricular)', 'atrial flutter', 'supraventricular tachycardia', 
                   'paroxysmal supraventricular tachycardia', 'trigeminal pattern (unknown origin, SV or Ventricular)']

_sub_diag_labels_ = ['ISCI', 'CRBBB', 'RAO/RAE', 'CLBBB', 'LVH', 'NST_', 'IRBBB', 'LAFB/LPFB', 'ISCA', 'WPW', 'RVH', 'NORM', 'IMI', 'AMI', 'ILBBB', 'STTC', 'PMI', 
                     '_AVB', 'LMI', 'SEHYP', 'IVCD', 'ISC_', 'LAO/LAE']

_super_diag_labels_ = ['CD', 'HYP', 'MI', 'STTC', 'NORM']

_sub_diag_text_ = ['ischemic in inferior leads', 'complete right bundle branch block', 'right atrial overload/enlargement', 'complete left bundle branch block',
                   'left ventricular hypertrophy', 'non-specific ST changes', 'incomplete right bundle branch block', 'left anterior/posterior fascicular block',
                   'ischemic in anterior leads', 'Wolf-Parkinson-White syndrome', 'right ventricular hypertrophy', 'normal ECG', 'inferior myocardial infarction',
                   'anterior myocardial infarction', 'incomplete left bundle branch block', 'ST/T change', 'posterior myocardial infarction', 'AV block',
                   'lateral myocardial infarction', 'septal hypertrophy', 'non-specific intraventricular conduction disturbance (block)', 'non-specific ischemic',
                   'left atrial overload/enlargement']

_super_diag_text_ = ['conduction disturbance', 'hypertrophy', 'myocardial infarction', 'ST/T change', 'normal ECG']

_csn_labels_ = ['Sinus Bradycardia', 'Sinus Rhythm', 'Atrial Flutter', 'Sinus Tachycardia', 'T wave Change', 'T wave opposite', 'Sinus Irregularity', 'Atrial Fibrillation', 'ST drop down', 'Axis left shift', 'atrial\xa0premature\xa0beats', 'ST-T Change', '1 degree atrioventricular block', 'abnormal Q wave', 'lower voltage QRS in all lead', 'Axis right shift', 'ST extension', 'Interior differences conduction', 'Supraventricular Tachycardia', 'right bundle branch block', 'left ventricle hypertrophy', 'QT interval extension', 'Early repolarization of the ventricles', 'Atrial Tachycardia', 'ventricular premature beat', 'atrioventricular block', 'left bundle branch block', 'ST tilt up', 'countercolockwise rotation', 'P wave Change', 'U wave', 'myocardial infarction', 'ventricular fusion wave', 'right ventricle hypertrophy', 'colockwise rotation', '3 degree atrioventricular block', 'junctional escape beat', 'WPW', '2 degree atrioventricular block', 'ventricular escape beat', 'PR interval extension', 'right atrial hypertrophy', '2 degree atrioventricular block(Type one)', 'Atrioventricular Reentrant Tachycardia', 'ventricular preexcitation', 'junctional premature beat', 'ventricular escape trigeminy', 'atrial bigeminy', 'fQRS Wave', 'ventricular bigeminy', 'Wandering in the atrioventricalualr node']

_cpsc_labels_ = ['Normal', 'Atrial fibrillation', 'First-degree atrioventricular block', 'Left bundle branch block', 'Right bundle branch block', 'Premature atrial contraction', 'Premature ventricular contraction', 'ST-segment depression', 'ST-segment elevated']

_ludb_labels_ = ['Sinus rhythm', 'Left ventricular hypertrophy', 'Left atrial hypertrophy', 'Electric axis of the heart: normal', 'Electric axis of the heart: left axis deviation', 'Incomplete right bundle branch block', 'Electric axis of the heart: vertical', 'Sinus bradycardia', 'Electric axis of the heart: horizontal', 'Non-specific repolarization abnormalities: inferior wall', 'Non-specific repolarization abnormalities: anterior wall', 'Right atrial overload', 'Left anterior hemiblock', 'Undefined ischemia/scar/supp.nstemi: lateral wall', 'Atrial fibrillation', 'Non-specific repolarization abnormalities: septal', 'Non-specific repolarization abnormalities: lateral wall', 'Undefined ischemia/scar/supp.nstemi: anterior wall', 'Left ventricular overload', 'Left atrial overload', 'Undefined ischemia/scar/supp.nstemi: apical', 'Non-specific repolarization abnormalities: apical', 'I degree av block', 'Ischemia: inferior wall', 'Pacemaker presence, undefined', 'Scar formation: septal', 'Non-specific repolarization abnormalities: posterior wall', 'Early repolarization syndrome', 'Sinus arrhythmia', 'Ischemia: lateral wall', 'Stemi: anterior wall', 'Stemi: septal', 'Scar formation: anterior wall', 'Stemi: lateral wall', 'Incomplete left bundle branch block', 'Ventricular extrasystole, type: single pvc', 'Unipolar ventricular pacing', 'Ischemia: apical', 'Scar formation: posterior wall', 'Iii degree av-block', 'Ischemia: anterior wall', 'Stemi: apical', 'Undefined ischemia/scar/supp.nstemi: septal', 'Scar formation: apical', 'Sinus tachycardia', 'Complete right bundle branch block', 'Complete left bundle branch block', 'Non-specific intravintricular conduction delay', 'Atrial extrasystole, type: single pac', 'Ischemia: septal', 'Undefined ischemia/scar/supp.nstemi: posterior wall', 'Atrial flutter, typical', 'Wandering atrial pacemaker', 'Electric axis of the heart: right axis deviation', 'Atrial extrasystole: sa-nodal extrasystole', 'Ventricular extrasystole, localisation: rvot, anterior wall', 'Right ventricular hypertrophy', 'Scar formation: inferior wall', 'Undefined ischemia/scar/supp.nstemi: inferior wall', 'Scar formation: lateral wall', 'Irregular sinus rhythm', 'Atrial extrasystole: undefined', 'Ventricular extrasystole, type: intercalary pvc', 'Atrial extrasystole: left atrial', 'Ventricular extrasystole, type: couplet', 'Ventricular extrasystole, morphology: polymorphic', 'Ventricular extrasystole, localisation: lvot, lvs', 'P-synchrony', 'Bipolar ventricular pacing', 'Ischemia: posterior wall', 'Aberrant conduction', 'Sinoatrial blockade, undefined', 'Atrial extrasystole: low atrial', 'Ventricular extrasystole, localisation: ivs, middle part', 'Ventricular extrasystole, localisation: rvot, antero-septal part', 'Atrial extrasystole, type: quadrigemini', 'Ventricular extrasystole, localisation: lv, undefined', 'Atrial extrasystole, type: allorhythmic pattern', 'Atrial extrasystole, type: bigemini', 'Right atrial hypertrophy', 'Unipolar atrial pacing', 'Biventricular pacing', 'Stemi: inferior wall']

_ptbxl_dir_ = '/mnt/sda1/xxxx/datasets/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
_cpsc_dir_ = '/mnt/sda1/xxxx/datasets/ECG/cpsc2018'
_csn_dir_ = '/mnt/sda1/xxxx/datasets/ECG/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
_mimic_dir_ = '/mnt/sda1/xxxx/datasets/ECG/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
_european_st_t_dir_ = '/mnt/sda1/xxxx/datasets/ECG/European_ST-T_Database/physionet.org/files/edb/1.0.0'
_mit_bih_st_dir_ = '/mnt/sda1/xxxx/datasets/ECG/MIT-BIH_ST_Change_Database/physionet.org/files/stdb/1.0.0'
_mit_bih_arrhythmia_dir_ = '/mnt/sda1/xxxx/datasets/ECG/MIT-BIH_Arrhythmia_Database/physionet.org/files/mitdb/1.0.0'

ecg_dir = {
    'ptbxl': _ptbxl_dir_,
    'cpsc': _cpsc_dir_,
    'csn': _csn_dir_,
    'mimic': _mimic_dir_,
    'european_st_t': _european_st_t_dir_,
    'mit_bih_st': _mit_bih_st_dir_,
    'mit_bih_arrhythmia': _mit_bih_arrhythmia_dir_
}
