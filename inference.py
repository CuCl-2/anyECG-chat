import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import warnings
import torch
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import requests
import ecg_plot
from datasets import load_dataset
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from concurrent.futures import ThreadPoolExecutor
from anyecg.dataset import FinetuningDataset, FinetuningCollator
from anyecg.utils import Collate_Fn
from anyecg.utils import ecg_transform, compute_iou
from anyecg.ecg_language_modeling import ECG_Language_Model
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--dataset', type=str, default='csn', choices=['ptbxl', 'cpsc', 'csn', 'european_st_t', 'mit_bih_st', 'mit_bih_arrhythmia', 
                                                                   'european_st_t_long', 'mit_bih_st_long', 'mit_bih_arrhythmia_long', 'ecgqa', 'mimic-multi'], help='Dataset')
parser.add_argument('--dataset_subtype', type=str, choices=['all', 'diag', 'form', 'rhythm', 'sub-diag', 'super-diag'], default='all')
parser.add_argument('--sampling_freq', type=int, default=100, help='Sampling frequency')
parser.add_argument('--mask_first_non_zero_lead', action='store_true', help='Mask the first non-zero lead') # for location task
parser.add_argument('--mask_second_non_zero_lead', action='store_true', help='Mask the second non-zero lead')
parser.add_argument('--mask_random_non_zero_lead', action='store_true', help='Mask a random non-zero lead')
# Model
parser.add_argument('--model_name', type=str, default='anyECG-chat', help='Model name', choices=['anyECG-chat', 'pulse', 'llava-med'])
parser.add_argument('--projection_ckpt', type=str, default=None)
parser.add_argument('--ecg_model_ckpt', type=str, default=None)
parser.add_argument('--lora_ckpt', type=str, default=None)
# Test
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature')
parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--text_embedding_model', type=str, default='/mnt/sda1/xxxx/huggingface/BioBERT-mnli-snli-scinli-scitail-mednli-stsb', help='Text embedding model')
parser.add_argument('--result_path', type=str, default='/mnt/sda1/xxxx/output/anyECG/debug.json')
args = parser.parse_args()

# MODEL
if args.model_name == 'anyECG-chat':
    ecg_language_model = ECG_Language_Model()
    # load ckpt
    assert args.projection_ckpt is not None, 'Please provide the projection checkpoint.'
    res = ecg_language_model.projection.load_state_dict(torch.load(args.projection_ckpt))
    print(f'Load projection: {res}')
    if args.ecg_model_ckpt is not None:
        res = ecg_language_model.ecg_model.load_state_dict(torch.load(args.ecg_model_ckpt))
        print(f'Load ecg model: {res}')
    if args.lora_ckpt is not None:
        ecg_language_model.language_model.load_adapter(args.lora_ckpt)
        print(f'Load lora: {args.lora_ckpt}')

    ecg_language_model = ecg_language_model.cuda()
    ecg_language_model.eval()
    def get_response(ecgs, messages):
        response = ecg_language_model.ecg_chat(ecgs, messages, args.temperature)
        return response
elif args.model_name == 'pulse':
    def get_response(ecgs, messages):
        def process_ecg(i):
            ecg = ecgs[i].cpu().numpy()
            message = messages[i][0]['content']
            ecg_dir = '/mnt/sda1/xxxx/output/anyECG/other_baseline/pulse/images/'
            ecg_name = f"ecg_example_{args.model_name}_{args.dataset}_{args.dataset_subtype}_{i}"
            ecg_plot.plot(ecg, sample_rate=args.sampling_freq)
            ecg_plot.save_as_png(ecg_name, ecg_dir)
            url = "http://localhost:8000/predict"
            headers = {"Content-Type": "application/json"}
            data = {
                "image_file": os.path.join(ecg_dir, ecg_name + ".png"),
                "query": message
            }
            return requests.post(url, headers=headers, json=data).json()['response']

        # with ThreadPoolExecutor() as executor:
        #     response = list(executor.map(process_ecg, range(len(ecgs))))

        response = []
        from tqdm import trange
        for i in trange(len(ecgs)):
            response.append(process_ecg(i))
        return response
elif args.model_name == 'llava-med':
    def get_response(ecgs, messages):
        def process_ecg(i):
            ecg = ecgs[i].cpu().numpy()
            message = messages[i][0]['content']
            ecg_dir = '/mnt/sda1/xxxx/output/anyECG/other_baseline/llava-med/images/'
            ecg_name = f"ecg_example_{args.model_name}_{args.dataset}_{args.dataset_subtype}_{i}"
            ecg_plot.plot(ecg, sample_rate=args.sampling_freq)
            ecg_plot.save_as_png(ecg_name, ecg_dir)
            url = "http://localhost:5000/ask"
            headers = {"Content-Type": "application/json"}
            data = {
                "image_path": os.path.join(ecg_dir, ecg_name + ".png"),
                "question": 'you are an expert in ECG, you can interpret the ECG.' + message
            }
            return requests.post(url, headers=headers, json=data).json()['response']

        # with ThreadPoolExecutor() as executor:
        #     response = list(executor.map(process_ecg, range(len(ecgs))))

        response = []
        from tqdm import trange
        for i in trange(len(ecgs)):
            response.append(process_ecg(i))
        return response    


def test_classification():
    # Data
    proportion = 0.1 if args.model_name != 'anyECG-chat' else 1.0
    test_data = FinetuningDataset(args.dataset, args.dataset_subtype, ecg_transform=ecg_transform, sampling_freq=args.sampling_freq, split_fold = 'test', proportion=proportion)
    collate_fn = FinetuningCollator()
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True, num_workers=32)

    # TEST
    text_model = SentenceTransformer(args.text_embedding_model)
    label_embeddings = text_model.encode(test_data._text_test_)
    labels_all, prediction_all = [], []
    reports_all, reports_prediction_all = [], [] # for BLEU, ROUGE
    for ecgs, labels in tqdm(test_loader):
        message = [[{'role': 'user', 'content': "Please provide the report for the following ECG."}] for _ in range(ecgs.shape[0])]
        response = get_response(ecgs, message)
        # for BLEU, ROUGE
        reports = ['Report: ' + ', '.join([test_data._text_test_[i] for i, val in enumerate(label) if val == 1]) for label in labels]
        reports_all.extend(reports)
        reports_prediction_all.extend(response)

        prediction_labels = [item.replace('Report: ', '').split(', ') for item in response]
        # prediction_labels = [[item] for item in response] # whole report for embeddings
        prediction_labels_embeddings = [text_model.encode(item) for item in prediction_labels]
        similarity = [cosine_similarity(label_embeddings, item) for item in prediction_labels_embeddings]
        prediction = [item.max(axis=1) for item in similarity]
        prediction_all.extend(prediction)
        labels_all.extend(labels)
    if args.model_name != 'anyECG-chat':
        labels_all.append([1] * len(test_data._text_test_))
        prediction_all.append([0.5] * len(test_data._text_test_))
    labels_all = np.array(labels_all)
    prediction_all = np.array(prediction_all)

    auc_all = roc_auc_score(labels_all, prediction_all, average=None)
    
    # Calculate BLEU
    bleu_1 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 0, 0, 1))
    # Calculate ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(reports_prediction_all, reports_all, avg=True)

    output_json = {
        'auc': auc_all.mean(),
        'detail': {test_data._text_test_[i]: auc_all[i] for i in range(len(auc_all))},
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'rouge_1': rouge_scores['rouge-1']['f'],
        'rouge_2': rouge_scores['rouge-2']['f'],
        'rouge_l': rouge_scores['rouge-l']['f'],
    }
    print(output_json)
    with open(args.result_path, 'w') as f:
        json.dump(output_json, f, indent=4)

def test_localization():
    # Data
    dataset = load_dataset('json', data_files=f'./data/location/{args.dataset}_test.json')['train']
    test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=Collate_Fn(return_abnormal_type=True), num_workers=64)
    # TEST
    mean_iou = {}
    for ecgs, messages, abnormal_types in tqdm(test_dataloader):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        # mask leads
        for ecg in ecgs:
            non_zero_leads = [i for i in range(ecg.shape[0]) if torch.sum(ecg[i]) != 0]
            # skip if only one lead
            if len(non_zero_leads) == 1:
                continue
            if args.mask_first_non_zero_lead:
                ecg[non_zero_leads[0]] = 0
            if args.mask_second_non_zero_lead:
                ecg[non_zero_leads[1]] = 0
            if args.mask_random_non_zero_lead:
                ecg[np.random.choice(non_zero_leads)] = 0

        response = get_response(ecgs, question_message)

        for i in range(len(response)):
            truth = answer_message[i][0]['content']
            prediction = response[i]
            abnormal_type = abnormal_types[i]
            if truth == prediction: # both no found
                iou = 1.0
            else:
                try:
                    iou = compute_iou(truth, prediction)
                except:
                    iou = 0
            if abnormal_type not in mean_iou:
                mean_iou[abnormal_type] = {'mean_iou': 0, 'count': 0}
            mean_iou[abnormal_type]['mean_iou'] += iou
            mean_iou[abnormal_type]['count'] += 1
    for key in mean_iou:
        mean_iou[key]['mean_iou'] /= mean_iou[key]['count']
    macro_iou = np.mean([mean_iou[key]['mean_iou'] for key in mean_iou])
    micro_iou = np.sum([mean_iou[key]['mean_iou'] * mean_iou[key]['count'] for key in mean_iou]) / np.sum([mean_iou[key]['count'] for key in mean_iou])
    output_json = {
        'macro_iou': macro_iou,
        'micro_iou': micro_iou,
        'detail': mean_iou
    }
    print(output_json)
    with open(args.result_path, 'w') as f:
        json.dump(output_json, f, indent=4)

def test_ecgqa():
    # Data
    dataset = load_dataset('json', data_files=f'./data/{args.dataset}_test.json')['train']
    if args.model_name != 'anyECG-chat':
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.1)))
        text_model = SentenceTransformer(args.text_embedding_model)
    test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=Collate_Fn(return_question_type=True), num_workers=64)
    # TEST
    mean_acc ={}
    for ecgs, messages, question_types in tqdm(test_dataloader):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        response = get_response(ecgs, question_message)
        for i in range(len(response)):
            truth = answer_message[i][0]['content']
            prediction = response[i]
            question_type = question_types[i]
            if 'verify' in question_type and args.model_name != 'anyECG-chat':
                label_embeddings = text_model.encode(['yes', 'no', 'not sure'])
                prediction_embeddings = text_model.encode([prediction])
                similarity = cosine_similarity(label_embeddings, prediction_embeddings)
                prediction = ['yes', 'no', 'not sure'][similarity.argmax()]
            acc = 1 if truth.lower() == prediction.lower() else 0
            if question_type not in mean_acc:
                mean_acc[question_type] = {'mean_acc': 0, 'count': 0}
            mean_acc[question_type]['mean_acc'] += acc
            mean_acc[question_type]['count'] += 1
    for key in mean_acc:
        mean_acc[key]['mean_acc'] /= mean_acc[key]['count']
    macro_acc = np.mean([mean_acc[key]['mean_acc'] for key in mean_acc])
    micro_acc = np.sum([mean_acc[key]['mean_acc'] * mean_acc[key]['count'] for key in mean_acc]) / np.sum([mean_acc[key]['count'] for key in mean_acc])
    output_json = {
        'macro_acc': macro_acc,
        'micro_acc': micro_acc,
        'detail': mean_acc
    }
    print(output_json)
    with open(args.result_path, 'w') as f:
        json.dump(output_json, f, indent=4)

def test_mimic_multi():
    # Data
    dataset = load_dataset('json', data_files=f'./data/mimic_llama3.3-70b_test.json')['train']
    test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=Collate_Fn(), num_workers=64)
    # TEST
    count = 0
    samples_all = []
    for ecgs, messages in tqdm(test_dataloader):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        response = get_response(ecgs, question_message)
        for i in range(len(response)):
            sample = {
                'id': count,
                'question': question_message[i][0]['content'].replace('<|reserved_special_token_1|>', '').replace('<|reserved_special_token_2|>', '').replace('<|reserved_special_token_3|>', ''),
                'answer': answer_message[i][0]['content'],
                'prediction': response[i],
            }
            count += 1
            samples_all.append(sample)
    output_json = {
        'samples': samples_all
    }
    with open(args.result_path, 'w') as f:
        json.dump(output_json, f, indent=4)

if args.dataset in ['ptbxl', 'cpsc', 'csn']:
    test_classification()
elif args.dataset == 'ecgqa':
    test_ecgqa()
elif args.dataset == 'mimic-multi':
    test_mimic_multi()
else:
    test_localization()

