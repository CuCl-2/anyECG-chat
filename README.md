# anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding

Official implementation of [anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding]

## Environment Set Up

```bash
conda create -n anyecgchat python=3.10
conda activate anyecgchat
pip install -r requirements.txt
```

## Data
+ The preprocessed QA data files are organized and stored in `data/stage1.json`, `data/stage2.json`, and `data/stage3.json`.
+ The ECG data still needs to be downloaded from the corresponding official website as mentioned in the main text.

## Pre-trained Models

We have released the **pre-trained ECG encoder trained on MIMIC** as well as the **Stage 3 checkpoint of anyECG-chat** on Hugging Face 🤗.

👉 Hugging Face repository:  
https://huggingface.co/cucl2/anyECG-chat


## Training anyECG-chat

You can either run `bash train.sh` to train all three stages at once or execute the following commands step by step:

### stage 1

```bash
python train.py --data_path ./data/mimic_report.json --unfreeze_ecg_model --batch_size 32 --accumulation_steps 8 --num_epochs 2 --output_dir /mnt/sda1/xxxx/output/anyECG/stage1
```

### stage 2
```bash
ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage1/step_6000"
python train.py --data_path ./data/stage2.json --unfreeze_ecg_model --use_lora --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --batch_size 8 --accumulation_steps 8 --num_epochs 2 --output_dir /mnt/sda1/xxxx/output/anyECG/debug
```
### stage 3
```bash
ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage2/step_28854"
python train.py --data_path ./data/stage3.json --unfreeze_ecg_model --use_lora --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --batch_size 8 --accumulation_steps 8 --num_epochs 1 --output_dir /mnt/sda1/xxxx/output/anyECG/stage3
```

## Testing

+ Run `bash inference.sh` to perform testing on all stages.
+ For the MIMIC Multi-ECG QA dataset, use the `QwQ` tool for additional evaluation:
    ```bash
    python mimic_multi_evaluate.py
    ```



