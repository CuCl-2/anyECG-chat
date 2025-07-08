import json
import pandas as pd
import argparse
answer_anyECGchat = json.load(open('/mnt/sda1/xxxx/output/anyECG/stage3/step_17057/mimic-multi_BioBERT-mnli-snli-scinli-scitail-mednli-stsb_tem0.6.json', 'r', encoding='utf-8'))['samples']
answer_llava_med = json.load(open('/mnt/sda1/xxxx/output/anyECG/other_baseline/llava-med/mimic-multi_BioBERT-mnli-snli-scinli-scitail-mednli-stsb_tem0.6.json', 'r', encoding='utf-8'))['samples']
answer_pulse = json.load(open('/mnt/sda1/xxxx/output/anyECG/other_baseline/pulse/mimic-multi_BioBERT-mnli-snli-scinli-scitail-mednli-stsb_tem0.6.json', 'r', encoding='utf-8'))['samples']

data = json.load(open('/home/xxxx/ANY_ECG/anyECG/data/mimic_llama3.3-70b_test.json', 'r', encoding='utf-8'))

df = pd.read_csv('/mnt/sda1/xxxx/datasets/ECG/clip_data/data/mimic.csv')

# Config
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default='anyECGchat', help='model name', choices=['anyECGchat', 'llava-med', 'pulse'])
args = parser.parse_args()
answer_evaluate = {
    'anyECGchat': answer_anyECGchat,
    'llava-med': answer_llava_med,
    'pulse': answer_pulse
}[args.model]
output_path = f'./data/evaluation/{args.model}_evaluate.json'

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self, model_id: str, device: str = "cuda:0", use_vllm=False, temperature=0.6):
        self.model_id = model_id
        self.use_vllm = use_vllm
        self.temperature = temperature
        if use_vllm:
            openai_api_key = "token-abc123"
            openai_api_base = "http://localhost:8002/v1"

            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.device = device

    def chat(self, messages: list) -> str:
        if self.use_vllm:
            chat_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature
            )
            response = chat_response.choices[0].message.content
            return response
        else:
            # Prepare the input data for the model
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Define the possible terminators (end of conversation tokens)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate the model's response
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
            )

            # Get the generated response and decode it
            response = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)
        
    def one_turn_chat(self, user_input: str) -> str:
        messages = [
            {"role": "user", "content": user_input}
        ]
        return self.chat(messages)


chatbot = Chatbot(model_id = '/mnt/sda1/xxxx/huggingface/QwQ-32B', use_vllm=True, temperature=0.3)
print(chatbot.one_turn_chat('who are you'))

import concurrent.futures

def process_question(i):
    question = data[i]['question']
    ecg_path = data[i]['ecg_path']
    labels = [df[df['path'] == item]['report'].values[0] for item in ecg_path]
    prediction = answer_evaluate[i]['prediction']
    id = answer_evaluate[i]['id']
    prompt = f'For the given question <{question}> about multiple ECG-QA, and the report {labels} corresponding to each ECG, score the answer below, where 0 means completely incorrect and 5 means completely correct. The answer is: <{prediction}>.'
    messages = [
        {"role": "user", "content": prompt}
    ]
    mid_response = chatbot.chat(messages)
    messages.append({"role": "assistant", "content": mid_response})
    messages.append({"role": "user", "content": "So what is the final score? Just give me a number between 0 and 5."})
    response = chatbot.chat(messages)
    return {
        'id': id,
        'mid_response': mid_response,
        'response': response,
    }

with concurrent.futures.ThreadPoolExecutor() as executor:
    response_all = list(executor.map(process_question, range(len(answer_evaluate))))

with open(output_path, 'w') as f:
    json.dump(response_all, f, indent=4)

    
