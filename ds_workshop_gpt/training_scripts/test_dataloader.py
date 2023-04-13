import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset as HFDataset
import pandas as pd
# try:
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,TextDataset,DataCollatorForLanguageModeling)

def get_eng_to_latex_dataset(tokenizer):
    '''
    '''

    # Add our singular prompt
    CONVERSION_PROMPT = 'LCT\n'  # LaTeX conversion task

    CONVERSION_TOKEN = 'LaTeX:'
    data = pd.read_csv('/run/determined/workdir/shared_fs/workshop_data/english_to_latex.csv')
    training_examples = f'{CONVERSION_PROMPT}English: ' + data['English'] + '\n' + CONVERSION_TOKEN + ' ' + data['LaTeX'].astype(str)
    task_df = pd.DataFrame({'text': training_examples})
    latex_data = HFDataset.from_pandas(task_df)  # turn a pandas DataFrame into a Dataset
    def preprocess(examples):  # tokenize our text but don't pad because our collator will pad for us dynamically
        return tokenizer(examples['text'], truncation=True)
    latex_data = latex_data.map(preprocess, batched=True,remove_columns='text')
    return latex_data

def get_datasets(dataset_name,tokenizer):
    '''
    '''
    if dataset_name=='english_to_latex':
        dataset = get_eng_to_latex_dataset(tokenizer)
    elif dataset_name=='data_science':
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path='/run/determined/workdir/shared_fs/workshop_data/PDS2.txt',  # Principles of Data Science - Sinan Ozdemir
            block_size=32  # length of each chunk of text to use as a datapoint
        )
    else:
        assert "Dataset Not Implemented"
    return dataset

def format_batch(batch):
    '''
    '''
    if dataset_name=='english_to_latex':
        inputs, outputs = (batch['input_ids'].to(device),batch['input_ids'].to(device))
    else:
        inputs, outputs = (batch['input_ids'].to(device),batch['input_ids'].to(device))
    return inputs, outputs

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dataset = get_datasets('english_to_latex',tokenizer)


data_c = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, collate_fn =data_c ,sampler=train_sampler, batch_size=2)

for b in train_dataloader:
    print(b)
    break