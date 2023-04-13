from __future__ import absolute_import, division, print_function


from tqdm import tqdm

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

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

class GPT2Finetune:
    
    def __init__(self) -> None:
        '''
        '''
        
        # get hyperparametes
        self.weight_decay=0.0
        self.learning_rate=5e-5
        self.adam_epsilon=1e-8
        self.warmup_steps=0
        self.epochs = 1
        self.device = 'cuda'
        self.gradient_accumulation_steps = 1
        # get tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # get dataset
        self.dataset_name='english_to_latex'
        self.dataset = self.get_datasets(self.dataset_name)
        self.t_total = len(self.dataset) // self.gradient_accumulation_steps * self.epochs

        
        # get pretrained model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        # get optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # get learn rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)
        self.train_batch_size = 32
        self.eval_batch_size = 32

    def get_eng_to_latex_dataset(self):
        '''
        '''
        # Add our singular prompt
        CONVERSION_PROMPT = 'LCT\n'  # LaTeX conversion task

        CONVERSION_TOKEN = 'LaTeX:'
        data = pd.read_csv('./data/english_to_latex.csv')
        training_examples = f'{CONVERSION_PROMPT}English: ' + data['English'] + '\n' + CONVERSION_TOKEN + ' ' + data['LaTeX'].astype(str)
        task_df = pd.DataFrame({'text': training_examples})
        latex_data = HFDataset.from_pandas(task_df)  # turn a pandas DataFrame into a Dataset
        def preprocess(examples):  # tokenize our text but don't pad because our collator will pad for us dynamically
            return self.tokenizer(examples['text'], truncation=True)
        latex_data = latex_data.map(preprocess, batched=True,remove_columns='text')
        return latex_data

    def get_datasets(self,dataset_name):
        '''
        '''
        if self.dataset_name=='english_to_latex':
            dataset = self.get_eng_to_latex_dataset()
        elif self.dataset_name=='data_science':
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path='./data/PDS2.txt',  # Principles of Data Science - Sinan Ozdemir
                block_size=32  # length of each chunk of text to use as a datapoint
            )
        else:
            assert "Dataset Not Implemented"
        return dataset

    def format_batch(self,batch):
        '''
        '''
        if self.dataset_name=='english_to_latex':
            inputs, outputs = (batch['input_ids'].to(self.device),batch['labels'].to(self.device))
        else:
            # print(batch)
            inputs, outputs = (batch['input_ids'].to(self.device),batch['labels'].to(self.device))
        return inputs, outputs
    def build_train_data_loader(self) -> None:
        '''
        '''
        self.train_sampler = RandomSampler(self.dataset)
        self.train_dataloader = DataLoader(self.dataset, collate_fn =self.data_collator ,sampler=self.train_sampler, batch_size=self.train_batch_size)
    
    def build_validation_data_loader(self) -> None:
        '''
        '''
        self.eval_sampler = SequentialSampler(self.dataset)
        self.validataion_dataloader = DataLoader(self.dataset,collate_fn =self.data_collator, sampler=self.eval_sampler, batch_size=self.eval_batch_size)
    
    def train_batch(self,batch,epoch_idx, batch_idx):
        '''
        '''
        inputs,labels = self.format_batch(batch)
        outputs = self.model(inputs, labels=labels)
        loss = outputs[0]
        return loss
    
    def evaluate_batch(self,batch):
        '''
        '''
        inputs,labels = self.format_batch(batch)
        outputs = self.model(inputs, labels=labels)
        lm_loss = outputs[0]
        eval_loss = lm_loss.mean().item()
        perplexity = torch.exp(torch.tensor(eval_loss))

        results = {
            "eval_loss": eval_loss,
            "perplexity": perplexity
        }
        return results

if __name__ == '__main__':
    
    gpt2_finetune = GPT2Finetune()
    
    epochs = 1
    gpt2_finetune.model.train()
    gpt2_finetune.model.to(gpt2_finetune.device)
    gpt2_finetune.build_train_data_loader()
    gpt2_finetune.build_validation_data_loader()
    print("Training...")
    for e in range(epochs):
        for ind,batch in tqdm(enumerate(gpt2_finetune.train_dataloader)):

            gpt2_finetune.model.zero_grad()
            train_results  = gpt2_finetune.train_batch(batch,epoch_idx=e, batch_idx=ind)
            train_results.backward()
            gpt2_finetune.optimizer.step()
            gpt2_finetune.scheduler.step()
            print(train_results)
            if ind>20:
                break
    print("Validation...")
    gpt2_finetune.model.eval()
    with torch.no_grad():

        for val_batch in gpt2_finetune.validataion_dataloader:
            results = gpt2_finetune.evaluate_batch(val_batch)
            print("results:")
            print(results)
            break
        gpt2_finetune.model.train()