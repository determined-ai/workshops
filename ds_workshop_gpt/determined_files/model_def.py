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

from determined.pytorch import DataLoader, LRScheduler, PyTorchTrial, PyTorchTrialContext

class GPT2Finetune(PyTorchTrial):
    
    def __init__(self,context: PyTorchTrialContext) -> None:
        '''
        '''
        self.context = context
        # get hyperparametes
        self.weight_decay=self.context.get_hparam("weight_decay")
        self.learning_rate=self.context.get_hparam("learning_rate")
        self.adam_epsilon=self.context.get_hparam("adam_epsilon")
        self.warmup_steps=self.context.get_hparam("warmup_steps")
        self.epochs = self.context.get_hparam("epochs")
        self.gradient_accumulation_steps = self.context.get_hparam("gradient_accumulation_steps")
        # get tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # get dataset
        self.dataset_name=self.context.get_hparam("dataset_name")
        self.dataset = self.get_datasets(self.dataset_name)
        
        self.t_total = len(self.dataset) // self.gradient_accumulation_steps * self.epochs
    
        
        # get pretrained model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model = self.context.wrap_model(self.model)
        # get optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.context.get_hparam("weight_decay")},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        self.optimizer = self.context.wrap_optimizer(
            AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
            )
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # get learn rate scheduler
        self.scheduler = self.context.wrap_lr_scheduler(
            get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                            num_training_steps=self.t_total),
            LRScheduler.StepMode.MANUAL_STEP
        )
        

    def get_eng_to_latex_dataset(self):
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
            return self.tokenizer(examples['text'], truncation=True)
        latex_data = latex_data.map(preprocess, batched=True,remove_columns='text')
        return latex_data

    def get_datasets(self,dataset_name):
        '''
        '''
        if self.dataset_name=='english_to_latex':
            dataset = self.get_eng_to_latex_dataset()
        else:
            dataset_path = '/run/determined/workdir/shared_fs/workshop_data/{}.txt'.format(self.dataset_name)
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=dataset_path,  # Principles of Data Science - Sinan Ozdemir
                block_size=32  # length of each chunk of text to use as a datapoint
            )
        return dataset

    def format_batch(self,batch):
        '''
        '''
        inputs=batch['input_ids']
        outputs = batch['labels']
        return inputs, outputs
    def build_training_data_loader(self) -> None:
        '''
        '''
        self.train_sampler = RandomSampler(self.dataset)
        self.train_dataloader = DataLoader(self.dataset, collate_fn =self.data_collator ,sampler=self.train_sampler, batch_size=self.context.get_per_slot_batch_size())
        return self.train_dataloader
    
    def get_batch_length(self, batch):
        '''
        Count the number of records in a given batch.
        Override this method when you are using custom batch types, as produced
        when iterating over the `DataLoader`.
        '''
        return batch['input_ids'].shape[0]
    def build_validation_data_loader(self) -> None:
        '''
        '''
        self.eval_sampler = SequentialSampler(self.dataset)
        self.validataion_dataloader = DataLoader(self.dataset,collate_fn =self.data_collator, sampler=self.eval_sampler, batch_size=self.context.get_per_slot_batch_size())
        return self.validataion_dataloader
    
    def train_batch(self,batch,epoch_idx, batch_idx):
        '''
        '''
        inputs,labels = self.format_batch(batch)
        outputs = self.model(inputs, labels=labels)
        loss = outputs[0]
        train_result = {
            'loss': loss
        }
        self.context.backward(train_result["loss"])
        self.context.step_optimizer(self.optimizer)
        return train_result
    
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