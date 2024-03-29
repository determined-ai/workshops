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



from datasets import Dataset, load_dataset
import peft
print("peft.__version__: ",peft.__version__)
 
import transformers

from transformers import (AdamW,
                          AutoTokenizer,
                          HfArgumentParser,
                          TrainingArguments,
                          AutoModelForCausalLM,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          BitsAndBytesConfig,
                          TextStreamer,
                          get_scheduler)
print("transformers.__version__: ",transformers.__version__)
import datasets
print("datasets.__version__: ",datasets.__version__)
from peft import LoraConfig, get_peft_config, get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from determined.pytorch import DataLoader, LRScheduler, PyTorchTrial, PyTorchTrialContext

class OPTFinetuneTrial(PyTorchTrial):
    
    def __init__(self,context: PyTorchTrialContext) -> None:
        '''
        '''
        self.context = context
        self.using_wikitext = self.context.get_hparam("use_hface")

        # get hyperparametes
        self.weight_decay=self.context.get_hparam("weight_decay")
        self.learning_rate=self.context.get_hparam("learning_rate")
        self.adam_epsilon=self.context.get_hparam("adam_epsilon")
        self.warmup_steps=self.context.get_hparam("warmup_steps")
        self.epochs = self.context.get_hparam("epochs")
        self.gradient_accumulation_steps = self.context.get_hparam("gradient_accumulation_steps")
        # get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # get dataset
        self.dataset_name=self.context.get_hparam("dataset_name")
        self.dataset = self.get_datasets(self.dataset_name)
        
        self.t_total = len(self.dataset) // self.gradient_accumulation_steps * self.epochs
    
        model_cache_dir = '/mnt/efs/shared_fs/determined/hf_cache/mistral_instruct_model_cache/'
        tokenizer_cache_dir = '/mnt/efs/shared_fs/determined/hf_cache/mistral_instruct_tokenizer_cache/'
        dataset_cache_dir = '/mnt/efs/shared_fs/determined/hf_cache/dataset_tokenizer_cache/'
        dataset_json_file = '/mnt/efs/shared_fs/determined/hf_cache/dataset_download_json_dir.json'
        finetune_results_dir = '/mnt/efs/shared_fs/determined/hf_cache/mistral_ft_results/'
        # get pretrained model
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m",
                                              load_in_4bit=False,
                                              # torch_dtype=torch.bfloat16,
                                              device_map={"": 0},
                                              trust_remote_code=True,
                                              # quantization_config=bnb_config,
                                              cache_dir=model_cache_dir,
                                              local_files_only=False)
        self.model = self.context.wrap_model(self.model)
        # get optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        learning_rate = 0.000025*2
        per_device_train_batch_size= 1
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        # optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        # These are the only changes you need to make
        # The first part sets the optimizer to use 8-bits
        # The for loop sets embeddings to use 32-bits

        self.optimizer =  self.context.wrap_optimizer(bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=learning_rate))

        # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
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
    
    def get_hface_dataset(self):
        self.using_wikitext = True
        dataset_cache_dir = '/mnt/efs/shared_fs/determined/hf_cache/dataset_tokenizer_cache/'
        dataset = load_dataset(self.context.get_hparam("dataset_name"), 
                               self.context.get_hparam("dataset_config"), 
                               split="train",
                               cache_dir=dataset_cache_dir)

        BATCHES=100
        dataset2 = dataset.select(indices=list(range(8*BATCHES)))
        def preprocess(samples):
            samples = self.tokenizer(samples['text'],truncation=True)# (11.6.2023) should change
            return samples
        self.mapped_dataset = dataset2.map(preprocess,batched=True ,remove_columns=['text'])
        return self.mapped_dataset

    def get_datasets(self,dataset_name):
        '''
        '''
        if self.dataset_name=='english_to_latex':
            dataset = self.get_eng_to_latex_dataset()
        else:
            if self.context.get_hparam("use_hface"):
                dataset = self.get_hface_dataset()
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
        if self.using_wikitext:
            return (batch['input_ids'] ,batch['input_ids'])
        else:
            inputs=batch['input_ids']
            outputs = batch['labels']
            return inputs, outputs
        
    def build_training_data_loader(self) -> None:
        '''
        '''
        if self.using_wikitext:
            return DataLoader(self.mapped_dataset, collate_fn =self.data_collator ,shuffle=True, batch_size=4)
        else:
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