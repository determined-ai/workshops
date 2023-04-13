from __future__ import absolute_import, division, print_function

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
'''
Length of DS book: 601088 examples block 32
'''
dataset_name='english_to_latex'
if dataset_name == 'data_science':
    output_dir='data_science_ckpts/'
    per_gpu_train_batch_size=32*4
    per_gpu_eval_batch_size=32*4
    train_batch_size = 32*4
    eval_batch_size=32*4
    num_train_epochs=3
    logging_steps=50
    save_steps=50
elif dataset_name == 'english_to_latex':
    output_dir='english_to_latex_ckpts/'
    per_gpu_train_batch_size=2
    per_gpu_eval_batch_size=32
    train_batch_size = 32
    eval_batch_size=32
    num_train_epochs=1
    logging_steps=50
    save_steps=50
local_rank=-1
max_steps=-1
# per_gpu_train_batch_size=32
# per_gpu_eval_batch_size=32
learning_rate=5e-5
# num_train_epochs=1
# train_batch_size = 32
# eval_batch_size=32
gradient_accumulation_steps = 1
adam_epsilon=1e-8
warmup_steps=0
weight_decay=0.0
mlm=False
n_gpu=False
fp16=False
# logging_steps=50
# save_steps=50
evaluate_during_training=True
# output_dir='test/'
device='cuda'
max_grad_norm=1.0

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

def get_datasets(tokenizer):
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = output_dir

    # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    # eval_dataset = TextDataset(
    #     tokenizer=tokenizer,
    #     file_path='./data/PDS2.txt',  # Principles of Data Science - Sinan Ozdemir
    #     block_size=32  # length of each chunk of text to use as a datapoint
    # )
    eval_dataset = get_datasets(tokenizer)
    if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, None) if mlm else (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if local_rank in [-1, 0] else logging.WARN)
                        
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # dataset = TextDataset(
    #     tokenizer=tokenizer,
    #     file_path='./data/PDS2.txt',  # Principles of Data Science - Sinan Ozdemir
    #     block_size=32  # length of each chunk of text to use as a datapoint
    # )
    dataset = get_datasets(tokenizer)
    config = GPT2Config.from_pretrained('gpt2')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    # train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler = RandomSampler(dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(dataset, collate_fn =data_collator ,sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(dataset) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   train_batch_size * gradient_accumulation_steps * (torch.distributed.get_world_size() if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    tb_writer = SummaryWriter()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0])
    set_seed(0)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # inputs, labels = mask_tokens(batch, tokenizer, None) if mlm else (batch, batch)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            inputs, labels = format_batch(batch)
            model.train()
            
            outputs = model(inputs, masked_lm_labels=labels) if mlm else model(inputs, labels=labels)
            # print("outputs: ",outputs.keys())
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            
            if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                if local_rank == -1 and evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(model, tokenizer)
                    for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/logging_steps, global_step)
                    logging_loss = tr_loss

                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    train_output_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(train_output_dir):
                        os.makedirs(train_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(train_output_dir)
                    # torch.save(os.path.join(train_output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", train_output_dir)

                    # _rotate_checkpoints(args, checkpoint_prefix)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()