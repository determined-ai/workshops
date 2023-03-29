from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, pipeline, \
                         Trainer, TrainingArguments
import torch

def load_model_from_checkpoint(checkpoint):
    '''
    '''
    path = checkpoint.download()
    loaded_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # loaded_model = inference_model
    ckpt = torch.load(path+'/state_dict.pth')
    # print(len(ckpt['models_state_dict']))
    loaded_model.load_state_dict(ckpt['models_state_dict'][0])
    return loaded_model