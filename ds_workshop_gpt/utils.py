from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, pipeline, \
                         Trainer, TrainingArguments
import torch
from collections import OrderedDict

def load_model_from_checkpoint(checkpoint):
    '''
    '''
    path = checkpoint.download()
    loaded_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # loaded_model = inference_model
    checkpoint = torch.load(path+'/state_dict.pth')
    # print(len(ckpt['models_state_dict']))
    # state_dict =ckpt['models_state_dict'][0]
    model_state_dict = checkpoint["models_state_dict"][0]
    # source: https://github.com/determined-ai/determined/blob/169729a82ecbd1d7fd8404e95f8033bf8475bcf0/harness/determined/pytorch/_pytorch_trial.py#L1101
    try:
        loaded_model.load_state_dict(model_state_dict)
    except Exception:
        # If the checkpointed model is non-DDP and the current model is DDP, append
        # module prefix to the checkpointed data
        if isinstance(loaded_model, torch.nn.parallel.DistributedDataParallel):
            print("Loading non-DDP checkpoint into a DDP model.")
            self._add_prefix_in_state_dict_if_not_present(model_state_dict, "module.")
        else:
            # If the checkpointed model is DDP and if we are currently running in
            # single-slot mode, remove the module prefix from checkpointed data
            print("Loading DDP checkpoint into a non-DDP model.")
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_state_dict, "module."
            )
        loaded_model.load_state_dict(model_state_dict)
    return loaded_model