from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, pipeline, \
                         Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

pds_data = TextDataset(
    tokenizer=tokenizer,
    file_path='./oreilly-transformers-video-series/data/PDS2.txt',  # Principles of Data Science - Sinan Ozdemir
    block_size=32  # length of each chunk of text to use as a datapoint
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,  # MLM is Masked Language Modelling
)

tokenizer.pad_token = tokenizer.eos_token

collator_example = data_collator([tokenizer('I am an input'), tokenizer('So am I')])

model = GPT2LMHeadModel.from_pretrained('gpt2')  # load up a GPT2 model

pretrained_generator = pipeline(
    'text-generation', model=model, tokenizer='gpt2',
    config={'max_length': 200, 'do_sample': True, 'top_p': 0.9, 'temperature': 0.7, 'top_k': 10}
)
print("Warmup steps: len(pds_data.examples) // 5: ", len(pds_data.examples) // 5)
training_args = TrainingArguments(
    output_dir="./gpt2_pds", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=50, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=len(pds_data.examples) // 5, # number of warmup steps for learning rate scheduler,
    logging_steps=50,
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    dataloader_num_workers=0
)
print("Length of Training dataset: ", int(len(pds_data.examples)*.8))
print("Length of Validation dataset: ", int(len(pds_data.examples)*.8) -int(len(pds_data.examples)*.8))
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=pds_data.examples[:int(len(pds_data.examples)*.8)],
    eval_dataset=pds_data.examples[int(len(pds_data.examples)*.8):]
)

# trainer.evaluate()
trainer.train()
trainer.evaluate()
trainer.save_model()