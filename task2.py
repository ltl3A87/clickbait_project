# !pip install "peft==0.2.0"
# !pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade 
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import pandas as pd
import numpy as np
import os
import json
import collections

from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
from datasets import Dataset, DatasetDict

import datasets
from transformers.trainer_utils import set_seed
from transformers import (AutoTokenizer, PreTrainedTokenizerFast,
                          AutoModelForQuestionAnswering, TrainingArguments,
                          Trainer, default_data_collator, DataCollatorWithPadding)
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import DataCollatorForSeq2Seq
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk

from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import torch




def load_input(df):
    org_df = df
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    ret = []
    for _, i in df.iterrows():
        try:
            answer = " ".join(i['spoiler'])
            # print("answer: ", answer)

            
            ret += [{'context': i['targetTitle'] + ' - ' + ' '.join(i['targetParagraphs']), 'id': i['uuid'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']), 'answers': answer}]
        except:
            ret += [{'context': i['targetTitle'] + ' - ' + ' '.join(i['targetParagraphs']), 'id': i['postId'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']), 'answers': 'NA'}]

    return pd.DataFrame(ret)


model_id = "google/flan-t5-large"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

squad_dataset = load_dataset("squad_v2", split='train[:30%]')
# squad_dataset = load_dataset("squad_v2", split='train')

def squad_preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    
#     prefix_question = "Please extract the clickbait spoiler of this post: "
#     prefix_context = "according to the following context: "
    inputs = []
    for title, ques, cont in zip(examples["title"], examples["question"], examples["context"]):
        # print(prefix_question + doc + '\n' + prefix_context + cont)
        # print(cont)
        cont = title + ' - ' + cont
        inputs.append("question: %s  context: %s </s>" % (ques, cont))
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # The "labels" are the tokenized outputs:
    print(len(examples["answers"]))
    ans_list = []
    for ans in examples["answers"]:
        ans_list.append(' '.join(ans['text']))
    # ans = ' '.join(examples["answers"][0]['text'])
    labels = tokenizer(text_target=ans_list, 
                      max_length=512,         
                      truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_squad_dataset = squad_dataset
train_squad_dataset = train_squad_dataset.map(squad_preprocess_function, batched=True, remove_columns=["answers"])

input_path_train = '~/clickbait/train.jsonl'
input_path_val = '~/clickbait/validation.jsonl'
input_path_test = '~/clickbait/test.jsonl'
input_data_train = load_input(input_path_train)
input_data_val = load_input(input_path_val)
input_data_test = load_input(input_path_test)

dataset = DatasetDict({
    "train": Dataset.from_pandas(input_data_train),
    "val": Dataset.from_pandas(input_data_val),
    "test": Dataset.from_pandas(input_data_test),
    })
train_dataset = dataset['train']
test_dataset = dataset["test"]
val_dataset = dataset['val']

def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    
#     prefix_question = "Please extract the clickbait spoiler of this post: "
#     prefix_context = "according to the following context: "
    inputs = []
    for doc, cont in zip(examples["question"], examples["context"]):
        # print(prefix_question + doc + '\n' + prefix_context + cont)
        inputs.append("question: %s  context: %s </s>" % (doc, cont))
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=examples["answers"], 
                      max_length=512,         
                      truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["answers"])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["answers"])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["answers"])



tokenized_inputs = concatenate_datasets([train_squad_dataset, train_dataset])


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return result


output_dir="lora-flan-t5-large-retrain-30-ep10"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=10,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy='no',
    save_total_limit=3,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_inputs,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()

peft_model_id="results_t5_large_retrain_30_ep10"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

peft_model_id = "lora-flan-t5-large-retrain-30-ep10"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

pred_list = []
count = 0

for i in range(400):
    print(count)
    sample = test_dataset[i]
    print(sample["question"])
    input_ids = tokenizer("question: %s  context: %s </s>" % (sample["question"], sample["context"]), return_tensors="pt", truncation=True).input_ids.cuda(5)
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=True, top_p=0.9)

    input_s = "question: %s  context: %s </s>" % (sample["question"], sample["context"])
    # print(f"input sentence: {input_s}\n{'---'* 20}")

    ans = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    pred_list.append(ans)
    print(f"summary:\n{ans}")
    count += 1
    
import csv

id_list = [i for i in range(400)]

id_list = ['id'] + id_list 
ans_list = ['spoiler'] + pred_list
for i, k in enumerate(ans_list):
    if not k:
        ans_list[i] = "na"
with open('outputs_lora-flan-t5-large-retrain-30-ep10.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(id_list, ans_list))


