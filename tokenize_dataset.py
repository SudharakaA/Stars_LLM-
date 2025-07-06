from datasets import load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
ds = load_from_disk("star-dataset")

def format_example(example):
    # combine prompt and completion for autoregressive models
    return {"text": example["prompt"] + " " + example["completion"]}

ds = ds.map(format_example)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_ds = ds.map(tokenize_function, batched=True)
tokenized_ds.save_to_disk("star-dataset-tokenized")