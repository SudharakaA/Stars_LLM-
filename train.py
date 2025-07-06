from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add padding token for training
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the tokenized dataset (it's not split, so access directly)
train_ds = load_from_disk("star-dataset-tokenized")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir="star-gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=False,   # set False if you get errors and no GPU
    eval_strategy="no",  # changed from evaluation_strategy
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("star-gpt2-finetuned")
tokenizer.save_pretrained("star-gpt2-finetuned")