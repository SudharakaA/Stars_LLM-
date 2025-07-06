from datasets import load_dataset

# Load the local JSONL file
dataset = load_dataset("json", data_files="train.jsonl", split="train")
print(f"Dataset loaded with {len(dataset)} samples")
print("First sample:")
print(dataset[0])

# Save the dataset to disk
dataset.save_to_disk("star-dataset")
print("Dataset saved to star-dataset/")
