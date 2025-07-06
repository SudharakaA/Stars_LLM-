from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "star-gpt2-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Describe the star HIP 171."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128, do_sample=True, top_p=0.95)
print(tokenizer.decode(outputs[0]))