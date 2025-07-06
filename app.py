import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("star-gpt2-finetuned")
model = AutoModelForCausalLM.from_pretrained("star-gpt2-finetuned")

def starbot(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, 
        max_length=100, 
        do_sample=True, 
        top_p=0.9, 
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=starbot, inputs="text", outputs="text").launch()