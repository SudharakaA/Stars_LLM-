#!/usr/bin/env python3
"""
Simple test script to check if the model is working
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model():
    model_path = "star-gpt2-finetuned"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Training may not be complete yet.")
        return
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test prompts
        test_prompts = [
            "Describe the star HIP 171.",
            "Tell me about the star HD 142.",
            "What is a G3V star?",
        ]
        
        print("\nTesting model with different prompts:")
        print("=" * 50)
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=128, 
                    do_sample=True, 
                    top_p=0.95, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing model: {e}")
        print("The model may not be fully trained yet.")

if __name__ == "__main__":
    import torch
    test_model()
