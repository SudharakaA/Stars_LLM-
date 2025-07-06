# Star LLM Fine-tuning Project 🌟

A specialized language model fine-tuned on astronomical star data to generate detailed stellar descriptions.

## 🚀 Project Overview

This project demonstrates end-to-end fine-tuning of a GPT-2 model on a custom dataset of star descriptions. The model learns to generate technical astronomical content including star classifications, distances, magnitudes, and temperatures.

## 📊 Results

- **Training Loss**: Reduced from 2.9063 → 1.7965 (33% improvement)
- **Dataset Size**: 360 star descriptions
- **Training Time**: ~4 minutes (540 steps, 3 epochs)
- **Model**: GPT-2 (124M parameters) fine-tuned for astronomy domain

## 🛠️ Features

- **Data Pipeline**: JSONL to HuggingFace dataset conversion
- **Custom Tokenization**: Optimized for astronomical terminology
- **Web Interface**: Interactive Gradio app for model testing
- **CLI Tools**: Command-line scripts for batch generation
- **Model Optimization**: Memory and speed optimizations for consumer hardware

## 📁 Project Structure

```
star-llm-finetune/
├── train.jsonl                 # Training dataset (360 star descriptions)
├── convert_to_hf_dataset.py    # Data conversion script
├── tokenize_dataset.py         # Tokenization pipeline
├── train.py                    # Model training script
├── generate.py                 # CLI text generation
├── test_model.py              # Model testing and validation
├── app.py                     # Gradio web interface
├── monitor_training.py        # Training progress monitoring
└── requirements.txt           # Python dependencies
```

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SudharakaA/Stars_LLM-.git
   cd Stars_LLM-
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model

1. **Convert dataset**:
   ```bash
   python convert_to_hf_dataset.py
   ```

2. **Tokenize data**:
   ```bash
   python tokenize_dataset.py
   ```

3. **Train model**:
   ```bash
   python train.py
   ```

### Using the Trained Model

1. **Web Interface** (Recommended):
   ```bash
   python app.py
   ```
   Visit: http://127.0.0.1:7860

2. **Command Line Generation**:
   ```bash
   python generate.py
   ```

3. **Test Model Performance**:
   ```bash
   python test_model.py
   ```

## 📈 Technical Implementation

### Data Processing
- **Format**: JSONL with prompt-completion pairs
- **Tokenization**: GPT-2 tokenizer with custom padding
- **Preprocessing**: Combined prompts and completions for causal language modeling

### Model Configuration
- **Base Model**: GPT-2 (124M parameters)
- **Training**: 3 epochs, batch size 2, learning rate 5e-5
- **Optimization**: Memory-optimized for consumer hardware
- **Generation**: Nucleus sampling with repetition penalty

### Key Optimizations
- **Memory**: Reduced batch size, FP32 precision for Apple Silicon compatibility
- **Speed**: Batched tokenization, efficient data loading with caching
- **Quality**: Tuned generation parameters (temperature=0.7, repetition_penalty=1.2)

## 🎯 Example Outputs

**Input**: "Describe the star HIP 171."

**Output**: "HIP 171 (HD 1064) (GJ 6) is a G8V MAINSEQ star located at 22.04 parsecs with a visual magnitude of 4.68 and effective temperature..."

## 🔍 Challenges Solved

1. **Data Format Compatibility**: JSONL to HuggingFace dataset conversion
2. **API Compatibility**: Updated deprecated parameters in transformers library
3. **Memory Constraints**: Optimized for Apple Silicon MPS backend
4. **Generation Quality**: Eliminated repetitive outputs through parameter tuning
5. **Deployment**: Created multiple interfaces for different use cases

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Gradio
- See `requirements.txt` for complete list

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Dataset**: Custom astronomical star descriptions
- **Model**: Based on GPT-2 from Hugging Face
- **Framework**: Built with PyTorch and Transformers library

---

**Star LLM**: Bringing AI to astronomy, one star at a time! ⭐
