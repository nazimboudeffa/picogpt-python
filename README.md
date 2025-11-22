# PicoGPT Python

A lightweight GPT-style language model implementation in PyTorch for learning and experimentation with transformer architectures. This project trains a small-scale GPT model on simple English phrases to understand language generation fundamentals.

## 🎯 Overview

This project implements a miniature GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. It's designed for educational purposes and experimentation with transformer-based language models.

## ✨ Features

- **Custom GPT Architecture**: Implements a small-scale GPT with configurable parameters
- **Multi-Head Self-Attention**: Full implementation of transformer attention mechanisms
- **Layer Normalization & GELU**: Modern architectural components
- **Training Pipeline**: Complete training loop with data loading and optimization
- **Text Generation**: Generate text based on trained model checkpoints
- **Jupyter Notebooks**: Interactive notebooks for training and inference

## 🏗️ Model Architecture

The model uses the following configuration (defined in `picogpt/config.json`):

```json
{
  "vocab_size": 50257,
  "emb_dim": 256,
  "context_length": 128,
  "n_heads": 4,
  "n_layers": 4,
  "drop_rate": 0.1,
  "qkv_bias": true
}
```

### Architecture Components

- **Token & Position Embeddings**: Learned embeddings for input tokens and positions
- **Transformer Blocks**: 4 layers with multi-head self-attention
- **Feed-Forward Networks**: Position-wise FFN with GELU activation
- **Layer Normalization**: Pre-normalization for stable training
- **Dropout**: Regularization to prevent overfitting

## 📁 Project Structure

```
picogpt-python/
├── model.ipynb              # Main training notebook
├── model_load.ipynb         # Model loading and inference notebook
├── picogpt.pt               # Trained model checkpoint
├── picogpt/
│   └── config.json          # Model configuration
├── simple_english_phrases.txt  # Training data
├── simple_french_phrases.txt   # Additional French phrases
├── simple_french_dictionary.txt # French vocabulary
├── suggestion_1.md          # Development notes
├── LICENSE                  # License file
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- tiktoken (OpenAI's tokenizer)
- tqdm (for progress bars)
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/nazimboudeffa/picogpt-python.git
cd french-gpt-python

# Install dependencies
pip install torch tiktoken tqdm jupyter
```

### Training the Model

Open and run `model.ipynb` in Jupyter Notebook:

```bash
jupyter notebook model.ipynb
```

The notebook includes:
1. Model architecture definition
2. Data loading from `simple_english_phrases.txt`
3. Training loop with loss tracking
4. Model checkpoint saving

### Using a Trained Model

Open `model_load.ipynb` to load and generate text:

```bash
jupyter notebook model_load.ipynb
```

This notebook demonstrates:
- Loading saved model weights (`picogpt.pt`)
- Text generation with different parameters
- Temperature-based sampling

## 📊 Training Data

The project uses `simple_english_phrases.txt` containing simple sentence patterns with:
- Basic subjects (The cat, My dog, The teacher, etc.)
- Common verbs (eats, runs, jumps, sleeps, etc.)
- Simple objects and locations

This simplified dataset helps the model learn basic grammatical structures.

## 🔧 Configuration

Modify `picogpt/config.json` to experiment with different model sizes:

- `vocab_size`: Size of the vocabulary (default: 50257 for GPT-2 tokenizer)
- `emb_dim`: Embedding dimension (increase for larger models)
- `context_length`: Maximum sequence length
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `drop_rate`: Dropout probability
- `qkv_bias`: Whether to use bias in attention projections

## 💡 Usage Example

```python
import torch
from model import GPTModel, generate
import tiktoken

# Load configuration
cfg = {...}  # Your config

# Initialize model
model = GPTModel(cfg)
model.load_state_dict(torch.load('picogpt.pt'))
model.eval()

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "The cat"
generated_text = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8)
print(generated_text)
```

## 🐛 Troubleshooting

### Common Issues

1. **Incoherent outputs**: 
   - Train for more epochs (20+ recommended)
   - Lower the learning rate (1e-4)
   - Increase training data size

2. **High loss**: 
   - Ensure loss decreases progressively (from ~7 to ~2)
   - Check data preprocessing
   - Verify model architecture

3. **Out of memory**:
   - Reduce `context_length`
   - Decrease `emb_dim` or `n_layers`
   - Use smaller batch sizes

## 📈 Future Improvements

- [ ] Add learning rate scheduling
- [ ] Implement gradient clipping
- [ ] Add validation set and metrics
- [ ] Support for custom tokenizers
- [ ] Multi-language support (French, etc.)
- [ ] Beam search for generation
- [ ] Model quantization for deployment

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [picoGPT](https://github.com/jaymody/picoGPT) - Inspiration for this implementation

## 👤 Author

**Nazim Boudeffa**

- GitHub: [@nazimboudeffa](https://github.com/nazimboudeffa)
- Repository: [picogpt-python](https://github.com/nazimboudeffa/picgpt-python)

---

Built with ❤️ for learning and experimenting with transformer models.
