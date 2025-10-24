# Object and Text Classification Using Deep Learning

This project explores CNN and RNN architectures for both image and text classification tasks.

## Tasks
- **Object Recognition:** CIFAR-10 dataset with CNNs, grid search on convolutional filters, and optimizer comparison.
- **Text Classification:** Wikipedia paragraphs classified into 15 categories using word/character embeddings and RNN variants (GRU, LSTM).

## Techniques
- Grid search and dropout regularization
- Word-level vs character-level embeddings
- Model comparison: CNN vs RNN performance and convergence speed

## Results
- Improved CIFAR-10 accuracy from 34% → 54% using optimized CNN.
- Text classification up to **87.9%** accuracy using WordRNN with dropout.

## Tech Stack
- Python, Keras, TensorFlow, NumPy
