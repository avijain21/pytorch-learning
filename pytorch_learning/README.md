# PyTorch Learning Path: Beginner to Intermediate

A hands-on curriculum with real-world datasets from Kaggle and standard ML benchmarks.
Every notebook cell includes **🧠 Brain Analogies** (plain-language neuroscience mental models) and **⚙️ Engineer Analogies** (precise CS/hardware context) to make concepts accessible to complete beginners.

## Modules

| Module | Level | Topic | Dataset | Use Case |
|--------|-------|-------|---------|----------|
| 01 | Beginner | Tensors & Fundamentals | Synthetic / NumPy | Math ops, GPU basics |
| 02 | Beginner | First Neural Network | Iris (UCI/Kaggle) | Multi-class classification |
| 03 | Beginner-Mid | Data Pipeline | Titanic (Kaggle) | Binary classification |
| 04 | Intermediate | CNNs & Computer Vision | CIFAR-10 (torchvision) | Image classification |
| 05 | Intermediate | Transfer Learning | Dogs vs Cats (Kaggle) | Fine-tuning ResNet |
| 06 | Intermediate | RNNs / LSTM | IMDB Sentiment (torchtext) | Sentiment analysis |

## How Each Notebook is Structured

Each cell contains three layers of explanation:

1. **The concept** — what PyTorch code is doing
2. **🧠 Brain Analogy** — how it maps to how your brain works (neurons, memory, attention, etc.)
3. **⚙️ Engineer Analogy** — the precise technical/hardware view (GEMM, BPTT, PCIe DMA, etc.)

This dual-track approach lets beginners build intuition first, then layer on the technical depth.

## Setup

```bash
pip install torch torchvision torchaudio
pip install pandas scikit-learn matplotlib seaborn kaggle
pip install torchtext datasets
pip install jupyter notebook
```

## Kaggle Dataset Sources

- Titanic: https://www.kaggle.com/competitions/titanic/data
- Dogs vs Cats: https://www.kaggle.com/c/dogs-vs-cats/data
- Iris: Built into sklearn / https://www.kaggle.com/datasets/uciml/iris

## References

- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [LearnPyTorch.io](https://www.learnpytorch.io)
- [DataCamp PyTorch Cheatsheet](https://www.datacamp.com/cheat-sheet/deep-learning-with-py-torch)
- [Kaggle PyTorch Notebooks](https://www.kaggle.com/tags/torch)
