# CIFAR-10 Classification with CNN and Transfer Learning

This project implements and compares different deep learning approaches for image classification on the CIFAR-10 dataset using TensorFlow/Keras. The goal is to explore the effectiveness of:
- A custom Convolutional Neural Network (CNN)
- Data augmentation techniques
- Transfer learning with pretrained models
- Optimizer and batch size variations
- Model evaluation through confusion matrices and class-wise metrics

---

## Summary of Key Insights

### Training a CNN from Scratch
- The base CNN achieved approximately 75% accuracy with standard training.
- Overfitting began after a few epochs — mitigated using BatchNorm and Dropout.
- Smaller batch sizes helped improve generalization slightly.

### Impact of Data Augmentation
- Augmentations like horizontal flip, rotation, and zoom improved validation accuracy to around 80%.
- Reduced overfitting and increased robustness.
- Real-time augmentation was performed efficiently on GPU using Keras layers.

### Transfer Learning
- Using ResNet50 as a frozen feature extractor gave the best performance: around 87% test accuracy.
- Converged faster and generalized better than the custom CNN.
- Potential improvement: fine-tune the top layers of the pretrained model.

### Optimizer Comparison
| Optimizer | Accuracy | Convergence Speed |
|-----------|----------|--------------------|
| SGD       | ~71%     | Slow               |
| RMSprop   | ~76%     | Moderate           |
| Adam      | ~78%     | Fast               |

Adam consistently performed the best in terms of both speed and accuracy.

---

## Experiment Highlights

- Batch size comparison: smaller batches resulted in better generalization; larger batches trained faster.
- Dropout tuning showed 0.5 to be optimal for regularization.
- Training history was visualized using accuracy/loss curves.
- Final models were evaluated using precision, recall, F1-scores, and a confusion matrix.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/cifar10-cnn-classification.git
cd cifar10-cnn-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the notebook:
```bash
jupyter notebook cnn_cifer.ipynb
```

---

## Credits

- Developed as part of a Computer Vision assignment.
- ChatGPT was used to:
  - Clarify assignment instructions
  - Optimize training and evaluation strategies
  - Structure model comparisons and this report
