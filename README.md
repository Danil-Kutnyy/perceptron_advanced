# Exploring Complex Neural Network Architectures: DeepLinear and SubNetwork Models for MNIST Classification

## Abstract

This work explores the implementation and performance of complex neural network architectures designed to process individual input features with greater computational depth, diverging from the traditional \(x_i \cdot w_i\) computations. The proposed **DeepLinear** and **SubNetwork** models replace simple linear processing with multi-layer sub-networks for weights and biases. Despite promising theoretical underpinnings, the experimental results demonstrate limited performance gains on MNIST digit classification, achieving 95.99% accuracy compared to 97.56% by a baseline fully connected network. The findings suggest that while introducing architectural complexity has potential, achieving superior performance requires further refinement.

---

## Introduction

Traditional fully connected neural networks compute each output neuron as a sum of weighted inputs plus biases. This simplicity makes them computationally efficient but limits their ability to model complex relationships within the data. Inspired by biological neural computation, we introduce **DeepLinear**, which replaces linear operations with sub-networks for processing weights and biases. These sub-networks are tasked with extracting richer representations from inputs. This work evaluates the performance of DeepLinear against a baseline model and discusses its implications for complex function modeling.

---

## Methodology

### Dataset and Preprocessing

We used the MNIST dataset, comprising 60,000 training and 10,000 test grayscale images of handwritten digits, normalized to \([-1, 1]\). Images were resized to \(28 \times 28\) pixels and flattened for fully connected layers.

### Baseline Architecture

The baseline model is a traditional multi-layer perceptron (MLP) with three fully connected layers. The architecture is as follows:

1. **Input Layer**: Flattened 28 × 28 input pixels.
2. **Hidden Layer 1**: 256 neurons with ReLU activation, followed by LayerNorm.
3. **Hidden Layer 2**: 128 neurons with ReLU activation, followed by LayerNorm.
4. **Output Layer**: 10 neurons with Softmax activation for classification.

### Proposed Models

#### **DeepLinear**

DeepLinear introduces additional complexity by replacing simple weight and bias computations with multi-dimensional parameter tensors. Key architectural components:

- **Parameters**: 
  - Weights (\(w1\), \(w21\), \(w22\)) and biases (\(b1\), \(b21\), \(b22\)) are four-dimensional tensors initialized uniformly in \([-2, 2]\).
- **Layer Normalization**: Applied after every computation stage to stabilize learning.
- **Multi-Dimensional Operations**:
  - Input tensors are expanded into \( (batch, input, units, 2) \) shapes.
  - Intermediate representations are computed using element-wise multiplications and reductions.
- **Non-Linearity**: Leaky ReLU activation after each layer to introduce non-linearity.

#### **SubNetwork Model**

The SubNetwork model replaces weights in fully connected layers with **SimpleNN** sub-networks, which process individual input features. SimpleNN has the following architecture:

1. \(1 \rightarrow 2 \) Linear transformation with ReLU activation.
2. \(2 \rightarrow 2 \) Linear transformation with ReLU activation.
3. \(2 \rightarrow 1 \) Linear transformation.

These sub-networks process each input feature independently, creating a richer feature map for subsequent layers.

### Training

All models were trained for 10 epochs using the AdamW optimizer (\(lr=0.0003\)), minimizing cross-entropy loss. Batch size was set to 64 for training and 1000 for testing. 

---

## Results and Discussion

### MNIST Classification

| **Model**       | **Epochs** | **Accuracy (%)** |
|------------------|------------|------------------|
| Baseline MLP     | 10         | **97.56**        |
| DeepLinear       | 10         | 95.99            |
| SubNetwork       | 10         | 93.11            |

The Baseline model outperformed the proposed architectures. DeepLinear achieved 95.99% accuracy, slightly lower than the baseline, despite its complex operations. SubNetwork performed the worst, likely due to overfitting or insufficient feature generalization.

### Complexity vs. Performance

DeepLinear and SubNetwork models introduce higher parameter counts and computational overhead. Despite this, the increase in architectural complexity did not translate into superior performance. Key challenges include:

1. **Overfitting**: The added depth may lead to overfitting due to limited training data.
2. **Optimization Difficulties**: The high number of parameters requires more sophisticated training strategies.
3. **Feature Redundancy**: Processing each feature independently limits inter-feature learning.

---

## Additional Experimentation

To evaluate generalizability, the DeepLinear architecture was integrated into a GPT-style transformer model and tested on Shakespeare's text dataset. Results were poor, suggesting that these complex weight-processing mechanisms may not align with sequence-based tasks.

---

## Conclusion and Future Work

This work demonstrates that adding computational depth to weight and bias processing introduces new possibilities but does not guarantee better performance. Future work should focus on:

1. **Regularization**: To address overfitting.
2. **Hybrid Architectures**: Combining traditional and complex layers.
3. **Task-Specific Optimization**: Designing architectures tailored to specific problems.

This work contributes to ongoing research on creating biologically inspired, computationally rich neural networks and provides insights for future explorations in architectural innovation.
