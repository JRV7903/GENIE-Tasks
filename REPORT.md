# ML4SCI GENIE GSoC 2026 – Evaluation Tasks Report

## 1. Overview

This report presents the implementation and evaluation of the ML4SCI GENIE GSoC tasks, including:

* Autoencoder-based representation learning of jet images
* Graph-based jet classification using Graph Neural Networks (GNNs)
* Non-local GNN with self-attention for modeling long-range dependencies

The goal is to explore effective representations of quark and gluon jets and evaluate the impact of graph-based modeling and non-local interactions on classification performance.

---

## 2. Dataset and Preprocessing

The dataset consists of simulated jet events represented as three-channel images:

* ECAL (Electromagnetic Calorimeter)
* HCAL (Hadronic Calorimeter)
* Tracks

Each event is represented as a **125 × 125 × 3** image. The data is highly sparse, with only a small fraction of pixels containing non-zero energy deposits.

### Preprocessing Steps

* Images are normalized per sample using maximum energy scaling
* Non-zero pixels are extracted to form sparse representations
* Each non-zero pixel is converted into a node with features:

  * Spatial coordinates (x, y)
  * Energy value
  * Channel index

This transformation enables representing each jet as a point cloud suitable for graph construction.

---

## 3. Task 1: Autoencoder for Representation Learning

### Objective

To verify that jet images contain learnable structure that can be compressed into a lower-dimensional latent space while preserving relevant physical information.

### Model Architecture

A convolutional autoencoder was implemented with:

* Encoder: 3 convolutional layers (3 → 16 → 32 → 64 channels)
* Latent space: Fully connected projection
* Decoder: Symmetric transposed convolutional layers
* Activation: ReLU (hidden), Sigmoid (output)

### Training Details

* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Epochs: 20
* Batch size: 128

### Results

* Final Test MSE: **0.000499**

The low reconstruction error indicates that the jet images lie on a structured manifold that can be effectively captured in a compressed representation.

![WhatsApp Image 2026-03-31 at 03 47 13](https://github.com/user-attachments/assets/252174e2-bd4f-4d4a-ad45-8773fc61465c)

Figure 1: Original and reconstructed jet images across ECAL, HCAL, and Tracks channels. The autoencoder successfully captures the spatial distribution of energy deposits, preserving key structural features despite strong sparsity in the input.
The reconstructed outputs exhibit slight spatial diffusion due to the smoothing effect of convolutional layers, but retain the key localization and structure of energy deposits.

The reconstructed outputs preserve the spatial distribution of energy deposits across all channels, confirming that the model successfully learns meaningful features.

---

## 4. Task 2: Jets as Graphs

### Graph Construction

Jets are converted into graphs using the following pipeline:

1. Extract all non-zero pixels
2. Convert to point cloud representation
3. Apply top-400 truncation based on energy
4. Construct edges using k-Nearest Neighbors (k = 12)
5. Assign edge weights using inverse distance

This representation eliminates redundant zero regions and captures the intrinsic structure of jet events.

---

### Baseline Model: Residual GCN

The baseline classifier consists of:

* Three Residual GCN blocks
* Batch Normalization and ReLU activation
* Global mean and max pooling
* Fully connected classification head

### Training Configuration

* Loss: Binary Cross-Entropy with logits
* Optimizer: AdamW
* Epochs: 30
* Dataset size: 50,000 samples

---

### Results

| Model        | ROC-AUC |
| ------------ | ------- |
| Baseline GCN | 0.7674  |

The baseline model demonstrates the effectiveness of graph representations for jet classification by capturing local spatial dependencies.

---

## 5. Task 4: Non-Local GNN

### Motivation

Standard GNNs operate on local neighborhoods defined by graph connectivity. However, jet physics often involves long-range correlations that are not captured by local message passing.

---

### Model Extension

To address this, a non-local attention mechanism was introduced:

* Multi-head self-attention layer
* Applied after GCN feature extraction
* Uses dense batching via `to_dense_batch`

This allows interactions between all nodes within a graph, enabling global context modeling.

---

### Results

| Model         | ROC-AUC |
| ------------- | ------- |
| Baseline GCN | 0.7674 |
| Non-local GNN | 0.7850 |

<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/a848e24f-c7eb-4bf2-9cd2-1d574d854a26" />

Figure 2: ROC curve comparison between the baseline GCN and the non-local GNN model. The non-local architecture achieves improved performance (AUC: 0.7850 vs 0.7674), demonstrating the benefit of incorporating long-range dependencies through self-attention.


The non-local model achieves improved performance, demonstrating the importance of long-range dependencies in jet classification.

---

## 6. Analysis and Key Insights

* **Graph representations** significantly improve efficiency by eliminating sparse regions
* **Residual GCNs** effectively capture local structure without over-smoothing
* **Non-local attention** provides measurable gains by modeling global interactions
* The improvement in ROC-AUC confirms that long-range dependencies are relevant for jet classification

---

## 7. Computational Considerations

The introduction of non-local attention introduces a key limitation:

* Dense batching (`to_dense_batch`) leads to **O(N²)** memory complexity
* Memory usage scales rapidly with node count
* Limits scalability for larger graphs and datasets

This bottleneck highlights the need for more efficient attention mechanisms.

---

## 8. Limitations

* Quadratic memory scaling in attention layer
* Static graph construction (fixed k-NN)
* No incorporation of physics-based symmetries
* Limited exploration of hyperparameter space

---

## 9. Future Work

The following directions can improve performance and scalability:

* Replace dense attention with **linear attention (O(N))**
* Introduce **dynamic graph construction (EdgeConv)**
* Incorporate **Lorentz-equivariant features**
* Optimize training and inference for large-scale datasets

---

## 10. Conclusion

This work demonstrates that graph-based representations combined with non-local attention provide a strong foundation for quark–gluon jet classification. The results show clear improvements over baseline models, while also identifying key computational bottlenecks.

The findings motivate further exploration of scalable attention mechanisms and physics-informed architectures to advance the state of the art in geometric deep learning for high-energy physics.

---
