# ML4SCI Jet Classification

Project for quark-gluon jet classification using GNNs and Non-local Attention.

## Structure
- `main.py`: Entry point for local execution.
- `src/genie-tasks.ipynb`: Main notebook covering all tasks (AE, Baseline GNN, Non-local GNN).
- `src/`: Model, data, and training logic.
- `configs/`: YAML configs.

## Methodology
1. **Graph Construction**: k-NN graph (k=12) from jet point clouds. 
2. **Model**: GCN with residual blocks and global pooling (mean/max).
3. **AE**: CNN-based autoencoder for feature sanity checks.
4. **Non-local GNN**: Integration of multi-head self-attention to capture long-range topological correlations.

## Usage & Setup

Due to the extreme size of the dataset (~140K samples) and the computational intensity of training graph neural networks, executing the training pipeline on a local CPU is prohibitively slow. 

**For Evaluation:**
Please run the provided `src/genie-tasks.ipynb` notebook natively on Kaggle or Google Colab environments leveraging GPU hardware (T4 / P100). 
- All hyperparameter configurations and datasets paths are natively embedded within the notebook environment to execute linearly.

If running locally with sufficient hardware (e.g. CUDA bounds):
1. Install requirements: `pip install -r requirements.txt`
2. Run standard training: `python main.py`

Outputs (reconstruction images, loss curves, and AUC comparisons) are generated during execution.

## Model Justification
- **CNN Autoencoder**: Convolutional layers natively capture local spatial hierarchies, making them an excellent baseline to distill the highly structured but sparse multi-channel (ECAL/HCAL/Tracks) visual layout of quark-gluon events into a latent manifold.
- **Graph Representation**: Jets are intrinsically unordered clusters of energy deposits. Mapping only non-zero pixels into a k-NN point cloud immediately eliminates massive computational overhead from zero-padded bounds and restores translation invariance.
- **Baseline GCN**: Graph Convolutional Networks with residual pathways prevent over-smoothing while serving as an industry-standard baseline for aggregating local point-cloud neighborhoods.
- **Non-local Attention**: While GCNs only observe local 12-hop connections, non-local multi-head attention acts dynamically across the entire node set, explicitly modeling long-range physical correlations between separated energy clusters.

## Results and Discussion
- **Evaluation Environment & Scale**: Training was executed on a subset of 50,000 jets using Kaggle's Dual Tesla T4 configuration. This scale confirms the model's ability to handle significant topological complexity within standard session limits.
- **Reconstruction Quality**: The autoencoder accurately reconstructs energy density regions, yielding a **Final Test MSE of 0.000499**, validating the sparse representation approach.
- **Jets as Graphs**: Representing jets structurally provides a substantial inductive bias over pixel grids, leading to stable representation surfaces for analysis.
- **Baseline vs Non-local GNN**: The baseline model effectively segregates quark and gluon patterns. However, the Non-local GNN demonstrably improves performance by establishing message passing across spatially separated clusters.
- **ROC-AUC Performance**: The comparison curve confirms the discriminative superiority of Non-Local Attention. The benchmark results:
    - **Baseline AUC**: 0.7750
    - **Non-Local AUC**: 0.7859
