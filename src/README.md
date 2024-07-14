# Quantum Federated Learning with Fully Homomorphic Encryption (FHE)

This directory contains notebooks and scripts for running experiments on different datasets using Quantum Federated Learning models, with and without Fully Homomorphic Encryption (FHE). Our approach enhances data privacy and security while leveraging the computational advantages of quantum neural networks.

## Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

### FHE-enabled Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - **Notebook:** `FHE_FedQNN_CIFAR.ipynb`
  - **Description:** The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is widely used for image classification tasks.

- **DNA Sequence Dataset:**
  - **Notebook:** `FHE_FedQNN_DNA.ipynb`
  - **Description:** This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **MRI Scan Dataset:**
  - **Notebook:** `FHE_FedQNN_MRI.ipynb`
  - **Description:** This dataset contains MRI scans used for medical image analysis, particularly for detecting and diagnosing conditions based on scan data.

### Standard Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - **Notebook:** `Standard_FedQNN_CIFAR.ipynb`
  - **Description:** The same CIFAR-10 dataset, utilized without the FHE layer, for benchmarking and comparison.

- **DNA Sequence Dataset:**
  - **Notebook:** `Standard_FedQNN_DNA.ipynb`
  - **Description:** The same DNA sequence dataset, used without FHE for standard federated learning experiments.

- **MRI Scan Dataset:**
  - **Notebook:** `Standard_FedQNN_MRI.ipynb`
  - **Description:** The same MRI scan dataset, used without FHE to evaluate the performance of standard federated learning models.

## Datasets

Download the datasets using the following commands:

```bash
# DNA Sequence Dataset
kaggle datasets download -d nageshsingh/dna-sequence-dataset
mkdir -p data/DNA
unzip dna-sequence-dataset.zip -d data/DNA
rm dna-sequence-dataset.zip

# MRI Scan Dataset
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
mkdir -p data/MRI
unzip brain-tumor-mri-dataset.zip -d data/MRI
rm brain-tumor-mri-dataset.zip
```
