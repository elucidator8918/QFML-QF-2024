# QFed+FHE: Quantum Federated Learning with Secure Fully Homomorphic Encryption (FHE)

![image-E7XxcHrEu-transformed](https://github.com/user-attachments/assets/a4edc92a-0d16-4758-867f-0466edd6af2d)

## Overview

Welcome to the Quantum Federated Learning (QFL) repository for our cutting-edge project, which utilizes Secure Fully Homomorphic Encryption (FHE). This initiative, developed for the QF 2024 Hackathon, seeks to advance privacy-preserving machine learning in quantum environments.

## Repository Structure

```
.
├── assets/
├── dashboard_src/
│   ├── dashboard_utils/
│   └── utils/
├── Flower/
├── src/
│   ├── utils/
│   ├── FHE_FedQNN_CIFAR.ipynb
│   ├── FHE_FedQNN_DNA.ipynb
│   ├── FHE_FedQNN_MRI.ipynb
│   ├── Standard_FedQNN_CIFAR.ipynb
│   ├── Standard_FedQNN_DNA.ipynb
│   └── Standard_FedQNN_MRI.ipynb
├── dashboard.py
├── run-cpu.sh
├── run-gpu.sh
├── .gitignore
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/elucidator8918/QFML-QF-2024.git
cd QFML-QF-2024
```

### Install Dependencies

#### For CPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-cpu.sh
```

#### For GPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-gpu.sh
```

## Usage

### Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

#### FHE-enabled Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - Notebook: `src/FHE_FedQNN_CIFAR.ipynb`
  - Description: CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is widely used for image classification tasks.

- **DNA Sequence Dataset:**
  - Notebook: `src/FHE_FedQNN_DNA.ipynb`
  - Description: This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **MRI Scan Dataset:**
  - Notebook: `src/FHE_FedQNN_MRI.ipynb`
  - Description: This dataset contains MRI scans used for medical image analysis, particularly for detecting and diagnosing conditions based on scan data.

#### Standard Quantum Federated Learning

- **CIFAR-10 Dataset:**
  - Notebook: `src/Standard_FedQNN_CIFAR.ipynb`
  - Description: The same CIFAR-10 dataset, utilized without the FHE layer, for benchmarking and comparison.

- **DNA Sequence Dataset:**
  - Notebook: `src/Standard_FedQNN_DNA.ipynb`
  - Description: The same DNA sequence dataset, used without FHE for standard federated learning experiments.

- **MRI Scan Dataset:**
  - Notebook: `src/Standard_FedQNN_MRI.ipynb`
  - Description: The same MRI scan dataset, used without FHE to evaluate the performance of standard federated learning models.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Thank you to the QF 2024 Hackathon organizers for providing us with this opportunity.

## Contact

For inquiries, please reach out to Team DuoLicht at:
- forsomethingnewsid@gmail.com
- pavana.karanth17@gmail.com
