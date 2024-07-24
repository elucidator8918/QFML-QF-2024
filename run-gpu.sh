#!/bin/bash

pip install tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas opacus pyyaml tenseal kaggle sentence_transformers torch torchvision torchaudio streamlit zenodo-get
git clone https://github.com/data-science-lover/flower.git
cd flower
pip install .