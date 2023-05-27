# HomCountGNNs
This repository contains the code of the GNN part of the paper **[Expectation-Complete Graph Representations with Homomorphisms]()** (ICML, 2023).

## Setup

1. Clone repository
```
git clone https://github.com/ocatias/HomCountGNNs/
cd HomCountGNNs
```

2. Create and activate conda environment (this assume miniconda is installed)
```
conda create --name HOM
conda activate HOM
```

3. Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

4. Install PyTorch (Geometric)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
```

5. Install remaining dependencies
```
pip install -r requirements.txt
```

## Recreating experiments
Run experimentes with the following scripts. Results will be in the Results directory.

**Main experiments.** Homomorphism counts against no homomorphism counts:
```
bash Scripts/large_datasets.sh
bash Scripts/small_datasets.sh
```

**Ablation.** Impact of misaligned homomorphism counts:
```
bash Scripts/misaligned_feats.sh
```
