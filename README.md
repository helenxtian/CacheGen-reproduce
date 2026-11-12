# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

**For the latest update and integration, please check out the [LMCache](https://github.com/LMCache/LMCache) project!**

> **Reproduction on Chameleon Cloud (UChicago CMSC 330 reproducibility project).** This fork reproduces **Figure 9** and **Table 1** of the paper.
> - **Start here:** the notebook [`cachegen_fig9_table1.ipynb`](cachegen_fig9_table1.ipynb). Open it from Chameleon's JupyterHub and run the cells in order. It reserves a GPU node, installs everything, runs the experiment, and checks the results against the paper.
> - **Our results and analysis:** [`reproduce-results/README.md`](reproduce-results/README.md). In short: CacheGen 173 MB vs. 8-bit quantization 587 MB on LongChat, a 3.4x reduction; the paper reports 176 vs. 622 MB, 3.5x.
> - **Our scripts:** [`scripts/reproduce/`](scripts/reproduce/README.md).

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf) (SIGCOMM'24). 
The code structure is organized as follows:

- ```LMCache```: The modules for KV cache encoding / decoding with CacheGen's customized codec 
- ```test_data```: The example testing cases for CacheGen. 
- ```src```: Some helper functions used by CacheGen (e.g., transforming tensor to tuple, transforming tuple to tensor etc.)

## Installation

To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yaml
conda activate cachegen
pip install -e LMCache
cd LMCache/third_party/torchac_cuda 
python setup.py install
```

### Examples 

Please refer to the page [sigcomm_ae.md](sigcomm_ae.md) for running examples for CacheGen. 

### Contact 
Yuhan Liu (yuhanl@uchicago.edu)
