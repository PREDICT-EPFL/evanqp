# EVANQP

[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2206.13374-yellow.svg)](https://doi.org/10.48550/arXiv.2206.13374) [![Preprint](https://img.shields.io/badge/Preprint-arXiv-orange.svg)](https://arxiv.org/abs/2206.13374) [![Funding](https://img.shields.io/badge/Grant-NCCR%20Automation%20(51NF40180545)-red.svg)](https://nccr-automation.ch/)

EPFL Verifier for Approximate Neural Networks and QPs

This repository provides the code accompanying the paper [Stability Verification of Neural Network Controllers using Mixed-Integer Programming](https://arxiv.org/abs/2206.13374).

## Getting Started

To get started with the code, clone this repo, and install the evanqp python package with
```
python setup.py install
```
Jupiter Notebooks with examples can be found in `examples/`.

### Running Benchmarks

There are benchmarks available for two different examples available: the dc-dc converter example (`examples/dc_dc_converter/`) and the lipschitz example (`examples/lipschitz/`). To run the benchmarks change the parameters in `run_benchmarks.sh` to match your hardware configuration and execute the benchmark with
```
bash run_benchmarks.sh
```
The results can then be analysed in the Jupiter Notebook `benchmark_analysis.ipynb`.

## Citing our Work

To cite our work in other academic papers, please use the following BibTex entry:
```
@misc{schwan2022,
author = {Roland Schwan and Colin N. Jones and Daniel Kuhn},
title = {Stability Verification of Neural Network Controllers using Mixed-Integer Programming},
year = {2022},
eprint = {arXiv:2206.13374},
}
```