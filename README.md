# Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts

This repository is the official implementation for the paper: [Meta-Unlearning on Diffusion Models:Preventing Relearning Unlearned Concepts]()

## Installation

We follow the [Diffuser](https://github.com/huggingface/diffusers) to install the required dependencies, please run the following commands:

```shell
conda create -n meta python=3.10
conda activate meta
pip install --upgrade diffusers[torch]
```

*All experiments are conducted on NVIDIA A100 GPUs with 80GB of memory.*

## Generate images datasets
First you can use the shell scripts to generate **hrm** dataset, **irt** dataset and **target** dataset. In our code, the **hrm** dataset means the unlearned concept, the **irt** dataset means the retain dataset, and the **target** dataset means the concept realted to unlearned concept. 

Here is the exmaple generate scipt, you can change the text prompt according to your goal.
```shell
bash scripts/tofu/me_gd.sh
```

## Baseline: unlearning

```shell
bash scripts/esd_baseline.sh
bash scripts/sdd_baseline.sh
```

[UCE](https://github.com/rohitgandikota/unified-concept-editing) and [RECE](https://github.com/CharlesGong12/RECE) can be trained by the original code in their paper.


## Our paradigm: meta-unlearning

For ESD and SDD based meta-unlearning:

```shell
bash scripts/esd_meta.sh
bash scripts/sdd_meta.sh
```
Note that we only give the example hyperparameter in code and you should change the hyperparameter refer to our paper. 

For UCE and RECE based meta-unlearning:

```shell
bash scripts/meta.sh
```
Note that you should change the model path to your unlearned model.

## Acknowledgments

This repository is based on the codebase of the [SDD](https://github.com/nannullna/safe-diffusion/tree/main). Thanks for their impressive works!
