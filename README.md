# Graphical Normalizing Flows
Offical codes and experiments for the paper: 
> Graphical Normalizing Flows,  Antoine Wehenkel and Gilles Louppe.  (May 2020).
> [[arxiv]](https://arxiv.org/abs/2006.02548)
# Dependencies
The list of dependencies can be found in requirements.txt text file and installed with the following command:
```bash
pip install -r requirements.txt
```
# Code architecture
This repository provides code to build diverse normalizing flow models in PyTorch. The core components are located in **models** folder. The different flow models are described in the file **NormalizingFlow.py** and all follows the structure of the parent **class NormalizingFlow**.
A flow step is usually designed as a combination of a **normalizer** (such as the ones described in Normalizers sub-folder) with a **conditioner** (such as described in Conditioners sub-folder). Following the code hierarchy provided makes the implementation of new conditioners, normalizers or even complete flow architecture very easy.
# Paper's experiments
## UCI Datasets
You first have to download the datasets with the following command:
```bash
python UCIdatasets/download_dataset.py 
```
Then you can run the experiment of your choice with the following command:
```bash
python UCIExperiments.py -load_config <exp-name>
```
where <exp-name> defines the experimental configuration loaded from *UCIExperimentsConfigurations.yml* file, e.g. *power-mono-DAG*.
See also UCIExperiments.py for other optional arguments.
## MNIST 
### Affine Normalizers
##### Graphical  Conditioner
```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Affine -conditioner DAG -nb_flow 1 -nb_steps_dual 10 -l1 0. -prior_A_kernel 2
```
##### Autoregressive  Conditioner
```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Affine -conditioner Autoregressive -nb_flow 1 -emb_net 1024 1024 1024 2
```
##### Coupling  Conditioner

```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Affine -conditioner Coupling -nb_flow 1 -emb_net 1024 1024 1024 2
```
### Monotonic Normalizers
##### Graphical  Conditioner
```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Monotonic -conditioner DAG -nb_flow 1 -nb_steps_dual 10 -l1 0. -prior_A_kernel 2
```
##### Autoregressive  Conditioner
```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Monotonic -conditioner Autoregressive -nb_flow 1 -emb_net 1024 1024 1024 30
```
##### Coupling  Conditioner

```bash
python ImageExperiments.py -dataset MNIST -b_size 100 -normalizer Monotonic -conditioner Coupling -nb_flow 1 -emb_net 1024 1024 1024 30
```
