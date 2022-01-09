Meta Learning Backpropagation And Improving It (VSML)
=====================================================

This is research code for the NeurIPS 2021 publication [Kirsch & Schmidhuber 2021](https://arxiv.org/abs/2012.14905).

*Many concepts have been proposed for meta learning with neural networks (NNs), e.g., NNs that learn to reprogram fast weights, Hebbian plasticity, learned learning rules, and meta recurrent NNs. Our Variable Shared Meta Learning (VSML) unifies the above and demonstrates that simple weight-sharing and sparsity in an NN is sufficient to express powerful learning algorithms (LAs) in a reusable fashion. A simple implementation of VSML where the weights of a neural network are replaced by tiny LSTMs allows for implementing the backpropagation LA solely by running in forward-mode. It can even meta learn new LAs that differ from online backpropagation and generalize to datasets outside of the meta training distribution without explicit gradient calculation. Introspection reveals that our meta learned LAs learn through fast association in a way that is qualitatively different from gradient descent.*

## Installation

Create a virtual env
```bash
python3 -m venv venv
. venv/bin/activate
```

Install pip dependencies
```bash
pip3 install --upgrade pip wheel setuptools
pip3 install -r requirements.txt
```

Initialize [weights and biases](https://wandb.ai/)
```bash
wandb init
```

Inspect your results at https://wandb.ai/.

## Run instructions

### Non distributed

For any algorithm that does not require multiple workers.
```bash
python3 launch.py --config_files CONFIG_FILES --config arg1=val1 arg2=val2
```

### Distributed

For any algorithm that does require multiple workers
```bash
GPU_COUNT=4 mpirun -n NUM_WORKERS python3 assign_gpu.py python3 launch.py
```
where `NUM_WORKERS` is the number of workers to run.
The `assign_gpu` python script distributes the mpi workers evenly over the specified GPUs

Alternatively, specify the `CUDA_VISIBLE_DEVICES` instead of `GPU_COUNT` env variable:
```bash
CUDA_VISIBLE_DEVICES=0,2,3 mpirun -n NUM_WORKERS python3 assign_gpu.py python3 launch.py
```

### Slurm-based cluster

Modify `slurm/schedule.sh` and `slurm/job.sh` to suit your environment.
```bash
bash slurm/schedule.sh --nodes=7 --ntasks-per-node=12 -- python3 launch.py --config_files CONFIG_FILES
```
If only a single worker is required (non-distributed), set `--nodes=1` and `--ntasks-per-node=1`.

### Remote (via ssh)

Modify `ssh/schedule.sh` to suit your environment.
Requires [gpustat](https://pypi.org/project/gpustat/) in `.local/bin/gpustat`, via `pip3 install --user gpustat`.
Also install `tmux` and `mpirun`.
```bash
bash ssh/schedule.sh --host HOST_NAME --nodes=7 --ntasks-per-node=12 -- python3 launch.py --config_files CONFIG_FILES
```

## Example training runs

### Section 4.2 Figure 6

VSML
```bash
slurm/schedule.py --nodes=128 --time 04:00:00 -- python3 launch.py --config_files configs/rand_proj.yaml
```
You can also try fewer nodes and use `--config training.population_size=128`. 
Or use backpropagation-based meta optimization `--config_files configs/{rand_proj,backprop}.yaml`. 

### Section 4.4 Figure 8

VSML
```bash
slurm/schedule.py --array=1-11 --nodes=128 --time 04:00:00 -- python3 launch.py --array configs/array/datasets.yaml
```
Meta RNN (Hochreiter 2001)
```bash
slurm/schedule.py --array=1-11 --nodes=32 --time 04:00:00 -- python3 launch.py --array configs/array/datasets.yaml --config_files configs/{metarnn,pad}.yaml --tags metarnn
```
Fast weight memory
```bash
slurm/schedule.py --array=1-11 --nodes=32 --time 04:00:00 -- python3 launch.py --array configs/array/datasets.yaml --config_files configs/{fwmemory,pad}.yaml --tags fwmemory
```
SGD
```bash
slurm/schedule.py --array=1-4 --nodes=2 --time 00:15:00 -- python3 launch.py --array configs/array/sgd.yaml --config_files configs/sgd.yaml --tags sgd
```
Hebbian
```bash
slurm/schedule.py --array=1-11 --nodes=32 --time 04:00:00 -- python3 launch.py --array configs/array/datasets.yaml --config_files configs/{hebbian,pad}.yaml --tags hebbian
```

