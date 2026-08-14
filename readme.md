# Physics-Informed Neural Networks for Fault Diagnosis
This repo provides basic features to setup, train and evaluate physics-informed neural networks (PINNs) for fault diagnosis of the ReVolt vessel. It consists of the following modules:
- **architecture:** Base class + example to implement NN-based fault estimator
- **diagnosis:** Base class for fault diagnosis module + examples of other fault estimation approaches, runnable as part of the rest of the simulation framework
- **eval:** Evaluation of the model on the test set
- **excitation:** A collection of common excitation signals to be used as actuator commands when generating a new dataset
- **tune:** (under development) Implementation of optuna for hyperparameters tuning
- **utils:** A collection of functions that might be useful across different parts of the code
- **vessel:** GNC architecture for the ReVolt vessel. Note that simulations are handled by the `PythonVehicleSimulator` submodule.

This repository is still in early development. If you catch any bug / issue, please report it in the *Issues* tab. 

# TODOs
- Design, implement and train a PINN for actuators fault estimation
- Write a paper
- Win award for your paper
- Win Nobel Prize for your paper
- Relax


# Installation
```
git clone --recurse-submodules git@github.com:stiefen1/pinn-fault-diagnosis.git
```

Make sure each submodule is set to branch ```pinn-fault-diagnosis``` and then create your conda environment based on the provided ```env.yml```:

```
conda env create -f env.yml
```
and activate it with 
```
conda activate pinn-fault-diagnosis
```
Install PyTorch 

- On the cluster: ```pip install torch --index-url https://download.pytorch.org/whl/cu124```
- On your local machine (CPU-only): ```pip install torch=2.12.0```

Then install the submodule with

```
pip install -e submodules/PythonVehicleSimulator
```

# Setting up NTNU's High Performance Computing (HPC) unit
## Request access
To get access to the [HPC unit](https://www.hpc.ntnu.no/idun/how-to-get-access-to-idun/), send an e-mail to support@itk.ntnu.no mentioning who you are and why you need access to it and put your main supervisor and I as CC.
## Connect
Once access was granted to you, open a terminal and run
```
ssh <username>@idun-login1.hpc.ntnu.no
```
## Launch your jobs
Once you've installed this repo on the HPC, a training session can be launched using:

```python -m scripts.launch_slurm.py -c configs/train.yaml --submit```

You can check the status of your launched jobs using `squeue --me`, or cancel it with `scancel <jobid>`. To ease your work, we highly recommend you to mount your personal folder from the hpc to easily access it from your local machine. Among other things, this will allow you to load weights trained on the cluster directly from your personal computer. To do so, please follow the instructions of **Method 2** [here](https://www.hpc.ntnu.no/idun/documentation/transferring-data/#ib-toc-anchor-4).

# Usage
All the code provided in this repository is made to be runnable both from your local machine and on NTNU's HPC. Most of the features rely heavily on high-level configuration files (.yaml).

## Running scripts locally
After installing the code as explained before on your computer, you can simply run scripts from the base folder of this repo in command line. Here are the most important ones:
- `python -m src.train -c configs/train.yaml`
- `python -m src.dataset.generator -c configs/dataset.yaml`

### Tracking performance along learning (Tensorboard)
Open another terminal and launch
```tensorboard --logdir <path_to_your_log_dir>```
make sure you replace ```<path_to_your_log_dir>``` by a valid logging directory (e.g. `.\tb_logs\demo\`).

## Running scripts on the HPC
After installing the code as explained before in your personal session **on the cluster**, you can launch jobs (dataset generation or training) on it using the [`scripts/launch_slurm.py`](scripts/launch_slurm.py) file:
- `python -m scripts.launch_slurm -c configs/train.yaml --submit`
- `python -m scripts.launch_slurm -c configs/dataset.yaml --submit`

### Tracking performance along learning (Tensorboard)
Open another terminal on the cluster and run
```
tensorboard --logdir <path_to_your_log_dir> --port 6006
```
Make sure you replace ```<path_to_your_log_dir>``` by a valid logging directory (e.g. `.\tb_logs\demo\`). On your laptop, open an ssh tunnel that forwards port 6006 to your local machine:
```
ssh -L 6006:localhost:6006 <username>@idun-login1.hpc.ntnu.no
```
And then open ```http://localhost:6006```

## Generating Dataset
This feature is mainly done through the FaultIdentificationDatasetGenerator class located in [`src/dataset/generator.py`](src/dataset/generator.py). The most important entries of the associated configuration file are:
- **episodes** : Describes how large we want the dataset to be
- **dataset** : Where to save the resulting data
- **env** : What are the environmental disturbances acting on the vessel throughout simulations
- **vessel** : What is the vessel's GNC architecture, what excitation signals do we use and what kind of faults can occur

To generate a dataset based on a given configuration file, simply run:
```
python -m src.dataset.generator -c <path_to_config>
```
Where <path_to_config> must be a path starting from the base folder of this repository to a valid .yaml file (e.g. ['configs/dataset.yaml'](configs/dataset.yaml)).

### Side note on dataset generation
In system identification (sysid), an important part of the job is to find control commands that make the identification of unknown parameters easier. In non-linear systems, applying different control commands can result in different observability conditions for the parameters to be estimated. The direct consequence is that the same fault can be more challenging (or even impossible) to detect in some situations. When generating a fault identification dataset, it's not only important to cover a wide range of states to avoid out-of-distribution (OOD) samples, but also to make sure we train our system in both easy and challenging configurations. 

### Parameterizing auxiliary signals for dataset generation
In your configuration file (e.g. [`configs/dataset.yaml`](configs/dataset.yaml)), you can manage whether you want your dataset to be generated solely using the NMPC controller, an auxiliary excitation signal, or both together. 

To enable an auxiliary excitation signal, simply set vessel.auxiliary_excitation.enabled to **true**. Then, you can provide the desired excitation signal as an explicit string for both the azimuth angle and thruster speed of each thruster (port and starboard). For example

`azimuth: 'Sinus(frequency=0.1, amplitude=-3.14159, time_delay_sec=0.0, y_offset=0.0) + GaussianNoise(amplitude=0.5)'`

The complete list of available signals is available in [`src/excitation/signals.py`](src/excitation/signals.py).

To enable control commands from the NMPC controller, simply set vessel.control.enabled to **true**. If an auxiliary signals was requested as well, the resulting control commands will be the sum of both. This could be useful e.g. to mix NMPC control commands with gaussian noise to obtain a more diverse dataset of excitation signals. 

## Designing a new NN architecture
All the NN architecture to be tested must inherit from the base class `BaseFaultEstimatorNN`, located in [`src/architecture/base.py`](src/architecture/base.py), to ensure compliance with other parts of the code. We provide a simple Multi-Layer Perceptron (MLP) architecture in [`src/architecture/mlp.py`](src/architecture/mlp.py) as an example to use this base class. The idea is that any new architecture must be a new class that inherit from `BaseFaultEstimatorNN`. To actually implement this new architecture, you must implement two methods:

- **init_architecture(self) -> None**: Setup the type and number of layers of the NN and store it as a [nn.Sequential](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sequential.html) object.
- **forward(self, x: Tensor) -> Tensor**: Compute the forward pass through the NN


## Training & Validation
Once your new architecture is ready and you have enough data, it's time for training and validation. You can setup this step using a configuration file (.yaml), as before, and run the [`src/train.py`](src/train.py) script:
```
python -m src.train -c <path_to_config>
``` 

This command will train your model directly on your machine. When dealing with large models or huge datasets, it is better to exploit the HPC. To launch a training job on the HPC, you can simply run

```
python -m scripts.launch_slurm -c <path_to_config> --submit
```

# Contributing to the repository
To add your own contributions to this repository, create your own branch based on the `origin/master` branch:
```
git fetch origin
git checkout -b <new-branch-name> origin/master
```
When applicable, the feature can be merged via a pull request. 

If you need to modify one of the submodules, do it the same way, by creating a new branch with your name, but this time based on the `origin/pinn-fault-diagnosis`, e.g.

```
cd submodules/PythonVehicleSimulator
git fetch origin
git checkout -b <new-branch-name> origin/pinn-fault-diagnosis
```
After creating a new branch for one of the submodules, you need to update the main repo (this one, i.e. pinn-fault-diagnosis) to point to the correct branch, e.g.
```
cd ../.. # change directory to /pinn-fault-diagnosis
git submodule set-branch --branch <new-branch-name> submodules/PythonVehicleSimulator
git submodule sync
git submodule update --init --remote
git add .gitmodules submodules/PythonVehicleSimulator
git commit -m "Update submodule to branch <new-branch-name>
```

## Testing - TODO

# Things that will save your time
- Everything related to the ReVolt's dynamics is implemented in the PythonVehicleSimulator submodule. You can find it in the ReVolt3Dynamics class in

    [`submodules/PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/revolt3.py`](https://github.com/stiefen1/PythonVehicleSimulator/blob/pinn-fault-diagnosis/src/python_vehicle_simulator/vehicles/revolt3.py). 
 
    This class inherits from a very important base class named `IDynamics` that you can find in

    [`submodules/PythonVehicleSimulator/src/python_vehicle_simulator/lib/dynamics.py`](https://github.com/stiefen1/PythonVehicleSimulator/blob/pinn-fault-diagnosis/src/python_vehicle_simulator/lib/dynamics.py). 
    
    And `IDynamics` is important because it gives you access to both continuous and discrete time dynamics, as well as jacobians w.r.t states, control inputs and fault parameters of the model. In the context of PINNs, this will be of huge interest.
    

# Ressources
## Fault Modelling
See [`ReVolt_Model_with_faults.pdf`](ReVolt_Model_with_faults.pdf).
## Papers


General about PINNs:
- [*Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561) (2017, **Note:** This is the paper introducing PINNs for the first time)

PINNs applied to maritime systems:
- [*Physics-Informed Neural Networks for Robust System Identification of Ship Roll Dynamics With Noise Resilience*](https://ieeexplore.ieee.org/document/10892333) (2025, **Note:** In this one they estimate paramters that are differentiable; this is not our case, which will probably be a good motivation for the proposed hybrid architecture)

Other methods for online parameter estimation:

Artificial Neural Networks:
- [*Study on blade fault diagnosis of free flying UAV based on multiparameter and deep learning method*](https://iopscience.iop.org/article/10.1088/2631-8695/acfbd9) (2023)

Parity Space:
- [*Fault estimation for a quadrotor unmanned aerial vehicle by integrating the parity space approach with recursive least squares*](https://journals.sagepub.com/doi/10.1177/0954410017691794) (2017)

Particle Filter, also called Sequential Monte Carlo (SMC) methods: 
- [*Particle Filter for Fault Diagnosis and Robust Navigation of Underwater Robot*](https://ieeexplore.ieee.org/document/6736066) (2014)

Moving Horizon Estimation:
- [*Moving Horizon Estimation of Faults in Renewable Microgrids*](https://www.sciencedirect.com/science/article/pii/S240589631930165X?pes=vor&utm_source=clarivate&getft_integrator=clarivate) (2019)

## Videos
- Full course on physics-informed ML (Steve Brunton, UW): https://www.youtube.com/watch?v=JoFW2uSd3Uo&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa
- Introduction to PINNs (Ben Moseley, ETH) : https://youtu.be/D-F7BYRhAkQ?si=ZPk7hf2VSyl3BrJe

