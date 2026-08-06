# Physics-Informed Neural Networks for Fault Diagnosis
This repo provides basic features to setup, train and evaluate physics-informed neural networks (PINNs) for fault diagnosis of the ReVolt vessel. It consists of the following modules:
- **architecture:** Base class + example to implement NN-based fault estimator
- **diagnosis:** Base class for fault diagnosis module + examples of other fault estimation approaches, runnable as part of the rest of the simulation framework
- **eval:** Evaluation of the model on the test set
- **excitation:** A collection of common excitation signals to be used as actuator commands when generating a new dataset
- **tune:** (under development) Implementation of optuna for hyperparameters tuning
- **utils:** A collection of functions that might be useful across different parts of the code
- **vessel:** GNC architecture for the ReVolt vessel. Note that simulations are handled by the PythonVehicleSimulator submodule.



# TODOs
## Implement and train a PINN for actuators fault estimation


# Installation

# Setting up NTNU's High Performance Computing (HPC) unit
## Request access
## Connect
## Launch your jobs

# Usage
Most of the features rely heavily on high-level configuration files (.yaml).
## Generating Dataset
This feature is mainly done through the FaultIdentificationDatasetGenerator class located in ```src/dataset/generator.py```. The most important entries of the associated configuration file are:
- **episodes** : Describes how large we want the dataset to be
- **dataset** : Where to save the resulting data
- **env** : What are the environmental disturbances acting on the vessel throughout simulations
- **vessel** : What is the vessel's GNC architecture, what excitation signals do we use and what kind of faults can occur

To generate a dataset based on a given configuration file, simply run:
```
python -m src.dataset.generator -c <path_to_config>
```
Where <path_to_config> must be a path starting from the base folder of this repository to a valid .yaml file (e.g. configs/dataset.yaml).

### Side note on dataset generation
In system identification (sysid), an important part of the job is to find control commands that make the identification of unknown parameters easier. In non-linear systems, applying different control commands can result in different observability conditions for the parameters to be estimated. The direct consequence is that the same fault can be more challenging (or even impossible) to detect in some situations. When generating a fault identification dataset, it's not only important to cover a wide range of states to avoid out-of-distribution (OOD) samples, but also to make sure we train our system in both easy and challenging configurations. 

### Parameterizing auxiliary signals for dataset generation
In your configuration file (e.g. [`configs/dataset.yaml`](configs/dataset.yaml)), you can manage whether you want your dataset to be generated solely using the NMPC controller, an auxiliary excitation signal, or both together. 

To enable an auxiliary excitation signal, simply set vessel.auxiliary_excitation.enabled to **true**. Then, you can provide the desired excitation signal as an explicit string for both the azimuth angle and thruster speed of each thruster (port and starboard). For example

`azimuth: 'Sinus(frequency=0.1, amplitude=-3.14159, time_delay_sec=0.0, y_offset=0.0) + GaussianNoise(amplitude=0.5)'`

The complete list of available signals is available in [`src/excitation/signals.py`](src/excitation/signals.py).

To enable control commands from the NMPC controller, simply set vessel.control.enabled to **true**. If an auxiliary signals was requested as well, the resulting control commands will be the sum of both. This could be useful e.g. to mix NMPC control commands with gaussian noise to obtain a more diverse dataset of excitation signals. 


## Training & Validation
## Testing - TODO

# Things that will save your time
- Everything related to the ReVolt's dynamics is implemented in the PythonVehicleSimulator submodule. You can find it in the ReVolt3Dynamics class in

    [`submodules/PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/revolt3.py`](/submodules/PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/revolt3.py). 
 
    This class inherits from a very important base class named `IDynamics` that you can find in

    [`submodules/PythonVehicleSimulator/src/python_vehicle_simulator/lib/dynamics.py`](/submodules/PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/revolt3.py). 
    
    And `IDynamics` is important because it gives you access to both continuous and discrete time dynamics, as well as jacobians w.r.t states, control inputs and fault parameters of the model. In the context of PINNs, this will be of huge interest.
    


# Ressources
## Fault Modelling
See [`ReVolt_Model_with_faults.pdf`](ReVolt_Model_with_faults.pdf).
## Papers
## Videos
- Full course on physics-informed ML (Steve Brunton, UW): https://www.youtube.com/watch?v=JoFW2uSd3Uo&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa
- Introduction to PINNs (Ben Moseley, ETH) : https://youtu.be/D-F7BYRhAkQ?si=ZPk7hf2VSyl3BrJe

