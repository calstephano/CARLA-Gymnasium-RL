# Reinforcment Learning (RL) in CARLA
![Status](https://img.shields.io/badge/Status-Thesis%20Milestone-blueviolet) ![Language](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)   ![Platform](https://img.shields.io/badge/Platform-Ubuntu%20%7C%20Windows-lightgrey?logo=linux) ![CARLA](https://img.shields.io/badge/CARLA-0.9.14%2B-orange)  ![Gymnasium](https://img.shields.io/badge/Made%20with-Gymnasium-brightgreen) ![RL Framework](https://img.shields.io/badge/Stable--Baselines3-v1.7-green)  ![Visualization](https://img.shields.io/badge/Powered%20by-TensorBoard-orange?logo=tensorflow) ![Lab](https://img.shields.io/badge/Associated%20Lab-AIEA-red)

## Overview 
This repository features **a custom CARLA Gymnasium environment**. Built upon the foundation of [this custom gym environment](https://github.com/cjy1992/gym-carla.git), it has been upgraded to leverage Gymnasium for improved compatibility and functionality, and incorporates Stable-Baselines3 (SB3) for advanced reinforcement learning capabilities. As well, the code has been heavily modularized and improved with additional features.

_This version of the project was created for my undergraduate thesis on reward design in reinforcement learning for autonomous vehicles (available upon request.)_ 

A special thanks to my thesis supervisor, Professor Leilani Gilpin, and my AIEA Lab peers, Vik Dhillon and Dominick Rangel, for their support and guidance throughout this research.

## Features  
- **Scalable observation space:** Supports state vectors and multi-camera sensor inputs.  
- **Reward function design:** An 8-component reward function balancing safety (e.g., lane adherence, collision avoidance) and efficiency (e.g., waypoint completion). Currently only uses state vector observations.
- **Visualization:** Four camera sensors incorporated for enhanced visualization, rendered through Pygame. Added upon the existing bird's-eye feature from the original gym environment.
- **Training & monitoring:** RL agents trained using SB3's Deep Q-Networks (DQN) and multi-input policy, with performance tracked using TensorBoard.  

## Setup
### 1. Set up the platform and install CARLA.
This project can be run locally on Windows or through a Linux-based GUI on the Nautilus cluster to manage large-scale simulations. [Follow this tutorial](https://github.com/calstephano/Nautilus-GUI-AV-Setup) for cluster setup. Once successful, follow the CARLA installation guide to install CARLA via the GUI's terminal.

### 2. Upon up a second terminal, clone the repository and download dependencies.
```
git clone https://github.com/calstephano/CARLA-Gymnasium-RL.git
cd CARLA-Gymnasium-RL
pip3 install -r requirements.txt
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:<path to CARLA>/PythonAPI/carla/dist/carla-<version here>-py3
```

### 3. Return to the first terminal and host CARLA.
The first terminal should already be at CARLA's installation directory and if not, navigate there. Host CARLA through the following command:

`./CarlaUE4.sh -carla-rpc-port=2000 -norelativemousemode`

### 4. Return to the second terminal and run the project.
`python3 run.py`

## Viewing Data
TensorBoard is a visualization tool used to monitor training metrics, such as rewards and losses. To launch TensorBoard, open a new terminal and run the following command in the main directory:

`tensorboard --logdir ./logs --port 6006`

Open your browser and go to:

`http://localhost:6006`
