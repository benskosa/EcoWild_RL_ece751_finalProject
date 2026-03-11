# EcoWild --- RL-Based Wildfire Detection (Inference Module)

This directory contains the inference and environment implementation for
the EcoWild reinforcement learning framework.

It includes the runtime policy evaluation logic, the custom wildfire
environment, and the full energy-aware system modeling used to simulate
battery-constrained wildfire detection in remote sensor deployments.

------------------------------------------------------------------------

## Overview

EcoWild formulates adaptive wildfire monitoring as a sequential
decision-making problem under strict energy constraints.

During inference, a trained TD3 policy dynamically adjusts the image
sampling interval based on:

-   Current battery level
-   Solar energy harvesting
-   Machine learning detection characteristics (TP / FP rates)
-   Communication energy constraints
-   Reserved battery thresholds

The objective is to minimize detection latency while preserving
long-term energy sustainability.

------------------------------------------------------------------------

## Requirements

-   Python 3.8+
-   PyTorch
-   Stable-Baselines3
-   Gymnasium
-   NumPy
-   Pandas
-   Matplotlib
-   Shimmy (required for SB3 + Gym compatibility)

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Dataset

The test dataset `grouped_weather_data_with_solar_energy.csv`
is not included due to size constraints.

Please download it from: https://uwprod-my.sharepoint.com/:x:/g/personal/nyildirim_wisc_edu/IQBCAUzZMYE-SI-KK7-mU5cWAf8gtLMX-ENsK9bLF0sMiSs?e=nZwqKk 

The image dataset is available on the cloud. You can try downloading it from here:
https://www.dropbox.com/scl/fi/3q2sa0cko0mxlgqmz3i5l/dataset_grouped.tar?rlkey=luwrweslqbi1fc9aacswgfcns&e=1&st=9no1ahaw&dl=0

------------------------------------------------------------------------

## Energy-Aware System Modeling

The environment models:

-   Camera capture energy
-   ML inference energy
-   LoRa communication energy
-   Standby power components
-   Battery leakage
-   Solar harvesting (with configurable efficiency loss)
-   Reserved communication energy constraints

Battery state is updated at minute-level granularity to ensure realistic
energy depletion and recovery behavior.

------------------------------------------------------------------------

## Directory Structure

    inference/
    ├── inference_main.py
    ├── wildfire_env.py
    ├── data_utils.py
    ├── config_setup_*.json
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Running Inference

1.  Ensure a trained TD3 model checkpoint is available.
2.  Configure experiment parameters in the JSON configuration file.
3.  Run:

```bash
python inference_main.py config/config_setup_0.90036452TP_0.58399005FP_1dayReservedEnergy.json
```

Outputs include:

-   Detection time statistics
-   Battery trajectories
-   Sampling interval behavior
-   Reward metrics

Results are stored inside the `Inference/` directory.

------------------------------------------------------------------------

## Reference

This implementation is based on:

Nuriye Yildirim, Mengqi Cao, Minwoo Yun, Jaehyun Park, and Umit Y.
Ogras,\
"EcoWild: Reinforcement Learning for Energy-Aware Wildfire Detection in
Remote Environments,"\
Sensors, 25(19), 6011, 2025.

Note: This repository contains an actively developed research
implementation. Certain components --- including reward formulation,
inference configuration, energy modeling refinements, and evaluation
setup --- may differ from the exact experimental configuration described
in the published paper.

------------------------------------------------------------------------

## Research Use

This code is intended for research and academic experimentation. If you
use this work in your research, please cite the referenced publication.
