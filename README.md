# Project on Reinforcement Learning (Course project MLDL 2025 - POLITO)

Implementation of "Project 4: Reinforcement Learning" for the Machine Learning and Deep Learning (MLDL) 2025 course at Polytechnic of Turin. Official assignment at [Google Doc](https://docs.google.com/document/d/16Fy0gUj-HKxweQaJf97b_lTeqM_9axJa4_SdqpP_FaE/edit?usp=sharing).

**Authors:**
- Francesco Vanella
- Gabriele Imperiale
- Luciano Selimaj

*Politecnico di Torino — MSc in Data Science and Engineering*

---

## Getting started

### 1. Local on Linux (recommended)

If you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

**Installation**
- (recommended) create a new conda environment, e.g. `conda create --name mldl pip=22 python=3.8 setuptools=65.5.0 wheel=0.38`
- Run `pip install -r requirements.txt`
- Install MuJoCo 2.1 and the Python Mujoco interface:
	- follow instructions here: https://github.com/openai/mujoco-py
	- see Troubleshooting section below for solving common installation issues.

Check your installation by launching `python test_random_policy.py`.


### 2. Local on Windows
As the latest version of `mujoco-py` is not compatible for Windows explicitly, you may:
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- (not recommended) Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) of `mujoco-py`.
- (not recommended) Stick to the Google Colab template (see below), which runs on the browser regardless of the operating system. This option, however, will not allow you to render the environment in an interactive window for debugging purposes.


### 3. Remotely on Google Colab

Alternatively, you may also complete the project on [Google Colab](https://colab.research.google.com/):

- Download `colab_starting_code.ipynb` from the root of this repo.
- Load it on [Google Colab](https://colab.research.google.com/) and follow the instructions in the notebook to run the experiments.

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once (e.g. using a Linux system), to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.



## Troubleshooting
- General installation guide and troubleshooting: [Here](https://docs.google.com/document/d/1j5_FzsOpGflBYgNwW9ez5dh3BGcLUj4a/edit?usp=sharing&ouid=118210130204683507526&rtpof=true&sd=true)
- If having trouble while installing mujoco-py, see [#627](https://github.com/openai/mujoco-py/issues/627) to install all dependencies through conda.
- If installation goes wrong due to gym==0.21 as `error in gym setup command: 'extras_require'`, see https://github.com/openai/gym/issues/3176. There is a problem with the version of setuptools.
- If you get a `cannot find -lGL` error when importing mujoco_py for the first time, then have a look at my solution in [#763](https://github.com/openai/mujoco-py/issues/763#issuecomment-1519090452)
- If you get a `fatal error: GL/osmesa.h: No such file or directory` error, make sure you export the CPATH variable as mentioned in mujoco-py[#627](https://github.com/openai/mujoco-py/issues/627)
- If you get a `Cannot assign type 'void (const char *) except * nogil' to 'void`, then run `pip install "cython<3"` (see issue [#773](https://github.com/openai/mujoco-py/issues/773))

---

## Project Implementation and Experiments

This repository extends the official MLDL 2025 Project 4 template by implementing and evaluating multiple reinforcement learning algorithms and domain randomization strategies for sim-to-real transfer on the MuJoCo Hopper environment.

The objective of the project is to analyze how different domain randomization techniques affect **robustness**, **stability**, and **generalization** when transferring policies from a simulated **source** environment to a **target** environment with different physical parameters.

The experimental comparison focuses on three settings:
- **No Domain Randomization (No-DR)**: fixed physical parameters
- **Uniform Domain Randomization (UDR)**: uniform sampling of selected link masses
- **Adversarial Beta Curriculum (AdvBeta)**: failure-driven curriculum learning over physical parameters

All experiments are performed using multiple random seeds to ensure fair and reproducible results.

---

## Environment Design (Source and Target Domains)

A custom version of the MuJoCo Hopper environment is implemented to explicitly model sim-to-real mismatch.
- The **source domain** introduces a controlled modeling bias by modifying the torso mass.
- The **target domain** represents the nominal physical parameters and is used for transfer evaluation.

The training behavior is controlled via a single `dr_method` parameter:
- `none`: fixed physical parameters (No-DR)
- `udr`: uniform mass randomization at each episode reset
- `adv_beta`: curriculum-based sampling using the Adversarial Beta method (SAC only)

---

## Domain Randomization Strategies

Three domain randomization strategies are considered:

### 1. No Domain Randomization (No-DR)
Physical parameters remain fixed throughout training and evaluation.

### 2. Uniform Domain Randomization (UDR)
Thigh, leg, and foot masses are sampled uniformly within **±30%** of their nominal values at each episode reset.

### 3. Adversarial Beta Curriculum (AdvBeta)
A curriculum is learned online by fitting Beta distributions over parameters that lead to low returns. Sampling is performed by mixing uniform exploration with curriculum-driven sampling.

The `dr_method` parameter controls the strategy:

| Strategy | `dr_method` |
| :--- | :--- |
| **No-DR** | `none` |
| **UDR** | `udr` |
| **AdvBeta** | `adv_beta` |

---

## Environment Configuration
Before running any scripts, set the required environment variables:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco210/bin
export PYTHONPATH=/path/to/mldl_rl_project_25
export PYTHONUNBUFFERED=1
```
Replace `/home/<user>` and `/path/to/mldl_rl_project_25` with your actual paths. `PYTHONPATH` is required so that all scripts can import `env.custom_hopper` regardless of which subdirectory they live in.

---

## Installation Check
After installing all dependencies and MuJoCo (see sections above), verify the setup by running:
```bash
python test_random_policy.py
```
If the script runs correctly, the environment is properly installed.

---

## Actor–Critic and REINFORCE

### Training (Single Run)

**Train an Actor–Critic agent on the source domain:**
```bash
python ActorCritic/train_actor_critic.py \
  --env-id CustomHopper-source-v0 \
  --n-episodes 50000 \
  --actor-critic \
  --baseline 20.0 \
  --seed 42 \
  --device cuda
```
**Train a REINFORCE agent without baseline:**
```bash
python ActorCritic/train_actor_critic.py \
  --env-id CustomHopper-source-v0 \
  --n-episodes 50000 \
  --baseline 0.0 \
  --seed 42 \
  --device cuda

```

### Training with Weights & Biases Sweep
Log in to Weights & Biases:
```bash
wandb login
```
Launch the sweep:
```bash
python ActorCritic/sweep_runner.py
```
The sweep evaluates different combinations of:
* Random seeds (e.g., 42, 43, 44)
* Actor–Critic vs REINFORCE
* Baseline values
* Domain randomization strategy (`none`, `udr`)

### Evaluation (Actor–Critic / REINFORCE)
Evaluate a trained model on the target domain (default):
```bash
python ActorCritic/test.py \
  --model AC_seed42_ac_none.mdl \
  --episodes 10 \
  --device cpu
```
To run a sanity check on the source domain, add `--env-id CustomHopper-source-v0`.

| Flag | Description |
| :--- | :--- |
| `--model <str>` | Model filename (loaded from `ActorCritic/saved_models/`) |
| `--env-id <str>` | Environment id (default: `CustomHopper-target-v0`) |
| `--episodes <int>` | Number of test episodes |
| `--device <str>` | Device (`cpu` or `cuda`) |
| `--render` | Render the simulator |

---

## Soft Actor–Critic (SAC)

### Training (Single Run)
Train SAC on the source domain:
```bash
python Sac/train_sac.py \
  --env-id CustomHopper-source-v0 \
  --seed 42 \
  --dr-method udr
```

The `--dr-method` flag accepts `none`, `udr`, or `adv_beta`. SAC hyperparameters are loaded from `Sac/sweep_config_sac.yaml`. When using `adv_beta`, the curriculum hyperparameters, such as buffer size, warmup episodes, tau, etc., are loaded from `Sac/adv_beta_config.yaml`.

### Training with Weights & Biases Sweep (Recommended)
Launch the SAC sweep:
```bash
python Sac/sweep_runner_sac.py
```

The sweep explores:
* Multiple random seeds (e.g., 42, 43, 44)
* Domain randomization strategy (`none`, `udr`, `adv_beta`)
* Fixed SAC hyperparameters for fair comparison

### SAC Evaluation
Evaluate a trained SAC policy on the target domain (default):
```bash
python Sac/test_sb3.py \
  --model sac_final_seed42_id_SAC_seed42_udr \
  --episodes 10 \
  --device cpu
```
To run a sanity check on the source domain, add `--env-id CustomHopper-source-v0`.

| Flag | Description |
| :--- | :--- |
| `--model <str>` | Model filename (loaded from `Sac/saved_models/`) |
| `--env-id <str>` | Environment id (default: `CustomHopper-target-v0`) |
| `--episodes <int>` | Number of test episodes |
| `--device <str>` | Device (`cpu` or `cuda`) |
| `--render` | Render the simulator |

---

## Logging and Monitoring

All experiments are logged using **Weights & Biases** and **TensorBoard**.

Logged metrics include:
* Episode return and episode length
* Sampled environment parameters
* Curriculum statistics (alphas and betas)

For Adversarial Beta experiments, the evolution of the sampling distribution is saved as a PDF file (`evolution_<run_name>.pdf`) to enable qualitative analysis of the curriculum behavior.

---

## Experimental Protocol and Reproducibility

To ensure fair and reproducible comparisons:
1.  **Identical random seeds** are used across all strategies.
2.  **Hyperparameters** are kept fixed across methods.
3.  **Training** is always performed on the source domain.
4.  **Evaluation** is performed on both source and target domains.

Performance is analyzed in terms of average return, variability across episodes, and robustness under domain shift.

---

# Copyright & License

Copyright (c) 2026 Francesco Vanella, Gabriele Imperiale, Luciano Selimaj

The code and accompanying materials are released under the [MIT License](LICENSE).
