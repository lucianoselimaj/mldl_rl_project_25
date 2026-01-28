# Project on Reinforcement Learning (Course project MLDL 2025 - POLITO)
### Teaching assistants: Andrea Protopapa and Davide Buoso

Starting code for "Project 4: Reinforcement Learning" course project of MLDL 2025 at Polytechnic of Turin. Official assignment at [Google Doc](https://docs.google.com/document/d/16Fy0gUj-HKxweQaJf97b_lTeqM_9axJa4_SdqpP_FaE/edit?usp=sharing).


## Getting started

Before starting to implement your own code, make sure to:
1. read and study the material provided (see Section 1 in the assignment)
2. read the documentation of the main packages you will be using ([mujoco-py](https://github.com/openai/mujoco-py), [Gym](https://github.com/openai/gym), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local on Linux (recommended)

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

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

- Download the files contained in the `colab_template` folder in this repo.
- Load the `.ipynb` files on [https://colab.research.google.com/](colab) and follow the instructions on each script to run the experiments.

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once (e.g. using a Linux system), to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.



## Troubleshooting
- General installation guide and troubleshooting: [Here](https://docs.google.com/document/d/1j5_FzsOpGflBYgNwW9ez5dh3BGcLUj4a/edit?usp=sharing&ouid=118210130204683507526&rtpof=true&sd=true)
- If having trouble while installing mujoco-py, see [#627](https://github.com/openai/mujoco-py/issues/627) to install all dependencies through conda.
- If installation goes wrong due to gym==0.21 as `error in gym setup command: 'extras_require'`, see https://github.com/openai/gym/issues/3176. There is a problem with the version of setuptools.
- if you get a `cannot find -lGL` error when importing mujoco_py for the first time, then have a look at my solution in [#763](https://github.com/openai/mujoco-py/issues/763#issuecomment-1519090452)
- if you get a `fatal error: GL/osmesa.h: No such file or directory` error, make sure you export the CPATH variable as mentioned in mujoco-py[#627](https://github.com/openai/mujoco-py/issues/627)
- if you get a `Cannot assign type 'void (const char *) except * nogil' to 'void`, then run `pip install "cython<3"` (see issue [#773](https://github.com/openai/mujoco-py/issues/773))

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

The environment behavior is controlled dynamically at runtime using two flags:
- `randomize_on_reset`: enables mass randomization at each episode reset.
- `use_ext`: enables curriculum-based sampling using the Adversarial Beta method.

---

## Domain Randomization Strategies

Three domain randomization strategies are considered:

### 1. No Domain Randomization (No-DR)
Physical parameters remain fixed throughout training and evaluation.

### 2. Uniform Domain Randomization (UDR)
Thigh, leg, and foot masses are sampled uniformly within **±30%** of their nominal values at each episode reset.

### 3. Adversarial Beta Curriculum (AdvBeta)
A curriculum is learned online by fitting Beta distributions over parameters that lead to low returns. Sampling is performed by mixing uniform exploration with curriculum-driven sampling.

The following flag combinations are used:

| Strategy | `use_ext` | `randomize_on_reset` |
| :--- | :--- | :--- |
| **No-DR** | `false` | `false` |
| **UDR** | `false` | `true` |
| **AdvBeta** | `true` | `true` |

> **Note:** The combination `use_ext=true` and `randomize_on_reset=false` is intentionally avoided, as the curriculum would never be applied.

---
## Environment Configuration
Before running any scripts, you must set the required environment variables for MuJoCo and Python buffering.

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
python sweep_runner_ac.py
```
The sweep evaluates different combinations of:
* Random seeds (e.g., 42, 43, 44)
* Actor–Critic vs REINFORCE
* Baseline values

### Evaluation (Actor–Critic / REINFORCE)
Evaluate a trained model:
```bash
python test_actor_critic.py \
  --model AC_Seed42.mdl \
  --episodes 10 \
  --device cpu
```
The evaluation can be performed on either the source or target domain depending on the environment selected in the test script.

---

## Soft Actor-Critic (SAC)

### Training (Single Run)
Train SAC on the source domain:
```bash
python Sac/train_sac.py \
  --env-id CustomHopper-source-v0 \
  --seed 42
```

In this project, domain randomization and curriculum behavior are controlled internally through configuration flags passed to the environment.

### Training with Weights & Biases Sweep (Recommended)
Launch the SAC sweep:
```bash
python sweep_runner_sac.py
```

The sweep explores:
* Multiple random seeds (e.g., 42, 43, 44)
* No-DR, UDR, and AdvBeta configurations
* Fixed SAC hyperparameters for fair comparison

### SAC Evaluation (Source → Source and Source → Target)
Evaluate a trained SAC policy:
```bash
python test_sac.py \
  --model sac_final_seed42_id_SAC_seed42_UDR \
  --episodes 10 \
  --device cpu
```
#### Optional Evaluation Flags

| Flag | Description |
| :--- | :--- |
| `--randomize-on-reset` | Enable domain randomization during evaluation |
| `--use-ext` | Enable Adversarial Beta sampling |
| `--seed <int>` | Fix environment randomness |
| `--render` | Render the simulator |

#### Examples

**Evaluate on target domain with No Domain Randomization:**
```bash
python test_sac.py \
  --model sac_final_seed42_id_SAC_seed42_NoDR \
  --episodes 10 \
  --device cpu
```

**Evaluate on target domain with Uniform Domain Randomization:**
```bash
python test_sac.py \
  --model sac_final_seed42_id_SAC_seed42_UDR \
  --episodes 10 \
  --device cpu \
  --randomize-on-reset \
  --seed 42
```

**Evaluate on target domain with Adversarial Beta Curriculum:**
```bash
python test_sac.py \
  --model sac_final_seed42_id_SAC_seed42_AdvBeta \
  --episodes 10 \
  --device cpu \
  --randomize-on-reset \
  --use-ext \
  --seed 42
```

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
