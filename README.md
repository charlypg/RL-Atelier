# RL-Atelier
A lightweight reinforcement learning library for exploring, learning, and testing new ideas.

*RL-Atelier* aims to remain as simple, open, and easy to extend as possible, making it well suited for crafting new ideas, learning by doing and teaching.

## Installation

### Conda environment (optional but recommended)

Create conda environment:

```bash
conda create --name rl_atelier python=3.10
```
 
Activate conda environment:

```bash
conda activate rl_atelier
```

### Clone repository and install requirements

```bash
git clone https://github.com/charlypg/RL-Atelier.git
cd RL-Atelier
pip install -r requirements.txt
```

### Local pip install (optional)

For using the package as-is:

```bash
pip install -e .
```

## Training an RL agent

To train an agent, run ```train.py``` with a base configuration file that defines all hyperparameters. You can also provide an optional variant configuration to override specific values and compare performance.

```python train.py +base=conf/lunar_lander/base.yaml +variant=conf/lunar_lander/variants/buffer_1e6.yaml```

## Credits

See CREDITS.md for more information.

This project reuses buffers and samplers components from the following open-source repository:

- xpag: https://github.com/perrin-isir/xpag (commit fad4d9cf77053cf29b42f322091bdf182012a501)
  Licensed under the BSD 3-Clause License. Portions of this codebase were adapted for this project.

