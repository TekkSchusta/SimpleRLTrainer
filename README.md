# TekksTrainer

## Overview

TekksTrainer is a Rocket League self-play training system using RocketSim and PPO for 1v1.

## Installation

- Install Python 3.8 or newer.
- Create and activate a virtual environment. This is required.
- Install dependencies using pip from requirements.txt inside the virtual environment.
- Verify RocketSim and RLBot import in Python.

## Usage

- Run training with python run_training.py.
- Open live viewer with python viewer/live_viewer.py.
- Evaluate final models with python eval/evaluate.py.
- Copy a final model into rlbot/TekksTrainer/models/latest_ppo.zip to use in RLBot.
- Ensure RS_COLLISION_MESHES points to collision_meshes or leave default.

## Folder Structure

- outputs/checkpoints stores training checkpoints named with _ppo.zip.
- outputs/final stores final models.
- outputs/eval stores evaluation results and metrics.
- rlbot/TekksTrainer/models stores models for the RLBot.

## Training Configuration

- Edit config.json to change training hyperparameters and environment settings.
- total_steps sets training duration.
- update_interval controls PPO rollout size.
- batch_size affects update speed and stability.
- learning_rate controls optimizer speed.
- checkpoint_interval defines how often checkpoints are saved.
- opponent_refresh_interval controls self-play opponent updates.
- viewer_update_interval controls live dashboard refresh.

## Examples

- Self-play training and live monitoring.
- Evaluation across strategies for consistency.
- RLBot gameplay using the trained model.

## Troubleshooting

- rocketsim import error: Check Python version and install RocketSim in the virtual environment.
- rlbot import error: Install rlbot and ensure Rocket League is launched through RLBot tools.
- No metrics: Confirm outputs folder permissions and that training is running.
- No models loading in RLBot: Ensure latest_ppo.zip is present under models.
- Slow training: Reduce batch_size or increase learning_rate cautiously.
- Not in venv: Activate a virtual environment before running any scripts.

## Glossary

- Self-play: Training by playing against previous versions of the agent.
- PPO: Proximal Policy Optimization.
- Checkpoint: Saved snapshot of model parameters during training.
 - Observation: Numerical representation of game state for the agent.
 - Action: Control values the agent outputs.

## Author

TekkSchuster
