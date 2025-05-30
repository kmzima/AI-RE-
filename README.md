# AI-RE-

Reproducibility Study: Utilising Uncertainty for Efficient Learning of Likely-Admissible Aeuristics

This repository contains our implementation for reproducing the experiments from:
**"Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics"**  
*Ofir Marom, Benjamin Rosman*  
_ICAPS 2020: Proceedings of the International Conference on Automated Planning and Scheduling_

## Overview

This study reproduces the uncertainty-based framework for learning likely-admissible heuristics using the following components:

- Bayesian Neural Network (WUNN)
- Standard Feedforward NN with aleatoric uncertainty
- GenerateTaskPrac (uncertainty-based task generation)
- LearnHeuristicPrac (adaptive heuristic training)
- Full reproduction of 15-puzzle domain experiments

## Reproduction Goals

- Suboptimality Performance: Reproduce 2–6% suboptimality for various confidence levels
- Training Efficiency: Validate that uncertainty-guided task generation (GTP) outperforms fixed increments

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- Matplotlib

### Setup

git clone https://github.com/[kmzima]/AI-RE-.git

cd uncertainty-heuristic-reproduction

pip install torch numpy scipy matplotlib

### For quick test run use
python run_experiments.py --quick
- approximately 30 mins runtime

### For Full Reproduction run use
python run_experiments.py
- total runtime is approximately 672 mins

## Repository Structure
├── domain.py                  # 15-puzzle & IDA* search
├── StateRepresentation.py     # One-hot & feature encodings
├── NeuralNetworkComponents.py # FFNN, WUNN, uncertainty modeling
├── run_experiments.py         # Experiment entry point
├── hyperparameter_config.py   # Hyperparameter definitions
├── paper_experiments.py       # Experimental protocol
├── results/                   # Experiment output
├── plots/                     # Output visualizations
└── README.md

## Expected Results that we produced
- suboptimality_results.json
- efficiency_results.json
- training_progress.json

## Visual Plots
- Suboptimality vs Confidence
- Training progress
- Effieciency comparison (GTP vs Fixed)
- Search effort analysis

## Console logs 
- Iteration summary
- solve rates and timeouts
- Confidence level adaptation
- Reproducibility insights

## Hyperparameters
Hyperparameters were all set as in the original paper and can be accessed and modified in the hyperparameter_config.py:
- hidden_neurons = 20
- dropout_rate = 0.025
- learning_rate_ffnn = 0.001
- learning_rate_wunn = 0.01
- prior_mu = 0.0
- prior_sigma_squared = 10.0
- initial_beta = 0.05
- epsilon = 1.0
- num_iterations = 50
- num_tasks_per_iter = 10
- timeout_seconds = 60
- initial_confidence = 0.99

## System requirements 
- CPU: AMDA RYZEN 7 or similar
- RAM 16GB
- GPU: likely not required
- Runtime: approximately 11.19 hours for full reproduction of experiments. 

## Key results and findings
- Lower training solve rates
- Suboptimality difficulty due to limited task success
- Effeciency is partially reproduced.
Overall our findings contribute to understanding how reproduced implementations can vary from the original design.






