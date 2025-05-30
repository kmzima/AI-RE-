import torch
import torch.nn as nn
from typing import Dict, Any

class PaperHyperparameters:
    #Exact hyperparameters from the paper
    
    # Neural Network Architecture (Section 5)
    ARCHITECTURE = {
        'hidden_neurons_15_puzzle': 20,
        'hidden_neurons_other_domains': 8,  # For 24-puzzle, 24-pancake, 15-blocksworld
        'dropout_rate': 0.025,  # 2.5% for 15-puzzle only
        'activation_hidden': 'relu',
        'activation_output': 'linear',
        'weight_init': 'he_normal',
        'bias_init': 'zeros'
    }
    
    # Training Parameters (Section 5)
    TRAINING = {
        'learning_rate_wunn': 0.01,
        'learning_rate_ffnn': 0.001,
        'optimizer': 'adam',
        'monte_carlo_samples_training': 5,  # S for WUNN training
        'monte_carlo_samples_uncertainty': 100,  # K for uncertainty estimation
        'train_iter_ffnn': 1000,
        'max_train_iter_wunn': 5000,
        'mini_batch_size_wunn': 100,
        'early_stopping_threshold': 0.5,  # Stop if training error < 0.5
    }
    
    # Bayesian Parameters (Section 4 & 5)
    BAYESIAN = {
        'prior_mean': 0.0,  # μ₀
        'prior_variance': 10.0,  # σ₀²
        'initial_beta': 0.05,  # β₀
        'final_beta': 0.00001,  # β_final
        'gamma_decay_factor': None,  # Calculated as (β_final/β₀)^(1/NumIter)
        'kl_threshold': 0.64,  # κ
        'uncertainty_threshold': 1.0,  # ε
        'local_reparameterization': True,  # Reduce variance in WUNN training
    }
    
    # Experimental Setup (Section 5)
    EXPERIMENT = {
        'num_iterations': 50,  # NumIter for 15-puzzle
        'num_iterations_complex': 75,  # For 24-puzzle, 24-pancake, 15-blocksworld
        'num_tasks_per_iter': 10,
        'num_tasks_per_iter_thresh': 6,
        'max_steps': 1000,  # MaxSteps for 15-puzzle
        'max_steps_complex': 5000,  # For complex domains
        'memory_buffer_max': 25000,
        'quantile_q': 0.95,
        'sampling_constant_c': 1.0,  # For uncertainty-based sampling
    }
    
    # Confidence Level Adaptation (Section 4)
    CONFIDENCE = {
        'initial_alpha': 0.99,  # α₀
        'alpha_decrement': 0.05,  # Δ
        'min_alpha': 0.5,
        'solve_rate_threshold': 0.6,  # percSolvedThresh
        'confidence_levels_test': [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05],
    }
    
    # Timeout Settings (Section 5)
    TIMEOUTS = {
        '15_puzzle_subopt': 60,  # seconds
        '15_puzzle_efficiency': 1,  # seconds
        'complex_domains': 300,  # 5 minutes
        'test_evaluation': 60,  # seconds per test task
    }
    
    # Domain-Specific Settings
    DOMAINS = {
        '15_puzzle': {
            'size': 16,
            'representation': 'efficient_2d',  # 16×2×4 bits instead of 16²
            'use_dropout': True,
            'l2_penalty': 0.1,
        },
        '24_puzzle': {
            'size': 25,
            'representation': 'features_pdb',
            'use_dropout': False,
            'pdb_partitions': [
                [1, 2, 5, 6, 7],
                [3, 4, 8, 9, 14],
                [10, 15, 16, 20, 21],
                [13, 18, 19, 23, 24],
                [11, 12, 17, 22]
            ]
        }
    }

class NetworkFactory:
    #Factory for creating networks with paper hyperparameters
    
    @staticmethod
    def create_standard_nn(input_size: int, domain: str = '15_puzzle') -> nn.Module:
        #Create StandardNN with paper hyperparameters
        if domain == '15_puzzle':
            hidden_size = PaperHyperparameters.ARCHITECTURE['hidden_neurons_15_puzzle']
            use_dropout = True
            dropout_rate = PaperHyperparameters.ARCHITECTURE['dropout_rate']
        else:
            hidden_size = PaperHyperparameters.ARCHITECTURE['hidden_neurons_other_domains']
            use_dropout = False
            dropout_rate = 0.0
        
        class PaperStandardNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
                self.fc2 = nn.Linear(hidden_size, 2)  # mean and variance
                
                # Paper initialization
                nn.init.kaiming_normal_(self.fc1.weight)  # He Normal
                nn.init.kaiming_normal_(self.fc2.weight)
                nn.init.zeros_(self.fc1.bias)
                nn.init.zeros_(self.fc2.bias)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                mean = x[..., 0]
                log_var = x[..., 1]
                variance = torch.log1p(torch.exp(log_var))  # Softplus
                
                return mean, variance
        
        return PaperStandardNN()
    
    @staticmethod
    def create_weight_uncertainty_nn(input_size: int, domain: str = '15_puzzle') -> nn.Module:
        #Create WeightUncertaintyNN with paper hyperparameters
        if domain == '15_puzzle':
            hidden_size = PaperHyperparameters.ARCHITECTURE['hidden_neurons_15_puzzle']
        else:
            hidden_size = PaperHyperparameters.ARCHITECTURE['hidden_neurons_other_domains']
        
        class PaperWUNN(nn.Module):
            def __init__(self):
                super().__init__()
                from NeuralNetworkComponents import BayesianLinearLayer
                
                prior_mean = PaperHyperparameters.BAYESIAN['prior_mean']
                prior_std = PaperHyperparameters.BAYESIAN['prior_variance'] ** 0.5
                
                self.fc1 = BayesianLinearLayer(input_size, hidden_size, prior_mean, prior_std)
                self.fc2 = BayesianLinearLayer(hidden_size, 1, prior_mean, prior_std)
                
                self.num_samples = PaperHyperparameters.TRAINING['monte_carlo_samples_uncertainty']
            
            def forward(self, x, sample=True):
                x = torch.relu(self.fc1(x, sample))
                x = self.fc2(x, sample)
                return x
            
            def kl_divergence(self):
                return self.fc1.kl_divergence() + self.fc2.kl_divergence()
        
        return PaperWUNN()

class OptimizerFactory:
    #Factory for creating optimizers with paper hyperparameters
    
    @staticmethod
    def create_ffnn_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        #Create optimizer for StandardNN
        return torch.optim.Adam(
            model.parameters(),
            lr=PaperHyperparameters.TRAINING['learning_rate_ffnn']
        )
    
    @staticmethod
    def create_wunn_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        #Create optimizer for WeightUncertaintyNN
        return torch.optim.Adam(
            model.parameters(),
            lr=PaperHyperparameters.TRAINING['learning_rate_wunn']
        )

class LossFactory:
    #Factory for loss functions used in the paper
    
    @staticmethod
    def heteroscedastic_loss(mean_pred, var_pred, target):
        #Loss function for aleatoric uncertainty (Equation 1)
        # L = (y - μ)²/(2σ²) + 0.5*log(σ²)
        loss = 0.5 * torch.log(var_pred) + 0.5 * (target - mean_pred)**2 / var_pred
        return loss.mean()
    
    @staticmethod
    def elbo_loss(mean_pred, target, kl_div, beta, batch_size):
        #ELBO loss for WUNN (Equation 2)
        # L = βKL[q(w)||p(w)] - E[log p(D|w)]
        likelihood = -0.5 * (target - mean_pred)**2  # Negative log likelihood
        likelihood_term = -likelihood.mean()
        kl_term = kl_div / batch_size
        
        return likelihood_term + beta * kl_term

class TrainingScheduler:
    #Handles training schedules and adaptive parameters
    
    def __init__(self, num_iterations: int):
        self.num_iterations = num_iterations
        self.beta_schedule = self._create_beta_schedule()
        self.confidence_scheduler = ConfidenceScheduler()
    
    def _create_beta_schedule(self):
        #Create beta decay schedule (Section 4)
        beta_0 = PaperHyperparameters.BAYESIAN['initial_beta']
        beta_final = PaperHyperparameters.BAYESIAN['final_beta']
        
        # γ^NumIter * β₀ = β_final
        gamma = (beta_final / beta_0) ** (1.0 / self.num_iterations)
        
        schedule = []
        beta = beta_0
        for i in range(self.num_iterations):
            schedule.append(beta)
            beta *= gamma
        
        return schedule
    
    def get_beta(self, iteration: int) -> float:
        #Get beta value for current iteration
        return self.beta_schedule[min(iteration, len(self.beta_schedule) - 1)]

class ConfidenceScheduler:
    #Handles adaptive confidence level adjustment
    
    def __init__(self):
        self.alpha = PaperHyperparameters.CONFIDENCE['initial_alpha']
        self.min_alpha = PaperHyperparameters.CONFIDENCE['min_alpha']
        self.decrement = PaperHyperparameters.CONFIDENCE['alpha_decrement']
        self.threshold = PaperHyperparameters.CONFIDENCE['solve_rate_threshold']
        self.update_beta = True
    
    def update(self, solve_rate: float) -> bool:
        #Update confidence level based on solve rate
        if solve_rate < self.threshold:
            self.update_beta = False
            
            if self.alpha < 0.5:
                old_alpha = self.alpha
                self.alpha = min(self.alpha + self.decrement, self.min_alpha)
                return old_alpha != self.alpha
        else:
            self.update_beta = True
        
        return False

class ExperimentConfig:
    #Configuration for specific experiments
    
    @staticmethod
    def get_suboptimality_config() -> Dict[str, Any]:
        #Configuration for suboptimality experiment (Table 1)
        return {
            'num_iterations': PaperHyperparameters.EXPERIMENT['num_iterations'],
            'num_tasks_per_iter': PaperHyperparameters.EXPERIMENT['num_tasks_per_iter'],
            'timeout_ms': PaperHyperparameters.TIMEOUTS['15_puzzle_subopt'] * 1000,
            'max_steps': PaperHyperparameters.EXPERIMENT['max_steps'],
            'confidence_levels': PaperHyperparameters.CONFIDENCE['confidence_levels_test'],
            'num_runs': 10,  # Paper uses 10 independent runs
            'benchmark_size': 100,  # 100 test instances
        }
    
    @staticmethod
    def get_efficiency_config() -> Dict[str, Any]:
        #Configuration for efficiency experiment (Table 2)
        return {
            'num_iterations': 20,  # Reduced for efficiency test
            'num_tasks_per_iter': PaperHyperparameters.EXPERIMENT['num_tasks_per_iter'],
            'timeout_ms': PaperHyperparameters.TIMEOUTS['15_puzzle_efficiency'] * 1000,
            'length_increments': [1, 2, 4, 6, 8, 10],
            'test_tasks': 100,  # Tasks of increasing difficulty
            'test_timeout_ms': 60 * 1000,  # 60s for test evaluation
        }

class BenchmarkDatasets:
    #Standard benchmark datasets from the paper
    
    @staticmethod
    def get_korf_15_puzzle_tasks():
        #Get the standard 100 15-puzzle benchmark tasks (Korf 1985)
        # This would normally load from a file, but we'll generate deterministic ones
        import random
        random.seed(42)  # For reproducibility
        
        tasks = []
        for i in range(100):
            # Create a solvable scrambled puzzle
            puzzle = list(range(16))
            
            # Perform valid swaps to ensure solvability
            for _ in range(50 + i):  # Increasing difficulty
                # Swap adjacent tiles or tiles in same row/column
                pos1 = random.randint(0, 15)
                
                # Find valid swap positions
                valid_positions = []
                row, col = pos1 // 4, pos1 % 4
                
                # Adjacent positions
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 4 and 0 <= new_col < 4:
                        valid_positions.append(new_row * 4 + new_col)
                
                if valid_positions:
                    pos2 = random.choice(valid_positions)
                    puzzle[pos1], puzzle[pos2] = puzzle[pos2], puzzle[pos1]
            
            tasks.append(puzzle)
        
        return tasks

# Integration with existing code
class PaperExperimentRunner:
    #Main runner that integrates all paper hyperparameters
    
    def __init__(self, domain='15_puzzle'):
        self.domain = domain
        self.config = ExperimentConfig()
        self.scheduler = TrainingScheduler(
            PaperHyperparameters.EXPERIMENT['num_iterations']
        )
        
    def create_networks(self, input_size: int):
        #Create networks with paper hyperparameters
        solve_nn = NetworkFactory.create_standard_nn(input_size, self.domain)
        uncertainty_nn = NetworkFactory.create_weight_uncertainty_nn(input_size, self.domain)
        
        return solve_nn, uncertainty_nn
    
    def create_optimizers(self, solve_nn, uncertainty_nn):
        #Create optimizers with paper hyperparameters
        solve_optimizer = OptimizerFactory.create_ffnn_optimizer(solve_nn)
        uncertainty_optimizer = OptimizerFactory.create_wunn_optimizer(uncertainty_nn)
        
        return solve_optimizer, uncertainty_optimizer
    
    def run_training_iteration(self, iteration: int, heuristic, task_generator):
        #Run one training iteration with paper parameters
        beta = self.scheduler.get_beta(iteration)
        
        # Generate tasks
        plans = []
        solved_count = 0
        
        num_tasks = PaperHyperparameters.EXPERIMENT['num_tasks_per_iter']
        timeout_ms = PaperHyperparameters.TIMEOUTS['15_puzzle_subopt'] * 1000
        
        for task_num in range(num_tasks):
            # Generate task using uncertainty
            if heuristic.is_trained:
                start_state = task_generator.generate_task()
            else:
                # Random task for early iterations
                puzzle = task_generator.puzzle
                start_state = puzzle._scramble(puzzle.goal(), 10 + iteration * 2)
            
            # Solve task
            from domain import SlidingPuzzle
            task_puzzle = SlidingPuzzle(
                size=16, 
                heuristic=heuristic, 
                init=start_state.arr
            )
            
            success, path, cost, stats = task_puzzle.solve_ida_star(t_max_ms=timeout_ms)
            
            if success and len(path) > 0:
                solved_count += 1
                plans.append(path)
        
        # Update heuristic
        if plans:
            heuristic.update(plans)
        
        # Adaptive confidence adjustment
        solve_rate = solved_count / num_tasks
        heuristic.adapt_confidence_level(solve_rate)
        
        return {
            'iteration': iteration,
            'solved_count': solved_count,
            'solve_rate': solve_rate,
            'confidence_level': heuristic.confidence_level,
            'beta': beta
        }

# Usage example function
def create_paper_experiment():
    #Example of how to use the paper hyperparameters
    
    # Create experiment runner
    runner = PaperExperimentRunner(domain='15_puzzle')
    
    # Create representation
    from StateRepresentation import TwoDimRepresentation
    representation = TwoDimRepresentation(
        size=16,
        response_func=lambda x: x/10,  # Paper scaling
        response_func_inv=lambda x: x*10
    )
    
    # Create networks with paper hyperparameters
    solve_nn, uncertainty_nn = runner.create_networks(representation.get_num_features())
    
    # Create optimizers
    solve_optimizer, uncertainty_optimizer = runner.create_optimizers(solve_nn, uncertainty_nn)
    
    # Create heuristic
    from NeuralNetworkComponents import BayesianNeuralHeuristic
    heuristic = BayesianNeuralHeuristic(
        representation=representation,
        solve_nn=solve_nn,
        uncertainty_nn=uncertainty_nn,
        confidence_level=PaperHyperparameters.CONFIDENCE['initial_alpha'],
        l2_loss=True
    )
    
    # Create task generator
    from NeuralNetworkComponents import TaskGenerator
    from domain import SlidingPuzzle
    puzzle = SlidingPuzzle(size=16, heuristic=heuristic)
    task_generator = TaskGenerator(
        puzzle=puzzle,
        uncertainty_nn=uncertainty_nn,
        representation=representation,
        epsilon=PaperHyperparameters.BAYESIAN['uncertainty_threshold'],
        max_steps=PaperHyperparameters.EXPERIMENT['max_steps']
    )
    
    return runner, heuristic, task_generator

if __name__ == "__main__":
    # Example usage
    runner, heuristic, task_generator = create_paper_experiment()
    
    print("Paper Hyperparameters:")
    print(f"Hidden neurons: {PaperHyperparameters.ARCHITECTURE['hidden_neurons_15_puzzle']}")
    print(f"Learning rates: FFNN={PaperHyperparameters.TRAINING['learning_rate_ffnn']}, WUNN={PaperHyperparameters.TRAINING['learning_rate_wunn']}")
    print(f"Initial β: {PaperHyperparameters.BAYESIAN['initial_beta']}")
    print(f"Initial α: {PaperHyperparameters.CONFIDENCE['initial_alpha']}")
    
    # Run a few training iterations
    for iteration in range(5):
        result = runner.run_training_iteration(iteration, heuristic, task_generator)
        print(f"Iteration {iteration + 1}: Solved {result['solved_count']}/10, α={result['confidence_level']:.2f}")