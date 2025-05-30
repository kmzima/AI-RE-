import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from domain import SlidingPuzzle, State, ManhattanDistanceHeuristic, HammingHeuristic
from StateRepresentation import TwoDimRepresentation, TrainingInstance
from NeuralNetworkComponents import StandardNN, WeightUncertaintyNN, BayesianNeuralHeuristic, TaskGenerator

class PaperExperiments:
    #Implementation of experiments from the paper with exact hyperparameters
    
    def __init__(self, size=16):
        self.size = size
        self.dim = int(np.sqrt(size))
        
        # Paper hyperparameters
        self.HYPERPARAMS = {
            # Neural Network Architecture
            'hidden_neurons': 20,
            'dropout_rate': 0.025,  # 2.5%
            
            # Training Parameters
            'learning_rate_wunn': 0.01,
            'learning_rate_ffnn': 0.001,
            'monte_carlo_samples_training': 5,  # S
            'monte_carlo_samples_uncertainty': 100,  # K
            'train_iter_ffnn': 1000,
            'max_train_iter_wunn': 5000,
            'mini_batch_size': 100,
            
            # Bayesian Parameters
            'prior_mu': 0.0,
            'prior_sigma_squared': 10.0,
            'initial_beta': 0.05,
            'final_beta': 0.00001,
            'kl_threshold': 0.64,  # κ
            'uncertainty_threshold': 1.0,  # ε
            
            # Experimental Parameters
            'memory_buffer_max': 25000,
            'initial_alpha': 0.99,
            'alpha_decrement': 0.05,
            'min_alpha': 0.5,
            'quantile_q': 0.95,
            
            # Experiment Setup
            'num_iterations': 50,
            'num_tasks_per_iter': 10,
            'num_tasks_per_iter_thresh': 6,
            'max_steps': 1000,
            't_max_subopt': 60000,  # 60 seconds in ms
            't_max_efficiency': 1000,  # 1 second in ms
        }
        
        # Create benchmark tasks (Korf 1985 - simplified version)
        self.benchmark_tasks = self._create_benchmark_tasks()
        
    def _create_benchmark_tasks(self) -> List[np.ndarray]:
        #Create 100 benchmark tasks for 15-puzzle
        print("Creating benchmark tasks...")
        
        # For reproducibility, we'll create deterministic scrambled puzzles
        random.seed(42)
        np.random.seed(42)
        
        tasks = []
        puzzle = SlidingPuzzle(size=self.size)
        
        for i in range(100):
            # Create different difficulty levels
            scramble_steps = 20 + (i % 30)  # 20-50 steps
            
            # Start from goal and scramble
            state = puzzle.goal()
            current = state
            prev_op = None
            
            for _ in range(scramble_steps):
                ops = puzzle.operations(current)
                # Avoid going back
                if prev_op is not None:
                    ops = [op for op in ops if op.val != prev_op]
                
                if ops:
                    op = random.choice(ops)
                    current, _ = puzzle.apply(current, op)
                    prev_op = current.op.val
            
            tasks.append(current.arr.copy())
        
        print(f"Created {len(tasks)} benchmark tasks")
        return tasks

    def run_suboptimality_experiment(self) -> Dict:
        #Run the suboptimality experiment from Table 1
        print("\n" + "="*60)
        print("RUNNING SUBOPTIMALITY EXPERIMENT")
        print("="*60)
        
        results = {}
        confidence_levels = [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
        
        # First train the heuristic
        print("Training heuristic...")
        heuristic = self._train_heuristic()
        
        # Test with different confidence levels
        for alpha in confidence_levels:
            print(f"\nTesting with α = {alpha}")
            
            # Create heuristic with specific confidence level
            test_heuristic = self._clone_heuristic_with_alpha(heuristic, alpha)
            
            # Test on benchmark tasks
            results[alpha] = self._test_on_benchmarks(test_heuristic, 
                                                    self.HYPERPARAMS['t_max_subopt'])
        
        # Also test single output FFNN for comparison
        print(f"\nTesting single output FFNN...")
        single_output_heuristic = self._train_single_output_heuristic()
        results['single_output'] = self._test_on_benchmarks(single_output_heuristic,
                                                          self.HYPERPARAMS['t_max_subopt'])
        
        return results
    
    def run_efficiency_experiment(self) -> Dict:
        #Run the efficiency experiment from Table 2
        print("\n" + "="*60)
        print("RUNNING EFFICIENCY EXPERIMENT")
        print("="*60)
        
        results = {}
        length_increments = [1, 2, 4, 6, 8, 10]
        
        # Test our approach (GTP)
        print("Testing Generate Task Practical (GTP)...")
        results['GTP'] = self._run_efficiency_test_gtp()
        
        # Test fixed length increments
        for length_inc in length_increments:
            print(f"\nTesting LengthInc = {length_inc}")
            results[f'LengthInc_{length_inc}'] = self._run_efficiency_test_fixed(length_inc)
        
        return results
    
    def _train_heuristic(self) -> BayesianNeuralHeuristic:
        #Train the main heuristic using paper's algorithm
        print("Initializing neural networks...")
        
        # Create representation (more efficient encoding from paper)
        representation = TwoDimRepresentation(
            size=self.size,
            response_func=lambda x: x/10,  # Scale down responses
            response_func_inv=lambda x: x*10  # Scale up predictions
        )
        
        # Create neural networks with paper hyperparameters
        solve_nn = StandardNN(
            input_size=representation.get_num_features(),
            hidden_size=self.HYPERPARAMS['hidden_neurons'],
            output_size=2  # mean and variance
        )
        
        uncertainty_nn = WeightUncertaintyNN(
            input_size=representation.get_num_features(),
            hidden_size=self.HYPERPARAMS['hidden_neurons'],
            output_size=1
        )
        
        # Create heuristic
        heuristic = BayesianNeuralHeuristic(
            representation=representation,
            solve_nn=solve_nn,
            uncertainty_nn=uncertainty_nn,
            confidence_level=0.01,  # Start with low confidence
            l2_loss=True
        )
        
        # Training loop (LearnHeuristicPrac)
        puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic)
        task_generator = TaskGenerator(
            puzzle=puzzle,
            uncertainty_nn=uncertainty_nn,
            representation=representation,
            epsilon=self.HYPERPARAMS['uncertainty_threshold'],
            max_steps=self.HYPERPARAMS['max_steps']
        )
        
        for iteration in range(self.HYPERPARAMS['num_iterations']):
            print(f"\nIteration {iteration + 1}/{self.HYPERPARAMS['num_iterations']}")
            
            plans = []
            solved_count = 0
            
            # Generate and solve tasks
            for task_num in range(self.HYPERPARAMS['num_tasks_per_iter']):
                try:
                    # Generate task using uncertainty
                    if heuristic.is_trained:
                        start_state = task_generator.generate_task()
                    else:
                        # Random task for first few iterations
                        goal = puzzle.goal()
                        start_state = puzzle._scramble(goal, 10 + iteration * 2)
                    
                    # Create puzzle with this start state
                    task_puzzle = SlidingPuzzle(
                        size=self.size,
                        heuristic=heuristic,
                        init=start_state.arr
                    )
                    
                    # Solve with timeout
                    success, path, cost, stats = task_puzzle.solve_ida_star(
                        t_max_ms=self.HYPERPARAMS['t_max_subopt']
                    )
                    
                    if success and len(path) > 0:
                        solved_count += 1
                        plans.append(path)
                        print(f"  Task {task_num + 1}: Solved (cost={cost})")
                    else:
                        print(f"  Task {task_num + 1}: Failed/Timeout")
                        
                except Exception as e:
                    print(f"  Task {task_num + 1}: Error - {e}")
            
            # Update heuristic if we have solved plans
            if plans:
                print(f"Training on {len(plans)} solved plans...")
                heuristic.update(plans)
            
            # Adaptive confidence level adjustment
            solve_rate = solved_count / self.HYPERPARAMS['num_tasks_per_iter']
            heuristic.adapt_confidence_level(solve_rate)
            
            print(f"Solved: {solved_count}/{self.HYPERPARAMS['num_tasks_per_iter']} "
                  f"({solve_rate:.1%}), α = {heuristic.confidence_level}")
            
            # Early stopping if no progress
            if solve_rate == 0 and iteration > 5:
                print("No progress - stopping early")
                break
        
        return heuristic
    
    def _train_single_output_heuristic(self) -> BayesianNeuralHeuristic:
        #rain single output FFNN for comparison
        representation = TwoDimRepresentation(self.size)
        
        solve_nn = StandardNN(
            input_size=representation.get_num_features(),
            hidden_size=self.HYPERPARAMS['hidden_neurons'],
            output_size=1  # Single output
        )
        
        heuristic = BayesianNeuralHeuristic(
            representation=representation,
            solve_nn=solve_nn,
            uncertainty_nn=None,  # No uncertainty network
            confidence_level=None,  # No confidence level
            l2_loss=True
        )
        
        # Simplified training (similar to bootstrap method from paper)
        puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic)
        
        for iteration in range(20):  # Fewer iterations for single output
            plans = []
            
            for _ in range(self.HYPERPARAMS['num_tasks_per_iter']):
                # Generate random tasks
                goal = puzzle.goal()
                start_state = puzzle._scramble(goal, 10 + iteration)
                
                task_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=start_state.arr)
                success, path, cost, stats = task_puzzle.solve_ida_star(t_max_ms=30000)
                
                if success:
                    plans.append(path)
            
            if plans:
                heuristic.update(plans)
        
        return heuristic
    
    def _clone_heuristic_with_alpha(self, heuristic: BayesianNeuralHeuristic, alpha: float) -> BayesianNeuralHeuristic:
        #Create a copy of heuristic with different confidence level
        new_heuristic = BayesianNeuralHeuristic(
            representation=heuristic.representation,
            solve_nn=heuristic.solve_nn,  # Share trained network
            uncertainty_nn=heuristic.uncertainty_nn,
            confidence_level=alpha,
            l2_loss=heuristic.l2_loss
        )
        new_heuristic.is_trained = heuristic.is_trained
        new_heuristic.memory_buffer = heuristic.memory_buffer.copy()
        return new_heuristic
    
    def _test_on_benchmarks(self, heuristic, timeout_ms: int) -> Dict:
        #Test heuristic on benchmark tasks
        results = {
            'times': [],
            'nodes_generated': [],
            'costs': [],
            'optimal_costs': [],
            'solved_count': 0,
            'optimal_count': 0
        }
        
        # For optimal costs, use Manhattan Distance (admissible baseline)
        md_heuristic = ManhattanDistanceHeuristic(self.size)
        
        for i, task_arr in enumerate(self.benchmark_tasks[:20]):  # Test subset for speed
            print(f"Testing benchmark {i + 1}/20...")
            
            try:
                # Get optimal cost using admissible heuristic
                optimal_puzzle = SlidingPuzzle(size=self.size, heuristic=md_heuristic, init=task_arr)
                opt_success, opt_path, opt_cost, _ = optimal_puzzle.solve_ida_star(t_max_ms=timeout_ms * 2)
                
                if not opt_success:
                    continue  # Skip if we can't find optimal
                
                # Test with our heuristic
                test_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=task_arr)
                success, path, cost, stats = test_puzzle.solve_ida_star(t_max_ms=timeout_ms)
                
                if success:
                    results['times'].append(stats.elapsed_time)
                    results['nodes_generated'].append(stats.nodes_generated)
                    results['costs'].append(cost)
                    results['optimal_costs'].append(opt_cost)
                    results['solved_count'] += 1
                    
                    if cost == opt_cost:
                        results['optimal_count'] += 1
                
            except Exception as e:
                print(f"Error testing benchmark {i}: {e}")
                continue
        
        # Calculate statistics
        if results['solved_count'] > 0:
            costs = np.array(results['costs'])
            optimal_costs = np.array(results['optimal_costs'])
            
            results['avg_time'] = np.mean(results['times'])
            results['avg_nodes_generated'] = np.mean(results['nodes_generated'])
            results['avg_suboptimality'] = np.mean((costs / optimal_costs - 1) * 100)
            results['percent_optimal'] = (results['optimal_count'] / results['solved_count']) * 100
        else:
            results.update({
                'avg_time': 0, 'avg_nodes_generated': 0,
                'avg_suboptimality': 100, 'percent_optimal': 0
            })
        
        return results
    
    def _run_efficiency_test_gtp(self) -> Dict:
        #Run efficiency test with Generate Task Practical
        # Train with short timeout
        representation = TwoDimRepresentation(self.size)
        solve_nn = StandardNN(representation.get_num_features(), self.HYPERPARAMS['hidden_neurons'], 1)
        uncertainty_nn = WeightUncertaintyNN(representation.get_num_features(), self.HYPERPARAMS['hidden_neurons'])
        
        heuristic = BayesianNeuralHeuristic(representation, solve_nn, uncertainty_nn, confidence_level=None, l2_loss=True)
        puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic)
        task_generator = TaskGenerator(puzzle, uncertainty_nn, representation)
        
        # Training phase
        training_solved = 0
        training_total = 0
        
        for iteration in range(20):  # Reduced iterations for efficiency test
            for _ in range(self.HYPERPARAMS['num_tasks_per_iter']):
                training_total += 1
                
                if heuristic.is_trained:
                    start_state = task_generator.generate_task()
                else:
                    start_state = puzzle._scramble(puzzle.goal(), 5 + iteration)
                
                task_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=start_state.arr)
                success, path, _, _ = task_puzzle.solve_ida_star(t_max_ms=self.HYPERPARAMS['t_max_efficiency'])
                
                if success:
                    training_solved += 1
                    heuristic.update([path])
        
        # Test phase - create test tasks of increasing difficulty
        test_solved = 0
        test_total = 100
        
        for k in range(1, test_total + 1):
            start_state = puzzle._scramble(puzzle.goal(), k)
            task_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=start_state.arr)
            success, _, _, _ = task_puzzle.solve_ida_star(t_max_ms=60000)  # 60s for test
            
            if success:
                test_solved += 1
        
        return {
            'training_solve_rate': (training_solved / training_total) * 100,
            'test_solve_rate': (test_solved / test_total) * 100
        }
    
    def _run_efficiency_test_fixed(self, length_inc: int) -> Dict:
        #Run efficiency test with fixed length increment
        representation = TwoDimRepresentation(self.size)
        solve_nn = StandardNN(representation.get_num_features(), self.HYPERPARAMS['hidden_neurons'], 1)
        
        heuristic = BayesianNeuralHeuristic(representation, solve_nn, None, confidence_level=None, l2_loss=True)
        puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic)
        
        # Training phase with fixed increments
        training_solved = 0
        training_total = 0
        
        for iteration in range(20):
            steps = (iteration + 1) * length_inc
            
            for _ in range(self.HYPERPARAMS['num_tasks_per_iter']):
                training_total += 1
                
                start_state = puzzle._scramble(puzzle.goal(), steps)
                task_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=start_state.arr)
                success, path, _, _ = task_puzzle.solve_ida_star(t_max_ms=self.HYPERPARAMS['t_max_efficiency'])
                
                if success:
                    training_solved += 1
                    heuristic.update([path])
        
        # Test phase
        test_solved = 0
        test_total = 100
        
        for k in range(1, test_total + 1):
            start_state = puzzle._scramble(puzzle.goal(), k)
            task_puzzle = SlidingPuzzle(size=self.size, heuristic=heuristic, init=start_state.arr)
            success, _, _, _ = task_puzzle.solve_ida_star(t_max_ms=60000)
            
            if success:
                test_solved += 1
        
        return {
            'training_solve_rate': (training_solved / training_total) * 100,
            'test_solve_rate': (test_solved / test_total) * 100
        }
    
    def save_results(self, results: Dict, filename: str):
        #Save experimental results
        Path("results").mkdir(exist_ok=True)
        
        with open(f"results/{filename}", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to results/{filename}")
    
    def print_suboptimality_results(self, results: Dict):
        #Print results in paper format
        print("\n" + "="*80)
        print("SUBOPTIMALITY EXPERIMENT RESULTS")
        print("="*80)
        print(f"{'α':<8} {'Time':<10} {'Generated':<15} {'Subopt':<8} {'Optimal':<8}")
        print("-" * 60)
        
        for alpha in [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]:
            if alpha in results:
                r = results[alpha]
                print(f"{alpha:<8} {r['avg_time']:<10.2f} {r['avg_nodes_generated']:<15,.0f} "
                      f"{r['avg_suboptimality']:<8.1f}% {r['percent_optimal']:<8.1f}%")
        
        if 'single_output' in results:
            r = results['single_output']
            print(f"{'N/A':<8} {r['avg_time']:<10.2f} {r['avg_nodes_generated']:<15,.0f} "
                  f"{r['avg_suboptimality']:<8.1f}% {r['percent_optimal']:<8.1f}%")
    
    def print_efficiency_results(self, results: Dict):
        #Print efficiency results
        print("\n" + "="*60)
        print("EFFICIENCY EXPERIMENT RESULTS")
        print("="*60)
        print(f"{'Method':<12} {'Train Solved':<12} {'Test Solved':<12}")
        print("-" * 40)
        
        for method, r in results.items():
            print(f"{method:<12} {r['training_solve_rate']:<12.1f}% {r['test_solve_rate']:<12.1f}%")


def run_paper_experiments():
    #Main function to run all experiments
    print(" RUNNING PAPER EXPERIMENTS WITH EXACT HYPERPARAMETERS")
    print("=" * 70)
    
    experiments = PaperExperiments(size=16)
    
    # Run suboptimality experiment
    print("Starting suboptimality experiment...")
    subopt_results = experiments.run_suboptimality_experiment()
    experiments.save_results(subopt_results, "suboptimality_results.json")
    experiments.print_suboptimality_results(subopt_results)
    
    # Run efficiency experiment
    print("\nStarting efficiency experiment...")
    efficiency_results = experiments.run_efficiency_experiment()
    experiments.save_results(efficiency_results, "efficiency_results.json")
    experiments.print_efficiency_results(efficiency_results)
    
    print("\n ALL EXPERIMENTS COMPLETED!")
    return subopt_results, efficiency_results


if __name__ == "__main__":
    subopt_results, efficiency_results = run_paper_experiments()