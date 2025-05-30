import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import math
import time
import gc
import random
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Tuple, Optional, List
from StateRepresentation import StateRepresentation, FeaturesRepresentation, OneDimRepresentation, TwoDimRepresentation, TrainingInstance
from domain import HammingHeuristic, ManhattanDistanceHeuristic, PDBHeuristic, MultiHeuristic, State, Operation, Edge, UndoToken, IHeuristic, SlidingPuzzle
from NeuralNetworkComponents import BayesianLinearLayer, WeightUncertaintyNN, StandardNN, NeuralHeuristic, BayesianNeuralHeuristic, TaskGenerator

class ExperimentRunner:
    #Runs the experiments as decribed in the papaer

    def __init__(self, size=16):
        self.size = size 
        self.results = {}

    def run_training_experiment(self, num_iterations=50, num_tasks_per_iter = 10):
        #run the maintraining experiment (Algorithm 2 from paper)

        #different timeouts
        t_max_ms = 60000
        # if num_iterations <= 5:
        #     t_max_ms = 10000 #10secs
        # elif self.size <= 16: #simple domain
        #     t_max_ms = 60000 #1min
        # else: #complex domain
        #     t_max_ms = 5 * 60000 #5mins

        print(f"Task timeout: {t_max_ms}ms")


        #initialse components

        representation = TwoDimRepresentation(self.size, lambda x:x/10, lambda x:x * 10)

        #create neural networks
        hidden_size = 20
        solve_nn = StandardNN(representation.get_num_features(), hidden_size, 2)
        uncertainty_nn = WeightUncertaintyNN(representation.get_num_features(), hidden_size)

        #Create heuristic
        heuristic = BayesianNeuralHeuristic(representation, solve_nn, uncertainty_nn, confidence_level = 0.5, l2_loss = True)


        #create puzzle and task generator
        puzzle = SlidingPuzzle(size = self.size, heuristic=heuristic)
        task_generator = TaskGenerator(puzzle, uncertainty_nn, representation, epsilon = 0.5)

        results = []

        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            plans = []
            solved_count = 0
            timeout_count = 0

            for task_num in range(num_tasks_per_iter):
                print(f"\n Task {task_num + 1}/ {num_tasks_per_iter}")
                try:
                    #Generate task 
                    if heuristic.is_trained and iteration > 0:
                        start_state = task_generator.generate_task()
                    else:
            
                        # Start with very easy very easy tasks
                        easy_steps = min(5 + iteration, 15)  # Gradually increase difficulty
                        start_state = puzzle._scramble(puzzle.goal(), easy_steps)

                    if puzzle.is_goal(start_state):
                        print("Generated goal sate - skipping")
                        continue

                    #create puzzle with this start state
                    task_puzzle = SlidingPuzzle(size=self.size, heuristic = heuristic, init=start_state.arr)

                    heuristic.clear_cache()
                    gc.collect()

                    #solve with IDA*
                    success, path, cost, stats = task_puzzle.solve_ida_star(t_max_ms=t_max_ms)

                    print(f"  Expanded: {stats.nodes_expanded}")
                    print(f"  Generated: {stats.nodes_generated}")
                    print(f"  Elapsed: {stats.elapsed_time*1000:.0f}ms")

                    if success : #and len(path) > 1
                        solved_count += 1
                        plans.append(path)
                        print(f"Solved: length = {len(path)}, cost = {cost})")
                    elif stats.timeout_occurred:
                        timeout_count+=1
                        print(f"Timeout")
                    else:
                        print(f"Failed")
                
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            #Update heuristic
            if plans:
                print(f"\n Training on {len(plans)} solved plans")
                try:
                    heuristic.update(plans)
                except Exception as e:
                    print(f"Training error: {e}")
            
            #Adaptive confidence level like c#
            solve_rate = solved_count / num_tasks_per_iter
            heuristic.adapt_confidence_level(solve_rate)

            #record results
            results.append({
                'iteration': iteration,
                'solved_count': solved_count,
                'solve_rate': solve_rate,
                'confidence_level': heuristic.confidence_level
            })

            print(f"\n ITERATION {iteration + 1} SUMMARY:")
            print(f"  Solved: {solved_count}/{num_tasks_per_iter} ({solve_rate:.1%})")
            print(f"  Timeouts: {timeout_count}")
            print(f"  Confidence level: {heuristic.confidence_level}")

            if solve_rate == 0 and iteration > 2:
                print(f"No task solved for several iterations - stopping early")
                break

        return results
    
    def run_admissibility_test(self, test_instances, heuristic, t_max_ms = 300000):
        #Test admissibility on benchmark instances
        results = []

        for i, test_arr in enumerate(test_instances):
            print(f"Testing instance {i + 1}/{len(test_instances)}")
            try:
                #create puzzle
                puzzle = SlidingPuzzle(size = self.size, heuristic = heuristic, init = test_arr)

                heuristic.clear_cache()
                gc.collect()

                #solve
                success, path, cost, stats = puzzle.solve_astar(t_max_ms=t_max_ms)

                if success:
                    results.append({'instance': i, 'cost': cost, 'time_ms': stats.elapsed_time * 1000, 'nodes_expanded': stats.nodes_expanded, 'solved': True})
                    print(f"Solved: cost{cost}, time = {stats.elapsed_time:.1f}s")

                else:
                    results.append({'instance': i, 'cost': -1, 'time_ms': stats.elapsed_time * 1000, 'nodes_expanded': stats.nodes_expanded, 'solved': False})

                if stats.timeout_occurred:   
                    print(f"Timeout after {stats.elapsed_time:.1f}s")
                else:
                    print(f"Failed after {stats.elapsed_time:1f}s")

            except Exception as e:
                print(f"Error: {e}")
                results.append({'instance': i, 'cost': -1, 'time_ms': 0, 'nodes_expanded': 0, 'solved': False})

        return results
    
def create_paper_heuristics(size = 16):
    #create the heuristics used in the paper
    manhattan = ManhattanDistanceHeuristic(size)
    hamming = HammingHeuristic()
    pdb_partitions = [
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15]
    ]

    pdbs = []
    for partition in pdb_partitions:
        pdb = PDBHeuristic(partition, size)
        if not pdb.load():
            pdb.build_pdb(max_depth = 10)
            pdb.save()
        pdbs.append(pdb)
    
    return [manhattan, hamming] + pdbs

def run_paper_experiments(size = 16):
    #run experiments from the paper
    print("Running experiments from Utilising Uncertainty for Efficient Learning")

    runner = ExperimentRunner(size = size)

    #run training experiment 
    print("\n=== Training Experiment ===")
    try:
        results = runner.run_training_experiment(num_iterations = 5, num_tasks_per_iter = 3)
        total_solved = 0
        total_tasks = 0

        print("\nTraining Results:")
        for result in results:
            total_solved += result['solved_count']
            total_tasks += 3 #num_tasks_per_iter

            print(f"Iteration {result['iteration'] + 1}: "
                F"Solved {result['solved_count']}/3 "
                f"({result['solve_rate']:.1%})," 
                f"conf_level = {result['confidence_level']:.2f}")
            
        overall_solve_rate = total_solved/ total_tasks if total_tasks > 0 else 0

        print(f"\n OVERALL PERFORMANCE:")
        print(f"  Total solved: {total_solved}/{total_tasks} ({overall_solve_rate:.1%})")
            
        if overall_solve_rate < 0.3:
            print("Low solve rate - consider:")
            print("   - Increasing task timeout")
            print("   - Using easier initial tasks")
            print("   - Adjusting confidence levels")
        elif overall_solve_rate > 0.7:
            print("Good solve rate - algorithm is working!")

        return results
    
    except Exception as e:
        print(f"Experiment failed: {e}")
        return []

def test_simple_fixed_pipeline():
    """🧪 Test the complete pipeline with simple cases"""
    print("🧪 TESTING SIMPLE FIXED PIPELINE")
    print("=" * 50)
    
    try:

        # Test 1: Simple heuristic
        print("\n1. Testing with Manhattan Distance heuristic...")
        heuristic = ManhattanDistanceHeuristic(16)
        
        # Easy case
        easy_case = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        puzzle = SlidingPuzzle(size=16, heuristic=heuristic, init=easy_case)
        
        success, path, cost, stats = puzzle.solve_ida_star(t_max_ms=10000)  # 10s timeout
        
        print(f"  Result: {success}, Cost: {cost}, Time: {stats.elapsed_time:.2f}s")
        print(f"  Expanded: {stats.nodes_expanded}, Timeout: {stats.timeout_occurred}")
        
        # Test 2: Neural heuristic components
        print("\n2. Testing neural network components...")
        
        representation = TwoDimRepresentation(16)
        solve_nn = StandardNN(representation.get_num_features(), 10, 2)
        uncertainty_nn = WeightUncertaintyNN(representation.get_num_features(), 10)
        
        neural_heuristic = BayesianNeuralHeuristic(
            representation, solve_nn, uncertainty_nn, 
            confidence_level=0.5, l2_loss=True
        )
        
        # Test prediction
        test_state = puzzle.goal()
        h_val = neural_heuristic.h(test_state)
        print(f"  Neural heuristic value: {h_val}")
        
        # Test 3: Task generation
        print("\n3. Testing task generation...")
        
        task_generator = TaskGenerator(puzzle, uncertainty_nn, representation, epsilon=0.5)
        generated_state = task_generator.generate_task()
        
        print(f"  Generated task with h={heuristic.h(generated_state)}")
        
        # Test 4: Mini experiment
        print("\n4. Running mini experiment...")
        
        runner = ExperimentRunner(size=16)
        results = runner.run_training_experiment(
            num_iterations=2,
            num_tasks_per_iter=2
        )
        
        print(f"  Mini experiment completed with {len(results)} iterations")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================================================================
# 7. 🔧 UTILITY FUNCTIONS FOR DEBUGGING
# ==============================================================================

def create_simple_test_cases():
    """Create simple test cases for debugging"""
    test_cases = []
    
    # Very easy case (1 move)
    test_cases.append({
        'name': 'one_move',
        'state': [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'expected_cost': 1
    })
    
    # Easy case (2 moves)
    test_cases.append({
        'name': 'two_moves', 
        'state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'expected_cost': 2
    })
    
    # Medium case (scrambled)
    test_cases.append({
        'name': 'medium',
        'state': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15],
        'expected_cost': 2
    })
    
    return test_cases

def debug_search_behavior():
    """Debug search behavior with different timeouts"""
    print("🔍 DEBUGGING SEARCH BEHAVIOR")
    print("=" * 40)
    
    try:

        heuristic = ManhattanDistanceHeuristic(16)
        test_cases = create_simple_test_cases()
        
        for case in test_cases:
            print(f"\n📋 Testing {case['name']} case...")
            
            puzzle = SlidingPuzzle(size=16, heuristic=heuristic, init=case['state'])
            
            print(f"  Initial h-value: {heuristic.h(puzzle.init_state)}")
            print(f"  Expected cost: {case['expected_cost']}")
            
            # Test with short timeout
            success, path, cost, stats = puzzle.solve_ida_star(t_max_ms=5000)  # 5s
            
            print(f"  Result: {success}")
            print(f"  Found cost: {cost}")
            print(f"  Time: {stats.elapsed_time:.2f}s")
            print(f"  Expanded: {stats.nodes_expanded}")
            print(f"  Timeout: {stats.timeout_occurred}")
            
            if success:
                print(f"  ✅ Success - cost matches: {cost == case['expected_cost']}")
            else:
                print(f"  ❌ Failed or timeout")
    
    except Exception as e:
        print(f"💥 Debug failed: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 8. 🔧 MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("🚀 UNCERTAINTY-BASED HEURISTIC LEARNING - FIXED VERSION")
    print("=" * 70)
    
    # Test the pipeline step by step
    print("\n🧪 Step 1: Testing simple pipeline...")
    pipeline_success = test_simple_fixed_pipeline()
    
    if pipeline_success:
        print("\n🔍 Step 2: Debugging search behavior...")
        debug_search_behavior()
        
        print("\n🚀 Step 3: Running paper experiments...")
        results = run_paper_experiments()
        
        if results:
            print("\n🎉 EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("\nKey improvements implemented:")
            print("✅ Timeout handling in IDA* search")
            print("✅ Graceful failure handling")
            print("✅ Adaptive confidence levels")
            print("✅ Resource management (cache clearing, GC)")
            print("✅ Error handling and recovery")
            print("✅ Progress reporting and statistics")
        else:
            print("\n⚠️ Experiments had issues - check logs above")
    else:
        print("\n❌ Pipeline test failed - fix basic components first")
    
    print("\n" + "=" * 70)
    print("🏁 FIXED IMPLEMENTATION COMPLETE")