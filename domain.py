import math
import random
import numpy as np
import pickle
import time
import gc
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set, Optional, Callable, Any

class Operation:
    #Represents a move in the sliding puzzle
    def __init__(self, val: int):
        self.val = val
        
    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return self.val == other.val
    
    def __hash__(self):
        return hash(self.val)
    
    def __repr__(self):
        return f"Operation({self.val})"

class Edge:
    #Represents an edge in the search graph
    def __init__(self, cost: int = 1, op=None, prev_op=None):
        self.cost = cost
        self.op = op  # Operation that led to the child
        self.prev_op = prev_op  # Previous operation (at parent)
        self.child = None
        self.undo_token = None
        
    def __repr__(self):
        return f"Edge(cost={self.cost}, op={self.op}, prev_op={self.prev_op})"

class UndoToken:
    #Token for undoing moves
    def __init__(self):
        self.h = 0
        self.op = None

class State:
    #Represents a state in the sliding puzzle
    def __init__(self, arr=None, op=None):
        self.arr = arr if arr is not None else []
        self.op = op
        self.h = 0  # Heuristic value
        self.g = 0  # Path cost from start
        self.f = 0  # f = g + h
        
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return np.array_equal(self.arr, other.arr)
    
    def __hash__(self):
        return hash(tuple(self.arr))
    
    def __lt__(self, other):
        return self.f < other.f
    
    def copy(self):
        new_state = State(np.copy(self.arr), self.op)
        new_state.h = self.h
        new_state.g = self.g
        new_state.f = self.f
        return new_state
    
    def __repr__(self):
        dim = int(math.sqrt(len(self.arr)))
        s = ""
        for i in range(dim):
            for j in range(dim):
                val = self.arr[i * dim + j]
                s += f"{val:2d} " if val > 0 else " 0 "
            s += "\n"
        return s

class SearchStats:
    #track search statistics like the c# implementation
    def __init__(self):
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.start_time = None
        self.elapsed_time = 0
        self.max_depth_reached = 0
        self.timeout_occurred = False
        self.solution_found = False
        self.iterations = 0
    
    def start_timer(self):
        self.start_time = time.time()

    def update_time(self):
        if self.start_time:
            self.elapsed_time = time.time() - self.start_time
    
    def is_timeout(self, max_time_seconds):
        #check timeout like c# stopwatcg.ElapsedMillisecods > tMax
        if max_time_seconds is None:
            return False
        
        self.update_time()
        elapsed_ms = self.elapsed_time * 1000 #convert to milliseconds like c#

        if elapsed_ms > max_time_seconds:
            self.timeout_occurred = True
            return True
        return False


class IHeuristic:
    #Interface for heuristic functions
    def h(self, state: State, verbose: bool = False) -> int:
        raise NotImplementedError
    
    def update(self, paths):
        pass
    
    def clear_cache(self):
        pass

class HammingHeuristic(IHeuristic):
    #Hamming distance heuristic - counts misplaced tiles
    def h(self, state: State, verbose: bool = False) -> int:
        count = 0
        for i in range(len(state.arr)):
            if state.arr[i] == 0:
                continue
            if state.arr[i] != i:
                count += 1
        return count

class ManhattanDistanceHeuristic(IHeuristic):
    #Manhattan distance heuristic
    def __init__(self, size: int):
        self.size = size
        self.dim = int(math.sqrt(size))
        self.md = np.zeros((size, size), dtype=int)
        self.md_max = 0
        self._init_md()
        
    def _init_md(self):
        #Initialize Manhattan distance lookup table
        all_vals = []
        
        for t in range(self.size):
            grow = t // self.dim
            gcol = t % self.dim
            for l in range(self.size):
                row = l // self.dim
                col = l % self.dim
                val = abs(col - gcol) + abs(row - grow)
                self.md[t, l] = val
                all_vals.append(val)
        
        all_vals.sort(reverse=True)
        self.md_max = sum(all_vals[:self.size - 1])
    
    def h(self, state: State, verbose: bool = False) -> int:
        #Calculate Manhattan distance for a state
        sum_val = 0
        arr = state.arr
        zero_pos = state.op.val if state.op else np.where(arr == 0)[0][0]
        
        for i in range(self.size):
            if i == zero_pos:
                continue
            tile = arr[i]
            if tile == 0:
                continue
            sum_val += self.md[tile, i]
        
        return sum_val

class PDBHeuristic(IHeuristic):
    #Pattern Database heuristic
    def __init__(self, pdb_tiles, size=16):
        
        #Initialize with tiles to track in the pattern database
        #pdb_tiles: list of tile indices to track (e.g. [1, 2, 5, 6, 7])
        
        self.size = size
        self.dim = int(math.sqrt(size))
        self.pdb_tiles = sorted(pdb_tiles)
        self.pdb = {}  # Pattern database
        self.max_value = 0
        self.filename = f"pdb_{'_'.join(map(str, self.pdb_tiles))}.pkl"
    
    def _pattern_to_key(self, state):
        #Convert state to pattern key
        pattern = [0] * len(self.pdb_tiles)
        tiles_to_indices = {}
        
        # Map position of each tile in state
        for i, val in enumerate(state.arr):
            if val in self.pdb_tiles:
                tiles_to_indices[val] = i
        
        # Create pattern key
        for i, tile in enumerate(self.pdb_tiles):
            if tile in tiles_to_indices:
                pattern[i] = tiles_to_indices[tile]
            else:
                pattern[i] = -1  # Tile not found
        
        return tuple(pattern)
    
    def build_pdb(self, max_depth=30):
        #Build pattern database using BFS from goal state
        print(f"Building PDB for tiles {self.pdb_tiles}...")
        
        # Create goal state
        goal_state = State(np.arange(self.size, dtype=np.int8), Operation(0))
        
        # Initialize search
        queue = deque([(goal_state, 0)])  # (state, cost)
        self.pdb = {}
        visited = set()
        
        goal_key = self._pattern_to_key(goal_state)
        self.pdb[goal_key] = 0
        visited.add(goal_key)
        
        # Create sliding puzzle for operations
        puzzle = SlidingPuzzle(size=self.size)
        
        count = 0
        max_cost = 0
        
        # BFS
        while queue and count < 1000000:
            state, cost = queue.popleft()
            count += 1
            
            if cost > max_cost:
                max_cost = cost
                print(f"Depth {cost}, States: {count}, PDB size: {len(self.pdb)}")
            
            if cost >= max_depth:
                continue
            
            # Get possible operations
            ops = puzzle.operations(state)
            
            # Apply each operation
            for op in ops:
                child, _ = puzzle.apply(state, op)
                child_key = self._pattern_to_key(child)
                
                if child_key not in visited:
                    self.pdb[child_key] = cost + 1
                    visited.add(child_key)
                    queue.append((child, cost + 1))
        
        self.max_value = max(self.pdb.values())
        print(f"PDB built with {len(self.pdb)} patterns, max cost: {self.max_value}")
    
    def save(self):
        #Save pattern database to file
        with open(self.filename, 'wb') as f:
            pickle.dump(self.pdb, f)
        print(f"PDB saved to {self.filename}")
    
    def load(self):
        #Load pattern database from file
        try:
            with open(self.filename, 'rb') as f:
                self.pdb = pickle.load(f)
            self.max_value = max(self.pdb.values()) if self.pdb else 0
            print(f"PDB loaded from {self.filename}, size: {len(self.pdb)}, max cost: {self.max_value}")
            return True
        except FileNotFoundError:
            print(f"PDB file {self.filename} not found")
            return False
    
    def h(self, state: State, verbose: bool = False) -> int:
        #Calculate heuristic value using pattern database
        key = self._pattern_to_key(state)
        return self.pdb.get(key, 0)

class MultiHeuristic(IHeuristic):
    #Combines multiple heuristics, taking the maximum value
    def __init__(self, heuristics):
        self.heuristics = heuristics
    
    def h(self, state: State, verbose: bool = False) -> int:
        #Return maximum value from all heuristics
        return max(h.h(state, verbose) for h in self.heuristics)

class SlidingPuzzle:
    #Sliding puzzle domain implementation
    def __init__(self, size: int = 16, heuristic: Optional[IHeuristic] = None, init=None):
        self.size = size
        self.dim = int(math.sqrt(size))
        self.heuristic = heuristic
        self.init_state = None
        self.optab = self._build_op_tab()
        
        if init is not None:
            self.init_state = self._create_initial_state(init)
        else:
            # Create goal state and scramble it
            goal = self.goal()
            self.init_state = self._scramble(goal, 40)
    
    def _build_op_tab(self):
        #Build operation table for possible moves from each position
        #max_branch_factor = 4
        optab = []
        
        for i in range(self.size):
            ops = []
            # Up
            if i >= self.dim:
                ops.append(i - self.dim)
            # Left
            if i % self.dim > 0:
                ops.append(i - 1)
            # Right
            if i % self.dim < self.dim - 1:
                ops.append(i + 1)
            # Down
            if i < self.size - self.dim:
                ops.append(i + self.dim)
            
            optab.append(ops)
        
        return optab
    
    def _create_initial_state(self, init):
        #Create initial state from an array
        state = State(np.array(init, dtype=np.int8))
        
        # Find the blank tile (0)
        blank_pos = np.where(state.arr == 0)[0][0]
        state.op = Operation(blank_pos)
        
        if self.heuristic:
            state.h = self.heuristic.h(state)
            state.f = state.g + state.h
        
        return state
    
    def goal(self):
        #Create goal state
        arr = np.arange(self.size, dtype=np.int8)
        state = State(arr, Operation(0))
        return state
    
    def is_goal(self, state: State) -> bool:
        #Check if state is goal state
        for i in range(self.size):
            if state.arr[i] != i:
                return False
        return True
    
    def operations(self, state: State) -> List[Operation]:
        #Get possible operations from state
        blank_pos = state.op.val
        return [Operation(op) for op in self.optab[blank_pos]]
    
    def apply(self, state: State, op: Operation) -> Tuple[State, Edge]:
        #Apply operation to state and return new state and edge
        # Create a new state
        new_state = state.copy()
        
        # Create edge
        edge = Edge(1, op, state.op)
        
        # Create undo token
        undo_token = UndoToken()
        undo_token.h = state.h
        undo_token.op = state.op
        edge.undo_token = undo_token
        
        # Apply operation
        tile = state.arr[op.val]
        new_state.arr[state.op.val] = tile
        new_state.op = op
        new_state.arr[op.val] = 0
        
        # Update heuristic
        if self.heuristic:
            new_state.h = self.heuristic.h(new_state)
            new_state.g = state.g + edge.cost
            new_state.f = new_state.g + new_state.h
        
        return new_state, edge
    
    def undo(self, state: State, edge: Edge) -> State:
        #Undo operation
        undo_token = edge.undo_token
        new_state = state.copy()
        new_state.h = undo_token.h
        new_state.arr[state.op.val] = state.arr[undo_token.op.val]
        new_state.op = undo_token.op
        new_state.arr[undo_token.op.val] = 0
        return new_state
    
    def expand(self, state: State) -> List[Tuple[State, Edge]]:
        #Expand state and return list of (child_state, edge)
        ops = self.operations(state)
        children = []
        
        for op in ops:
            child, edge = self.apply(state, op)
            edge.child = child
            children.append((child, edge))
        
        return children
    
    def _scramble(self, state: State, steps: int) -> State:
        #Scramble the puzzle with random moves
        current = state.copy()
        prev_op = None
        
        for _ in range(steps):
            ops = self.operations(current)
            # Filter out the reverse of the previous operation
            if prev_op is not None:
                ops = [op for op in ops if op.val != prev_op]
            
            # Choose a random operation
            if ops:
                op = random.choice(ops)
                new_state, edge = self.apply(current, op)
                current = new_state
                prev_op = current.op.val
        
        return current
    
    def index_to_coord(self, index):
        #Convert 1D index to 2D coordinates (1-indexed)
        y = (index // self.dim) + 1
        x = (index % self.dim) + 1
        return (x, y)

    def solve_astar(self, t_max_ms:Optional[int] = None, max_nodes:int =500000) -> Tuple[bool, List[State], int, SearchStats]:
        #Solve the puzzle using A* search with time out handling
        stats = SearchStats()
        stats.start_timer()

        open_set = []
        closed_set = set()
        initial = self.init_state
        
        # Initialize f, g, h
        initial.g = 0
        if self.heuristic:
            initial.h = self.heuristic.h(initial)
        initial.f = initial.g + initial.h

        #Check if already solved
        if self.is_goal(initial):
            stats.solution_found = True
            return True, [initial], 0, stats
        
        # Push initial state to open set
        heappush(open_set, (initial.f, id(initial), initial))
        
        print(f"Starting A* - initial f: {initial.f}, Timeout: {t_max_ms}ms")
        
        while open_set and stats.nodes_expanded < max_nodes:

            #Timeout check like c# implementation
            if stats.is_timeout(t_max_ms):
                print(f"A* timeout after {stats.elapsed_time:.1f}s")
                return False, [], -1, stats
            
            _, _, current = heappop(open_set)

            if self.is_goal(current):
                stats.solution_found = True
                print(f"A* solved in {stats.elapsed_time:.1f}s")
                return True, [current], current.g, stats
            
            _, _, current = heappop(open_set)
            
            # Check if goal
            if self.is_goal(current):
                stats.solution_found = True
                return True, [current], current.g, stats
            
            state_hash = hash(tuple(current.arr))
            if state_hash in closed_set:
                continue

            closed_set.add(state_hash)
            stats.nodes_expanded += 1

            if stats.nodes_expanded % 10000 == 0:
                print(f"A* expanded {stats.nodes_expanded} nodes, f = {current.f}")

            for child, edge in self.expand(current):
                child_hash = hash(tuple(child.arr))
                if child_hash in closed_set:
                    continue

                stats.nodes_generated += 1 
                heappush(open_set, (child.f, id(child), child))

        if stats.nodes_expanded >= max_nodes:
            print(f"A* node limit reached({max_nodes})")
        else:
            print(f"A* open set exhausted")

        return False, [], -1, stats
            
            


    def solve_ida_star(self, t_max_ms: Optional[int] = None) -> Tuple[bool, List[State], int, SearchStats]:
        #Solve the puzzle using IDA* search with timeout
        initial = self.init_state
        
        stats = SearchStats()
        stats.start_timer()

        initial = self.init_state

        # Initialize h
        if self.heuristic:
            initial.h = self.heuristic.h(initial)
        
        bound = initial.h
        path = [initial]

        if self.is_goal(initial):
            stats.solution_found = True
            return True, path, 0, stats
        
        print(f"Starting IDA* - Initial bound: {bound}, Timeout: {t_max_ms}ms")

        iteration = 0
        while len(path) == 1:
            iteration += 1
            stats.iterations = iteration
            
            min_over_bound = float('inf')
            min_over_bound_ref = [min_over_bound]

            goal_result = self._ida_dfs(path, 0, Operation(-1), bound, min_over_bound_ref, stats, t_max_ms)

            if goal_result is None:
                print(f"IDA* timeout after {stats.elapsed_time:.1f}s, {iteration} iterations")
                return False, [], -1, stats

            if goal_result is True:
                stats.solution_found = True
                path.reverse()
                print(f"IDA* solved in {stats.elapsed_time:.1f}s, {iteration} iterations")
                return True, path, len(path) - 1, stats

            #update bound for next iteration
            min_over_bound = min_over_bound_ref[0]
            if min_over_bound == float('inf'):
                print("IDA* search space exhausted - no solution")
                return False, [], -1, stats
            bound = min_over_bound
            print(f"IDA* iteration {iteration}: bound={bound}, expanded = {stats.nodes_expanded}")

            if bound > 200:
                print(f"IDA* bound too high ({bound}) - likely insolvable")
                return False, [], -1, stats
            
        return False, [], -1, stats
    
    
    def _ida_dfs(self, path, cost, prev_op, bound, min_over_bound_ref, stats, t_max_ms):
        #Recursive IDA* search"""
        if stats.is_timeout(t_max_ms):
            return None
        
        current = path[-1]
        f = cost + current.h

        stats.nodes_expanded +=1

        if f <= bound and self.is_goal(current):
            return True
        
        if f> bound:
            min_over_bound_ref[0] = min(min_over_bound_ref[0], f)
            return False
        
        if len(path) > 200:
            return False
        
        ops = self.operations(current)
        stats.nodes_generated += len(ops)

        for op in ops:
            if prev_op.val != -1 and op.val == prev_op.val:
                continue

            child, edge = self.apply(current, op)

            child_hash = hash(tuple(child.arr))
            path_hashes = [hash(tuple(state.arr)) for state in path]
            if child_hash in path_hashes:
                continue

            path.append(child)

            goal_result = self._ida_dfs(path, cost + edge.cost, current.op, bound, min_over_bound_ref, stats, t_max_ms)

            path.pop()

            if goal_result is True:
                return True
            if goal_result is None:
                return None
            
        return False

    def create_solvable_test_case(self, difficulty=5):
    #Create a solvable test case by scrambling from goal
        goal = self.goal()
        current = goal.copy()
        prev_op = None
        
        for _ in range(difficulty):
            ops = self.operations(current)
            # Filter out reverse moves
            if prev_op is not None:
                ops = [op for op in ops if op.val != prev_op]
            
            if ops:
                op = random.choice(ops)
                new_state, _ = self.apply(current, op)
                current = new_state
                prev_op = current.op.val
        
        return current

def build_pdb_suite(size=16):
    #Build a suite of pattern databases for the sliding puzzle
    
    if size == 16:  # 15-puzzle
        # Different partitions of tiles for pattern databases
        pdb_partitions = [
            [1, 2, 5, 6, 7],
            [3, 4, 8, 9, 14],
            [10, 15, 16, 20, 21],  # Note: adjusted for 0-indexed
            [13, 18, 19, 23, 24],  # Note: adjusted for 0-indexed
            [11, 12, 17, 22],      # Note: adjusted for 0-indexed
        ]
        
        # Convert to 0-indexed
        pdb_partitions = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16],  # Note: 16 doesn't exist in 0-indexed 16-tile puzzle
        ]
        
        # Corrected partitions for 15-puzzle (0-15 indices)
        pdb_partitions = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15],
        ]
    else:
        # For other sizes, create simple partitions
        tiles = list(range(1, size))
        partition_size = max(3, size // 4)
        pdb_partitions = []
        
        for i in range(0, len(tiles), partition_size):
            partition = tiles[i:i + partition_size]
            if partition:
                pdb_partitions.append(partition)
    
    pdbs = []
    
    for i, partition in enumerate(pdb_partitions):
        print(f"Building PDB {i+1}/{len(pdb_partitions)} for tiles {partition}")
        pdb = PDBHeuristic(partition, size)
        
        # Try to load existing PDB, otherwise build it
        if not pdb.load():
            pdb.build_pdb(max_depth=12)  # Limit depth to keep reasonable size
            pdb.save()
        
        pdbs.append(pdb)
    
    return pdbs

# Example usage and testing
def test_timeout_behavior():
    #Test the timeout behavior with different puzzle difficulties
    print(" Testing timeout behavior...")
    
    # Easy test case (one move from solved)
    easy_case = [
        1, 0, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    ]
    
    # Harder test case
    hard_case = [
        5, 1, 3, 4,
        2, 7, 8, 12,
        9, 6, 11, 15,
        13, 10, 14, 0
    ]
    
    heuristic = ManhattanDistanceHeuristic(16)
    
    print("\n Testing easy case with 5s timeout...")
    puzzle = SlidingPuzzle(size=16, heuristic=heuristic, init=easy_case)
    success, path, cost, stats = puzzle.solve_ida_star(t_max_ms=5000)  # 5 seconds
    
    print(f"Result: {success}, Cost: {cost}, Time: {stats.elapsed_time:.2f}s, Timeout: {stats.timeout_occurred}")
    
    print("\n Testing hard case with 10s timeout...")
    puzzle = SlidingPuzzle(size=16, heuristic=heuristic, init=hard_case)
    success, path, cost, stats = puzzle.solve_ida_star(t_max_ms=10000)  # 10 seconds
    
    print(f"Result: {success}, Cost: {cost}, Time: {stats.elapsed_time:.2f}s, Timeout: {stats.timeout_occurred}")
    
    return success

if __name__ == "__main__":
    print(" ENHANCED SLIDING PUZZLE - FIXED VERSION")
    print("=" * 60)
    
    # Test timeout behavior
    print("\n1. Testing timeout behavior...")
    test_timeout_behavior()