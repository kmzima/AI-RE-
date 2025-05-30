from abc import ABC, abstractmethod
import numpy as np
from domain import HammingHeuristic, ManhattanDistanceHeuristic, PDBHeuristic, MultiHeuristic, State
from typing import List
import math
import time
import gc
import random
from collections import deque

class StateRepresentation(ABC):
    #Abstract Class for state representation

    @abstractmethod
    def get_features(self, state) -> np.ndarray:
        #convert state to a feature vector
        pass

    @abstractmethod
    def get_num_features(self) -> int:
        #get number of features
        pass

class FeaturesRepresentation(StateRepresentation):
    #feature based representation using multiple heuristics

    def __init__(self, heuristics:List, size =16):
        self.heuristics = heuristics
        self.size = 16
        self.num_features = len(heuristics) + 1

        #calculate scaling factors for each heuristic
        self.scales = []
        for h in heuristics:
            if hasattr(h, 'md_max'):        #max value for manhatten distance heuristic, used to normalise the md heuristic
                self.scales.append(h.md_max)   
            elif hasattr(h, 'max_value'):       #max value for the PBD hueristic or other heuristics, used to normalise the heuristic 
                self.scales.append(h.max_value)
            else:
                self.scales.append(100) #default scale

    def get_features(self, state:State) -> np.ndarray:
        #convert state to a feature representation using heuristics

        features = np.zeros(self.num_features)
        for i, h in enumerate(self.heuristics):
            h_val = h.h(state)
            if self.scales[i] > 0:
                #normalise the heuristic value by its scale
                features[i] = h_val / self.scales[i]
            else:
                features[i] = h_val

        #add blank tile as a feature
        if state.op:
            blank_pos = state.op.val 
        else:
            blank_pos = 0
        
        features[-1] = blank_pos/self.size

        return features
    
    def get_num_features(self) -> int:
        return self.num_features
    
class OneDimRepresentation(StateRepresentation):
    #1D representation of the state (one hot encoding)

    def __init__(self, size = 16, response_func = None, response_func_inv = None):
        self.size = size
        self.num_features = size * size
        self.response_func = response_func if response_func else lambda x:x
        self.response_func_inv = response_func_inv if response_func_inv else lambda x:x

    def get_features(self, state:State) -> np.ndarray:
        #convert state to 1D feature representation

        features = np.zeros(self.num_features)

        for pos, tile in enumerate(state.arr):
            if tile == 0:
                continue
            index = pos * self.size + tile
            if index < len(features):
                features[index] = 1

        return features
    
    def get_num_features(self) -> int:
        return self.num_features
    

class TwoDimRepresentation(StateRepresentation):
    #2D representation of the state as a grid position each tile

    def __init__(self, size = 16, response_func = None, response_func_inv = None):
        self.size = size
        self.dim = int(math.sqrt(size))
        self.num_features = self.dim * self.dim * (self.dim + self.dim)
        self.response_func = response_func if response_func else lambda x:x
        self.response_func_inv = response_func_inv if response_func_inv else lambda x:x

    def get_features(self, state:State) -> np.ndarray:
        #convert state to 2D feature representation
        features = np.zeros(self.num_features)

        count = 0
        for tile in range(self.size):
            try: 
                pos = np.where(state.arr == tile)[0][0] #find position of tile i
            except IndexError:
                pos = 0     #set tile position to 0 if tile not found

            #convert position to row and col
            row = pos // self.dim
            col = pos % self.dim

            #set features using one hot encoding
            if count + row < len(features):
                features[count + row] = 1 
            if count + self.dim + col < len(features):
                features[count + self.dim + col] = 1

            count += self.dim + self.dim

        return features

    def get_num_features(self) -> int:
        return self.num_features
    
class TrainingInstance:
    #Represents a training instance with state and cost to go
    
    def __init__(self, state:State, response:float):
        self.state = state
        self.response = response #cost-to-go

    def __eq__(self, otherTrainingInstance):
        if not isinstance(otherTrainingInstance, TrainingInstance):
            return False
        return (np.array_equal(self.state.arr, otherTrainingInstance.state.arr) and abs(self.response - otherTrainingInstance.response) < 1e-6)
    
    def __hash__(self):
        return hash((tuple(self.state.arr), self.response)) 