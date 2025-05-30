import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import math
import time
import gc
import random
import matplotlib.pyplot as plt
from collections import deque
from abc import abstractmethod
from typing import Tuple, Optional, List
from StateRepresentation import StateRepresentation, FeaturesRepresentation, OneDimRepresentation, TwoDimRepresentation, TrainingInstance
from domain import HammingHeuristic, ManhattanDistanceHeuristic, PDBHeuristic, MultiHeuristic, State, Operation, Edge, UndoToken, IHeuristic, SlidingPuzzle 

class BayesianLinearLayer(nn.Module):

    #This is the linear layer for the WUNN. 
    #instead of having fixed weights, each weight is a distibution

    def __init__(self, in_features, out_features, prior_mean = 0.0, prior_std = 10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mean
        self.prior_sigma = prior_std

        #set the weight and bias parameters: mu and sigma
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):

        #initialise the parameters
        init_factor = math.sqrt(2.0/self.in_features)
        self.weight_mu.data.normal_(0, init_factor *0.1)

        #start with low variance = high confidence level --> only update when necessary
        self.weight_rho.data.fill_(-5.0)

        #initialise the bias
        self.bias_mu.data.zero_()
        self.bias_rho.data.fill_(-5.0)

    def forward(self, x, sample = True):
        
        if sample:
            #sample the weights and bias from the distributions
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

            weight_epsilon = torch.randn_like(weight_sigma)
            bias_epsilon = torch.randn_like(bias_sigma)


            weight = self.weight_mu + weight_sigma * weight_epsilon
            bias = self.bias_mu + bias_sigma * bias_epsilon

        else:
            #use the mean of the distributions
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    

    def kl_divergence(self):
        #Calculate the KL divergence bwtween the prior and the posterior distributions
        # KL(q||p) = Integral q(w) log(q(w))/p(w) dw
        

        #convert rho to sigma
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        #KL for weight and biases
        weight_kl = self._kl_divergence_calc(self.weight_mu, weight_sigma, self.prior_mu, self.prior_sigma)
        bias_kl = self._kl_divergence_calc(self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma)

        total_kl = weight_kl + bias_kl

        return total_kl
    
    def _kl_divergence_calc(self, mu_q, sigma_q, mu_p, sigma_p):
        #calculates the KL divergence between two gaussian distributions
        #For Gaussians: KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

        sigma_q = torch.clamp(sigma_q, min = 1e-8)
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p**2)/ (2*sigma_p**2) - 0.5)

        return kl.sum()
    

class WeightUncertaintyNN(nn.Module):
    #This is the neural network that models epistemic uncertainty
    #it uses BayesianLinearLayer instead of nn.Linear 

    def __init__(self, input_size, hidden_size, output_size = 1):
        super().__init__()

        #define the layers using BayesianLinearLayer
        self.fc1 = BayesianLinearLayer(input_size, hidden_size)
        self.fc2 = BayesianLinearLayer(hidden_size, output_size)

        self.num_samples = 100 #number of monte carlo samples collected for uncertainty estimation

    def forward(self, x, sample = True):
        #forward pass through the network
        x = F.relu(self.fc1(x, sample))
        x = self.fc2(x, sample)

        return x

    def make_prediction_with_uncertainty(self, x, num_samples = None):
        #make predictions utilising uncertainty

        if num_samples is None:
            num_samples = self.num_samples

        self.eval() #set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                #each foward pass uses different sampled weights
                pred = self.forward(x, sample = True)
                predictions.append(pred)

        #stach predictions and calculate statistics
        predictions = torch.stack(predictions)  #shape[num_samples, batch_size, output_size]

        #calculate mean and variance across samples
        mean = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)

        return mean, epistemic_uncertainty
    
    def kl_divergence(self):
        #total kl divergence for the network
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()
                           



class StandardNN(nn.Module):
    #this network models Aleatoric Uncertainty -  uncertainty about the data itself.
    #it uses the standard nn.Linear to create the layers and has two outputs: predicted mean and variance(aleatoric uncertainty)
    #the loss function accounts for both the prediction error and uncertainty

    def __init__(self, input_size, hidden_size, output_size = 2):
        super().__init__()

        #define the layers using nn.Linear
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        #initialising the weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        #foward pass through the network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if x.shape[-1] == 2:
            #split the output into mean and variance
            mean = x[..., 0] #first output is the mean
            log_var = x[..., 1] #second output is the log variance
            #apply softplus to ensure variance is positive
            variance = F.softplus(log_var) +1e-6
            return mean, variance
        else:
            return x

    # def make_prediction_with_uncertainty(self, x):
    #     #making predictions with aleatoric uncertainty
    #     if self.fc2.outfeatures == 2:
    #         mean, aleatoric_variance = self.foward(x, verbose)
    #         return mean, aleatoric_variance
    #     else:
    #         pred = self.foward(x, verbose)
    #         return pred, torch.zeros_like(pred) #if no variance output, return x and 0 for variance
   
        
    # def loss_function(self,x,y_true, verbose = False):
    #     #loss function that accounts for both prediction error and aleatoric uncertainty
    #     # L = 0.5*log(sigma^2) + 0.5 * (y - mu)^2 / sigma^2

    #     if self.fc2.out_features == 2:
    #         mean, variance = self.forward(x)

    #         loss = 0.5 * torch.log(variance) + 0.5 * ((y_true - mean)**2)/variance

    #         if verbose:
    #             print(f'loss: {loss.item()}')
    #             return loss
    #         else:
    #             #standard MSE loss
    #             return loss.mean()


class NeuralHeuristic():
   #Base class for neural network-based heuristics

    def __init__(self, representation: StateRepresentation):
       self.representation = representation
       self.cache = {}
       self.max_cache_size = 100000
    
    def clear_cache(self):
       #clear heuristic cache
        self.cache.clear()
        gc.collect() #force garbage collection like c#
    
    def h(self, state, verbose: bool = False) -> int:
        state_hash = hash(tuple(state.arr))

        if state_hash in self.cache:
            return self.cache[state_hash]
        
        try:
            #get features and predict
            features = self.representation.get_features(state)
            h_val = self._predict(features)

            #endure non negative
            h_val = max(0, min(200, int(round(h_val))))
            
            #cache result
            if len(self.cache) < self.max_cache_size:
                self.cache[state_hash] = h_val
            elif len(self.cache) >= self.max_cache_size:
                self.clear_cache()
                self.cache[state_hash] = h_val
                
            return h_val
        except Exception as e:
            print(f"Heuristic prediction error: {e}")
            return 0 #safe default
    
    @abstractmethod
    def _predict(self, features:np.ndarray) -> float:
        #predict the heuristic value from features
        pass

class BayesianNeuralHeuristic(NeuralHeuristic):
    #Bayesian neural network heurist with uncertainty

    def __init__(self, representation: StateRepresentation, solve_nn:StandardNN, uncertainty_nn: WeightUncertaintyNN, confidence_level:Optional[float] = None, l2_loss: bool = True):
        super().__init__(representation)
        self.solve_nn = solve_nn
        self.uncertainty_nn = uncertainty_nn
        self.confidence_level = confidence_level
        self.l2_loss = l2_loss
        self.memory_buffer = []
        self.max_buffer_size = 25000
        self.is_trained = False

        #Adaptive parameters like c# implementation
        self.update_beta = True
        self.conf_level_threshold = 0.6 #percSolvedThres from C#
        self.conf_level_increment = 0.05 #incConfLevel from C#
        self._initialise_networks()

    def _initialise_networks(self):
        #Initialise networks with some reasonable starting behavior
        try:
            #Create some dummy training data to give networks a starting point
            dummy_states = []
            dummy_responses = []
            
            #Create a few simple states with known costs
            puzzle = SlidingPuzzle(size=len(self.representation.get_features(State(np.arange(16)))))
            goal = puzzle.goal()
            
            #Add goal state with cost 0
            dummy_states.append(goal)
            dummy_responses.append(0.0)
            
            #Add a few scrambled states with estimated costs
            for steps in [1, 2, 3, 5]:
                scrambled = puzzle._scramble(goal, steps)
                dummy_states.append(scrambled)
                dummy_responses.append(float(steps))
            
            #Convert to training instances
            training_instances = [TrainingInstance(state, response) 
                                for state, response in zip(dummy_states, dummy_responses)]
            
            self.memory_buffer = training_instances
            self._train_solve_network()
            
            print("Networks initialised with dummy data")
            
        except Exception as e:
            print(f"Network initialisation error: {e}")

    def _predict(self, features: np.ndarray) -> float:
        #predict with uncertainty
        try:
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                #get predictio from standard neural network
                if hasattr(self.solve_nn, 'forward') and self.solve_nn.fc2.out_features == 2:
                    mean, aleatoric_variance = self.solve_nn(features_tensor)
                    mean = float(mean)
                    aleatoric_std_dev = float(torch.sqrt(aleatoric_variance))

                else:
                    output = self.solve_nn(features_tensor)
                    mean = float(output)
                    aleatoric_std_dev = 1.0


                #if no confidence level specified, return mean only
                if self.confidence_level is None:
                    return max(0, mean)
                
                #apply confidence interval adjustment
                if self.l2_loss:
                    #using normal distribution
                    from scipy.stats import norm
                    adjusted_pred = norm.ppf(self.confidence_level, mean, aleatoric_std_dev) #z-score
                else:
                    #use Laplace distribution
                    adjusted_pred = self._inv_laplace(mean, aleatoric_std_dev, self.confidence_level)
                
                return adjusted_pred
            
        except Exception as e:
            print(f"Neural predictopn error: {e}")
            return 0.0 #safe default
        
    def _inv_laplace(self,mu:float, sigma:float, p:float) -> float:
        #inverse laplace CDF
        try:
            sign = 1 if p>0.5 else -1
            return mu - sigma * sign *math.log(1-2*abs(p-0.5))
        except:
            return mu #return mu if calculation fails
    
    def update(self, plans: List[List["State"]]):
        #updates the neural network with new training data
        try:
            #converts plans to training instances
            new_instances = []
            for plan in plans:
                for i, state in enumerate(plan[:-1]):  #exclude goal state
                    cost_to_goal = len(plan) -1 - i
                    new_instances.append(TrainingInstance(state, cost_to_goal))

            for instance in new_instances:
                if instance not in self.memory_buffer:
                    self.memory_buffer.append(instance)

            #trim buffer
            if len(self.memory_buffer) > self.max_buffer_size:
                self.memory_buffer = self.memory_buffer[-self.max_buffer_size:]
            if not self.memory_buffer:
                return

            #train networks
            self._train_solve_network()
            if self.uncertainty_nn is not None:
                self._train_uncertainty_network()

            self.is_trained = True

        except Exception as e:
            print(f"Update error:{e}")

    def _train_solve_network(self):
        #train the solve network
        if not self.memory_buffer:
            return
        
        try:
            #prepare training data
            X = []
            y = []

            for instance in self.memory_buffer:
                features = self.representation.get_features(instance.state)
                X.append(features)
                y.append(instance.response)

            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)

            #training loop
            optimizer = torch.optim.Adam(self.solve_nn.parameters(), lr=0.001)

            #training with timeout as in c#
            start_time = time.time()
            max_train_time  = 30.0 #30s max

            for epoch in range(1000):
                if time.time() - start_time > max_train_time:
                    print(f"Solve network training timeout at epocj {epoch}")
                    break

                optimizer.zero_grad()

                if hasattr(self.solve_nn, 'forward') and self.solve_nn.fc2.out_features == 2:
                    mean, variance = self.solve_nn(X)
                    
                    #negative log likelihood loss
                    loss = 0.5 * torch.log(variance) + 0.5 * (y - mean)**2/variance
                    loss = loss.mean()
                else:
                    X = self.solve_nn(X).squeeze()
                    loss = F.mse_loss(X, y)
                
                loss.backward()
                optimizer.step()

                #early stopping like c#
                if epoch % 100 == 0 and loss.item() < 0.5:
                    break

        except Exception as e:
            print(f"Solve network training error: {e}")
    
        

    def _train_uncertainty_network(self):
        #train the uncertainty WUNN network

        if not self.memory_buffer:
            return

        try:
            #prepare the training data
            X = []
            y = []

            for instance in self.memory_buffer:
                features = self.representation.get_features(instance.state)
                X.append(features)
                y.append(instance.response)

            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)

            #training loop with kl regularisation

            optimizer = torch.optim.Adam(self.uncertainty_nn.parameters(), lr=0.01)
            beta = 0.05 #kl weight

            start_time = time.time()
            max_train_time = 60.0 #1mins max

            for epoch in range(1000):
                if time.time() - start_time > max_train_time:
                    print(f"Uncertainty network training timeout at epoch {epoch}")
                    break
                #sample batch
                batch_size = min(50, len(X))
                indices =  torch.randint(0, len(X), (batch_size,))
                X_batch = X[indices]
                y_batch = y[indices]

                optimizer.zero_grad()

                #foward pass
                output = self.uncertainty_nn(X_batch)

                #likelihood loss
                likelihood_loss = F.mse_loss(output.squeeze(), y_batch)

                #kl divergence
                kl_loss = self.uncertainty_nn.kl_divergence() / len(X_batch)

                #total loss
                total_loss = likelihood_loss + beta * kl_loss

                total_loss.backward()
                optimizer.step()

                #check convergence (simplified)
                if epoch % 500 == 0:
                    with torch.no_grad():
                        _, epistemic_var = self.uncertainty_nn.make_prediction_with_uncertainty(X)
                        max_uncertainty = float(epistemic_var.sqrt().max())
                        if max_uncertainty < 1.0:
                            break
        except Exception as e:
            print(f"Uncertainty network training error: {e}")

    def adapt_confidence_level(self, solve_rate: float):
        #adapt confidence level like c# implementation
        if solve_rate < self.conf_level_threshold:
            self.update_beta = False

            if self.confidence_level is not None and self.confidence_level < 0.5:
                old_level = self.confidence_level
                self.confidence_level += self.conf_level_increment

                if self.confidence_level > 0.5:
                    self.confidence_level = 0.5
                
                print(f"Confidence level adapted: {old_level:.2f} -> {self.confidence_level:.2f}")
        else:
            self.update_beta = True


class TaskGenerator:
    #Generates training tasks using epistemic uncertainty

    def __init__(self, puzzle: 'SlidingPuzzle', uncertainty_nn: WeightUncertaintyNN, representation: StateRepresentation, epsilon: float = 1.0, max_steps: int = 1000):
        self.puzzle = puzzle
        self.uncertainty_nn = uncertainty_nn
        self.representation = representation
        self.epsilon = epsilon
        self.max_steps = max_steps

    def generate_task(self) -> 'State':
        #Generate a task using epistemic uncertainty (GenerateTaskAlgorithm)
        current_state = self.puzzle.goal()
        visited = set()
        prev_state = None

        start_time = time.time()
        max_time = 10.0 #10 second timeout for task generation

        for step in range(self.max_steps):
            #timeout protection
            if time.time() - start_time > max_time:
                print(f"Task generation timeout after {step} steps")
                break

            visited.add(hash(tuple(current_state.arr)))

            #Get possible previous states (reverse operations)
            ops = self.puzzle.operations(current_state)
            candidate_states = []
            uncertainties = []

            for op in ops:
                try: 
                    #Apply operation to get previous state
                    prev, _ = self.puzzle.apply(current_state, op)

                    #Skip if its the state we just came from
                    prev_hash = hash(tuple(prev.arr))
                    if prev_state is not None and prev_hash == hash(tuple(prev_state.arr)):
                        continue

                    #skip if already visited 
                    if prev_hash in visited:
                        continue

                    #calculate epistemic unceertainty
                    features = self.representation.get_features(prev)
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)

                    with torch.no_grad():
                        try:
                            _, epistemic_var = self.uncertainty_nn.make_prediction_with_uncertainty(features_tensor, num_samples = 5)
                            uncertainty = float(epistemic_var.sqrt())
                        except Exception as e:
                            print(f"Uncertainty calculation error: {e}")
                            uncertainty = 0.1 #default low uncertainty

                    candidate_states.append(prev)
                    uncertainties.append(uncertainty)
                except Exception as e:
                    print(f"Task generation error:{e}")
                    continue

            if not candidate_states:
                break

            #Sample from softmax distribution based on uncertainties
            if uncertainties:
                uncertainties = np.array(uncertainties)
                #apply softmax
                exp_uncertainties = np.exp(uncertainties - np.max(uncertainties))
                probabilities = exp_uncertainties / np.sum(exp_uncertainties)

                try:
                    #sample a state
                    chosen_idx = np.random.choice(len(candidate_states), p=probabilities)
                    chosen_state = candidate_states[chosen_idx]
                    chosen_uncertainty = uncertainties[chosen_idx]

                    #if uncertainty is above threshold, return this state
                    if chosen_uncertainty >= self.epsilon:
                        return chosen_state
                    
                    #otherwise, continue from from this state
                    prev_state = current_state
                    current_state = chosen_state
                except Exception as e:
                    print(f"Sampling error: {e}")
                    #fall back to random choice
                    chosen_state = random.choice(candidate_states)
                    prev_state = current_state
                    current_state = chosen_state

            else:
                break

        #Return current state if no high-uncertainty state found
        return current_state