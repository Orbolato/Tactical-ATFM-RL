import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import math

class RunwayEnv(gym.Env):
    """
    Runway Configuration and Separation Optimization Environment
    
    This environment simulates the optimization of runway configurations, arrival/departure
    processing, and separation standards at an airport.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        alpha: float,                  # Weight for arrival queue penalty
        beta: float,                   # Weight for separation penalty
        delta: float,                  # Weight for configuration switch penalty
        data,                          # DataFrame with historical data
        training_days: List[int],      # List of days available for training
        u: np.ndarray,                 # Arrival capacity matrix (weather_condition, runway_config)
        u_max: int,                    # Maximum arrival capacity
        v: np.ndarray,                 # Departure capacity matrix (weather_condition, runway_config)
        v_max: int,                    # Maximum departure capacity
        possible_actions: np.ndarray,  # Array of all possible actions
        valid_bool: np.ndarray,        # Boolean mask for valid actions
        sep_min: int, # Minimum separation
        sep_max: int, # Maximum separation
        sep_step: int, # Separation step
        wind_states: List[List[int]],  # Available runway configurations for each wind state
        observation_low: List[int],  # Lower bounds for observation space
        observation_high: List[int], # Upper bounds for observation space
        render_mode=None
    ):
        # Store environment parameters
        self.data = data
        self.training_days = training_days
        self.u = u
        self.u_max = u_max
        self.v = v
        self.v_max = v_max
        self.possible_actions = possible_actions
        self.valid_bool = valid_bool
        self.sep_min = sep_min
        self.sep_max = sep_max
        self.sep_step = sep_step
        self.sep_range = self.sep_max - self.sep_min
        self.wind_states = wind_states
        self.ema_factor = 0.1
        self.sep_scaling = 1
        
        # For storing results
        self.results = {}
        
        # Reward function parameters
        self.alpha = alpha  # Weight for arrival queue
        self.beta = beta    # Weight for separation
        self.delta = delta  # Weight for configuration switch
        self.mean_f_100 = np.mean(self.data['arrival_demand_100']) # Mean flow at 100 NM
        self.max_f_100 = np.max(self.data['arrival_demand_100']) # Max flow at 100 NM
        self.std_f_100 = np.std(self.data['arrival_demand_100']) # Standard deviation of flow at 100 NM
        self.mean_arr = np.mean(self.data['arrival_demand_40']) # Mean flow at 40 NM
        self.max_arr = np.max(self.data['arrival_demand_40']) # Max flow at 40 NM
        self.mean_dep = np.mean(self.data['departure_demand']) # Mean departures
        self.max_dep = np.max(self.data['departure_demand']) # Max departures

        # Define observation space
        observation_low = np.array(observation_low, dtype=np.float32)
        observation_high = np.array(observation_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high)

        # Define action space
        self.action_space = spaces.Discrete(len(self.possible_actions))
        
        # Initialize rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Initialize state variables
        self.day = None
        self.day_data = None
        self.time_step = None
        self.obs = None

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.obs
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            'day': self.day,                  # Current day being analyzed
            'runway_configuration': self.obs[0],  # Current runway configuration
            'arrival_queue': self.obs[1],     # Current arrival queue
            'departure_queue': self.obs[2],   # Current departure queue
            'weather_condition': self.obs[3], # Current weather condition
            'wind_state': self.obs[4],        # Current wind state
            'separation': self.obs[5],        # Current separation
            'flow_100': self.obs[6]             # Current flow at 100 NM
        }
    
    def compute_reward(self, 
                      RC_prev: int, 
                      RC_t: int, 
                      a_prev: int, 
                      d_prev: int, 
                      a_t: int, 
                      d_t: int,
                      f_t: int,
                      s_t: int,
                      training: bool,
                      evaluate: bool,
                      ) -> float:
        """
        Compute the reward for taking an action from the current state.
        
        Args:
            RC_prev: Previous runway configuration
            RC_t: New runway configuration
            a_prev: Previous arrival queue
            d_prev: Previous departure queue
            a_t: New arrival queue
            d_t: New departure queue
            f_t: Arrival demand
            s_t: New separation values
            
        Returns:
            float: Calculated reward
        """
        # Determine if runway configuration changed
        config_switch = int(RC_t != RC_prev)
        self.config_switch.append(config_switch)
        
        if not evaluate:
            
            # Calculate queue penalties (negative reward based on squared queue lengths)
            queue_penalty = -(self.alpha * a_t**2 + d_t**2)
            self.queue_cost -= queue_penalty
            
            # Adjusting scale for separation terms
            if training:
                self.sep_scaling = max(self.ema_factor * abs(queue_penalty) + (1 - self.ema_factor) * self.sep_scaling, 1)
            
            # Calculate switch penalty if configuration changed
            # Only penalize switches when there are aircraft in the queue
            switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2) 
            self.switch_cost -= switch_penalty
            
            # Calculate separation penalty (deviation from reference separation)
            flow_factor = self.sep_min / s_t 
            flow = int(f_t * flow_factor)
            overflow = f_t - flow
            sep_penalty = -self.beta * overflow
            self.sep_cost -= sep_penalty
        
        else:
            
            # Calculate queue penalties (negative reward based on squared queue lengths)
            queue_penalty = -(self.alpha * a_t**2 + d_t**2)
            self.queue_cost -= queue_penalty
            
            # Adjusting scale for separation terms
            if training:
                self.sep_scaling = max(self.ema_factor * abs(queue_penalty) + (1 - self.ema_factor) * self.sep_scaling, 1)
            
            # Calculate switch penalty if configuration changed
            # Only penalize switches when there are aircraft in the queue
            switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2) 
            self.switch_cost -= switch_penalty
            
            # Calculate separation penalty (deviation from reference separation)
            flow_factor = self.sep_min / s_t 
            flow = int(f_t * flow_factor)
            overflow = f_t - flow
            sep_penalty = -self.beta * overflow
            self.sep_cost -= sep_penalty
        
        return queue_penalty + switch_penalty + sep_penalty
    
    def reset(self, seed=None, options=None, day=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (not used)
            day: Specific day to use (if None, randomly select)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Seed the random number generator
        super().reset(seed=seed)
        
        # Reset time step
        self.time_step = 0
        
        # Select day to start (random or specified)
        if day is None:
            self.day = np.random.choice(self.training_days)
        else:
            self.day = day
            
        # Get data for selected day
        self.day_data = self.data.loc[self.data['day'] == self.day,:].reset_index(drop=True)
        
        # Set initial observation
        self.obs = np.array([
            0,  # Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            self.day_data.loc[self.time_step, 'arrival_demand_40'],      # Arrival queue
            self.day_data.loc[self.time_step, 'departure_demand'],       # Departure queue
            self.day_data.loc[self.time_step, 'weather_condition'],      # Weather condition (0 = VMC, 1 = IMC, 2 = LIMC)
            self.day_data.loc[self.time_step, 'wind_state'],             # Wind state
            self.sep_min,                                                # Separation
            self.day_data.loc[self.time_step, 'arrival_demand_100']     # Flow through the 100-NM cylinder
        ], dtype=np.int32)
        self.prev_sep = self.sep_min
        self.prev_prev_sep = self.prev_sep
        self.configs = []
        self.config_switch = []
        self.arrival_queue = []
        self.departure_queue = []
        self.arrivals_served = []
        self.departures_served = []
        self.separations = []
        self.objective = 0
        self.queue_cost = 0
        self.switch_cost = 0
        self.sep_cost = 0
        self.sep_mean = self.sep_min
        self.da_t = 0
        
        # Get state representation
        observation = self._get_obs()
        info = self._get_info()

        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action_index: int, training=True, evaluate=False) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action in environment and move to next state.
        
        Args:
            action_index: Index of the action to take
        Returns:
            observation: New observation
            reward: Reward for the transition
            terminated: Whether episode is done
            truncated: Whether episode was truncated (always False)
            info: Additional information
        """
        # Get selected action from action index
        action = self.possible_actions[action_index]
        
        # Extract current state variables
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Current arrival queue
        d_i = int(self.obs[2])   # Current departure queue
        self.prev_prev_sep = self.prev_sep # Second to last separation
        self.prev_sep = int(self.obs[5]) # Last but one separation
        
        # Extract action components
        RC_f = action[0]         # New runway configuration
        a_served = action[1]  # Number of arrivals to process
        d_served = action[2]  # Number of departures to process
        s_t = self.prev_sep + self.sep_step * (action[3] - 1) # New last separation values
        
        # Action smoothing strategy
        if (s_t - 2*self.prev_sep + self.prev_prev_sep) * (s_t - self.prev_sep) * (self.prev_sep - self.prev_prev_sep) != 0:
            s_t = self.prev_sep
        
        # Calculate new queue lengths after processing
        a_f = a_i - a_served  # New arrival queue
        d_f = d_i - d_served  # New departure queue
        
        # Ensure queues don't go negative (sanity check)
        if a_f < 0 or d_f < 0:
            print('Invalid queues detected.\n')
        
        # Ensure separation between bounds (sanity check)
        if s_t < self.sep_min or s_t > self.sep_max:
            print('Invalid separation detected.\n')
            
        # Get flow through fixes at 100 NM for current time step
        f_100_t = self.day_data.loc[self.time_step, 'arrival_demand_100']
        
        # Calculate reward
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, f_100_t, s_t, training, evaluate)
        
        # Storing results
        self.configs.append(RC_f)
        self.arrival_queue.append(a_f)
        self.departure_queue.append(d_f)
        self.arrivals_served.append(a_served)
        self.departures_served.append(d_served)
        self.separations.append(s_t)
        self.objective -= reward
        
        # Check if episode is done
        terminated = self.time_step == len(self.day_data) - 1
        
        if not terminated:
            # Move to next time step
            self.time_step += 1
            
            # Get flow through fixes at 40 NM for next time step
            f_40_t = self.day_data.loc[self.time_step, 'arrival_demand_40']
            
            # Calculate flow factor (how separation affects actual arrivals)
            # Use safe division to avoid issues
            flow_factor = self.sep_min / s_t 
            flow = int(f_40_t * flow_factor)
            overflow = f_40_t - flow
            
            # Calculate new queue sizes based on separation adjustments and new demand
            # Arrivals: previous queue + new demand (affected by separation)
            a_i = a_f + flow
            if self.time_step + 1 < len(self.day_data):
                self.day_data.loc[self.time_step + 1, 'arrival_demand_40'] += overflow
            d_i = d_f + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Update state for next time step
            self.obs = np.array([
                RC_f,  # New runway configuration
                a_i,   # Updated arrival queue
                d_i,   # Updated departure queue
                self.day_data.loc[self.time_step, 'weather_condition'], # Weather condition
                self.day_data.loc[self.time_step, 'wind_state'], # Wind state
                s_t,  # Separation
                self.day_data.loc[self.time_step, 'arrival_demand_100']     # Flow through the 100-NM cylinder
            ], dtype=np.int32)
            self.sep_mean = (self.time_step*self.sep_mean + s_t) / (self.time_step + 1)
            
        else:
            # Compile solution
            solution = {
                'objective': self.objective,
                'queue_cost': self.queue_cost,
                'switch_cost': self.switch_cost,
                'sep_cost': self.sep_cost,
                'configs': self.configs,
                'arrival_queue': self.arrival_queue,
                'departure_queue': self.departure_queue,
                'arrivals_served': self.arrivals_served,
                'departures_served': self.departures_served,
                'separations': self.separations
            }
            self.results[self.day] = solution
        
        # Get final observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def set_of_actions(self) -> np.ndarray:
        """
        Get the set of valid actions for the current state.
        
        Returns:
            Array of valid actions
        """
        # Extract current state components
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        separation = int(self.obs[5])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Safety check for queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
        
        # Safety check for separation
        if separation < self.sep_min or separation > self.sep_max:
            raise ValueError('Invalid separation value detected.')
            
        # Return valid actions using precomputed boolean mask
        return self.possible_actions[self.valid_bool[
            wind_state, 
            weather_condition,
            arrivals_queue, 
            departures_queue,
            separation
        ]]
    
    def action_masks(self) -> np.ndarray:
        """
        Get the action mask for the current state.
        
        Returns:
            Boolean array indicating which actions are valid
        """
        # Extract current state components
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        separation = int(self.obs[5])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Safety check for queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
            
        # Safety check for separation
        if separation < self.sep_min or separation > self.sep_max:
            raise ValueError('Invalid separation value detected.')
        
        # Return boolean mask for valid actions
        return self.valid_bool[
            wind_state, 
            weather_condition,
            arrivals_queue, 
            departures_queue,
            separation
        ]
    
    # def improve_action(self, s_t_mean: int, f_t: np.ndarray, s_prev: np.ndarray) -> np.ndarray:
    #     """
    #     Apply a heuristic to distribute the separation through the fixes.
        
    #     Returns:
    #         s_t_mean: Mean separation
    #         f_t: Flow through the fixes
    #         s_prev: Previous separations
    #     """
    #     # Create a concrete model
    #     model = pyo.ConcreteModel()
        
    #     # Define the indices for the 4 elements
    #     model.I = pyo.RangeSet(0, 3)
        
    #     # Define possible values for s_t
    #     allowed_values = [5,10,15,20,25,30]
        
    #     # Define variables
    #     model.s_t = pyo.Var(model.I, domain=pyo.Integers, bounds=(5, 30))
        
    #     # Binary variables to enforce discrete values
    #     model.y = pyo.Var(model.I, range(len(allowed_values)), domain=pyo.Binary)
        
    #     # Add auxiliary variable to represent the maximum
    #     # model.max_val = pyo.Var(domain=pyo.Integers)
        
    #     # Add auxiliary variable to represent the mean cost
    #     model.mean_cost = pyo.Var(domain=pyo.Integers)
        
    #     # Objective function: minimize dot product
    #     model.objective = pyo.Objective(
    #         expr=sum((f_t[i] * model.s_t[i] - model.mean_cost)**2 for i in model.I),
    #         sense=pyo.minimize
    #     )
        
    #     # Objective function: minimize the maximum value (instead of using max() directly)
    #     # model.objective = pyo.Objective(
    #     #     expr=model.max_val,
    #     #     sense=pyo.minimize
    #     # )
        
    #     # Constraints to define max_val as the maximum of f_t[i] * s_t[i]
    #     # model.max_constraints = pyo.ConstraintList()
    #     # for i in model.I:
    #     #     model.max_constraints.add(model.max_val >= f_t[i] * model.s_t[i])
        
    #     # Constraints to define mean_cost as the mean of f_t[i] * s_t[i]
    #     model.var_constraint = pyo.Constraint(
    #         expr=sum(f_t[i] * model.s_t[i] for i in model.I) == model.mean_cost * len(f_t)
    #     )
        
    #     # Constraints
        
    #     # 1. Mean constraint
    #     model.mean_constraint = pyo.Constraint(
    #         expr=sum(f_t[i] * model.s_t[i] for i in model.I) <= sum(f_t) * s_t_mean
    #     )
        
    #     # 2. Special constraint: np.sum(f_t) / s_t_mean = np.sum(f_t / s_t)
    #     model.special_constraint = pyo.Constraint(
    #         expr=sum(f_t[i] / model.s_t[i] for i in model.I) <= sum(f_t) / s_t_mean 
    #     )
        
    #     # 3. Enforce that each s_t[i] takes a value from the allowed set
    #     # Create constraint dictionaries
    #     model.value_selection = pyo.ConstraintList()
    #     model.value_assignment = pyo.ConstraintList()
        
    #     for i in model.I:
    #         # Each variable must take exactly one of the allowed values
    #         model.value_selection.add(
    #             sum(model.y[i, j] for j in range(len(allowed_values))) == 1
    #         )
            
    #         # Link the binary variables to the actual values
    #         model.value_assignment.add(
    #             model.s_t[i] == sum(allowed_values[j] * model.y[i, j] for j in range(len(allowed_values)))
    #         )
        
    #     # Solve the model
    #     solver = SolverFactory('ipopt')  # You might need to change the solver based on availability
        
    #     # Set solver options to increase maximum iterations
    #     # solver.options['max_iter'] = 3000  # Default is usually 1000
    #     # solver.options['print_level'] = 5  # Optional: increase output verbosity
    #     solver.options['tol'] = 1e-6
        
    #     result = solver.solve(model, tee=False)
        
    #     # Check if solution was found
    #     if result.solver.status == pyo.SolverStatus.ok:
    #         # Extract solution
    #         s_t_optimal = np.array(np.round([pyo.value(model.s_t[i]) for i in model.I]), dtype=np.int32)
    #         obj_value = pyo.value(model.objective)
    #         # print(s_t_optimal)
    #         for i in range(len(f_t)):
    #             if f_t[i] == 0:
    #                 s_t_optimal[i] = s_t_mean
    #         return s_t_optimal, obj_value
    #     else:
    #         # print("Solver did not find a solution.")
    #         return np.array([s_t_mean for i in range(len(f_t))], dtype=np.int32), None
    
    def naive_action(self, evaluate=False) -> int:
        """
        Determine a naive greedy action based on immediate rewards.
        
        Returns:
            action_index: Index of the selected action
        """
        # Get current state information
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Current arrival queue
        d_i = int(self.obs[2])   # Current departure queue
        s_prev = self.obs[5]  # Current separation values
        self.prev_prev_sep = self.prev_sep
        self.prev_sep = s_prev
        
        # Get set of valid actions for current state
        valid_actions = self.set_of_actions()
        
        if len(valid_actions) == 0:
            # If no valid actions (shouldn't happen with proper masking),
            # return a safe default action
            return 0
        
        # Calculate rewards for each valid action
        rewards = []
        for action in valid_actions:
            RC_f = action[0]
            a_f = a_i - action[1]
            d_f = d_i - action[2]
            s_t = s_prev + self.sep_step * (action[3] - 1)
            f_100_t = self.day_data.loc[self.time_step, 'arrival_demand_100']
            rewards.append(self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, f_100_t, s_t, False, evaluate))
        
        # Select action with highest reward
        greedy_action = valid_actions[np.argmax(rewards)]
        
        # Find index of the greedy action in the possible actions list
        for i, action in enumerate(self.possible_actions):
            if np.array_equal(action, greedy_action):
                return i
                
        # Fallback if action not found (shouldn't happen)
        return 0
    
    def base_action(self) -> np.ndarray:
        """
        Get the baseline action based on historical data.
        
        Returns:
            The baseline action
        """
        # Extract current queues
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        
        # Get baseline values from historical data
        rc = self.day_data.loc[self.time_step, 'base_configuration']
        arrivals = self.day_data.loc[self.time_step, 'base_arrivals_40']
        departures = self.day_data.loc[self.time_step, 'base_departures']
        
        # Limit by available queues
        if arrivals > arrivals_queue:
            # Can't serve more than what's in queue
            overflow = arrivals - arrivals_queue
            arrivals = arrivals_queue
            
            # Apply overflow to next time step if not at end
            if self.time_step < len(self.day_data) - 1:
                self.day_data.loc[self.time_step + 1, 'base_arrivals_40'] += overflow
        
        if departures > departures_queue:
            # Can't serve more than what's in queue
            overflow = departures - departures_queue
            departures = departures_queue
            
            # Apply overflow to next time step if not at end
            if self.time_step < len(self.day_data) - 1:
                self.day_data.loc[self.time_step + 1, 'base_departures'] += overflow
        
        # Construct the baseline action - note using float64 array instead of int64
        base_action = np.array([
            rc,                                                 # Runway configuration
            arrivals,                                           # Arrivals to serve
            departures,                                         # Departures to serve
            1                                        # Keep separation steady
        ], dtype=np.int32)
        
        return base_action
    
    def base_step(self, action: np.ndarray, evaluate=False) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step using the baseline action.
        
        Args:
            action: The baseline action to take
            
        Returns:
            Same as step() method
        """
        # Extract current state variables
        RC_i = int(self.obs[0])  # Previous runway configuration
        a_i = int(self.obs[1])   # Current arrival queue
        d_i = int(self.obs[2])   # Current departure queue
        s_prev = int(self.obs[5]) # Current separation values
        self.prev_prev_sep = self.prev_sep
        self.prev_sep = s_prev
        
        # Extract action components
        RC_f = action[0]         # New runway configuration
        a_served = action[1]  # Number of arrivals to process
        d_served = action[2]  # Number of departures to process
        s_t = s_prev + self.sep_step * (action[3] - 1)  # New separation values
        
        # Calculate new queue lengths after processing
        a_f = a_i - a_served  # New arrival queue
        d_f = d_i - d_served  # New departure queue
        
        # Ensure queues don't go negative (sanity check)
        if a_f < 0 or d_f < 0:
            print('Invalid queues detected.\n')
        a_f = max(0, a_f)
        d_f = max(0, d_f)
        
        # Ensure separation between bounds (sanity check)
        if s_t < self.sep_min or s_t > self.sep_max:
            print('Invalid separation detected.\n')
        s_t = max(s_t, self.sep_min)
        s_t = min(s_t, self.sep_max)
        
        # Calculate reward
        f_100_t = self.day_data.loc[self.time_step, 'arrival_demand_100']
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, f_100_t, s_t, False, evaluate)
        
        # Check if episode is done
        terminated = self.time_step == len(self.day_data) - 1
        
        if not terminated:
            # Move to next time step
            self.time_step += 1
            
            # Get flow through fixes at 40 NM for next time step
            f_40_t = self.day_data.loc[self.time_step, 'arrival_demand_40']

            # Calculate flow factor (how separation affects actual arrivals)
            # Use safe division to avoid issues
            flow_factor = self.sep_min / s_t
            flow = int(f_40_t * flow_factor)
            overflow = f_40_t - flow
            
            # Calculate new queue sizes based on separation adjustments and new demand
            # Arrivals: previous queue + new demand (affected by separation)
            a_i = a_f + flow    
            if self.time_step + 1 < len(self.day_data):
                self.day_data.loc[self.time_step + 1, 'arrival_demand_40'] += overflow
            d_i = d_f + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Update state for next time step
            self.obs = np.array([
                RC_f,  # New runway configuration
                a_i,   # Updated arrival queue
                d_i,   # Updated departure queue
                self.day_data.loc[self.time_step, 'weather_condition'],
                self.day_data.loc[self.time_step, 'wind_state'],
                s_t,
                self.day_data.loc[self.time_step, 'arrival_demand_100']
            ], dtype=np.int32)
        
        # Get final observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def compare_to_baseline(self, day: int) -> Dict[str, Any]:
        """
        Compare optimization results to baseline for a specific day.
        
        Args:
            day: Day index to compare
            
        Returns:
            Dictionary with comparison metrics
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get optimization results
        opt_results = self.results[day]
        
        # Get baseline data
        day_data = self.data.loc[self.data['day'] == day].reset_index(drop=True)
        
        # Check if baseline columns exist
        if 'cap_config' in day_data.columns:
            base_configs = day_data['cap_config'].tolist()
            base_arrivals = day_data['cap_arrivals'].tolist()
            base_departures = day_data['cap_departures'].tolist()
        else:
            # Use greedy baseline if base doesn't exist
            base_configs = day_data.get('base_configuration', [0] * len(day_data)).tolist()
            base_arrivals = day_data.get('base_arrivals_40', [0] * len(day_data)).tolist()
            base_departures = day_data.get('base_departures', [0] * len(day_data)).tolist()
        
        base_separations = [self.sep_min] * len(day_data)  # Assume baseline uses minimum separation
        
        # Simulate baseline performance
        base_arrival_queue = [day_data.loc[0, 'arrival_demand_40']]
        base_departure_queue = [day_data.loc[0, 'departure_demand']]
        base_config_switches = []
        
        prev_config = 0  # Initial config
        prev_separation = self.sep_min
        base_total_cost = 0.0
        
        for t in range(len(day_data)):
            # Check for config switch
            curr_config = base_configs[t]
            curr_separation = base_separations[t]
            config_switch = int(curr_config != prev_config)
            base_config_switches.append(config_switch)
            
            # Process arrivals and departures
            arr_served = min(base_arrivals[t], base_arrival_queue[-1])
            dep_served = min(base_departures[t], base_departure_queue[-1])
            
            # Calculate queues after service
            arr_queue_after = base_arrival_queue[-1] - arr_served
            dep_queue_after = base_departure_queue[-1] - dep_served
            
            # Calculate cost for this time step
            queue_cost = self.alpha * arr_queue_after**2 + dep_queue_after**2
            switch_cost = config_switch * self.delta * (self.alpha * base_arrival_queue[-1]**2 + base_departure_queue[-1]**2)
            sep_cost = self.beta * (curr_separation - self.sep_min)
            
            step_cost = queue_cost + switch_cost + sep_cost
            base_total_cost += step_cost
            
            # Calculate new queues
            if t < len(day_data) - 1:
                # Apply separation effect
                flow_factor = self.sep_min / curr_separation
                next_arr_demand = int(day_data.loc[t+1, 'arrival_demand_40'] * flow_factor)
                
                new_arr_queue = arr_queue_after + next_arr_demand
                new_dep_queue = dep_queue_after + day_data.loc[t+1, 'departure_demand']
                base_arrival_queue.append(new_arr_queue)
                base_departure_queue.append(new_dep_queue)
            
            prev_config = curr_config
            prev_separation = curr_separation
        
        # Compare metrics
        comparison = {
            'day': day,
            'opt_objective': opt_results['objective'],
            'base_objective': base_total_cost,
            'improvement': (base_total_cost - opt_results['objective']) / abs(base_total_cost) * 100 if base_total_cost != 0 else 0,
            'opt_configs': opt_results['configs'],
            'base_configs': base_configs,
            'opt_separations': opt_results['separations'],
            'base_separations': base_separations,
            'opt_arr_queue': opt_results['arrival_queue'],
            'base_arr_queue': base_arrival_queue,
            'opt_dep_queue': opt_results['departure_queue'],
            'base_dep_queue': base_departure_queue
        }
        
        return comparison
    
    def visualize_results(self, day: int, figsize=(15, 12)):
        """
        Visualize optimization results for a specific day.
        
        Args:
            day: Day index to visualize
            figsize: Figure size for the plot
        """
        if day not in self.results:
            raise ValueError(f"Day {day} has not been optimized yet")
        
        # Get results
        comparison = self.compare_to_baseline(day)
        results = self.results[day]
        
        # Create figure
        fig, axs = plt.subplots(5, 1, figsize=figsize)
        
        # Time steps
        time_steps = range(len(comparison['opt_configs']))
        n_time_steps = len(time_steps)
        
        # Plot runway configurations
        axs[0].plot(time_steps, comparison['opt_configs'], 'b-', label='DCB')
        axs[0].plot(time_steps, comparison['base_configs'], 'r--', label='CM')
        axs[0].set_ylabel('Runway Config')
        axs[0].set_title(f'Day {day} - Comparison (Improvement: {comparison["improvement"]:.2f}%)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot separations
        axs[1].plot(time_steps, comparison['opt_separations'], 'b-', label='DCB')
        axs[1].plot(time_steps, comparison['base_separations'], 'r--', label='CM')
        axs[1].set_ylabel('Separation')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot arrival queues - ensure we only plot the same number of points
        axs[2].plot(time_steps, comparison['opt_arr_queue'][:n_time_steps], 'b-', label='DCB')
        axs[2].plot(time_steps, comparison['base_arr_queue'][:n_time_steps], 'r--', label='CM')
        axs[2].set_ylabel('Arrival Queue')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot departure queues - ensure we only plot the same number of points
        axs[3].plot(time_steps, comparison['opt_dep_queue'][:n_time_steps], 'b-', label='DCB')
        axs[3].plot(time_steps, comparison['base_dep_queue'][:n_time_steps], 'r--', label='CM')
        axs[3].set_ylabel('Departure Queue')
        axs[3].legend()
        axs[3].grid(True)
        
        # Plot arrivals and departures served
        axs[4].plot(time_steps, results['arrivals_served'], 'b-', label='Arrivals Served')
        axs[4].plot(time_steps, results['departures_served'], 'g-', label='Departures Served')
        axs[4].set_xlabel('Time Step')
        axs[4].set_ylabel('Aircraft Served')
        axs[4].legend()
        axs[4].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"Day {day} Summary:")
        print(f"  Objective: DCB = {comparison['opt_objective']:.2f}, CM = {comparison['base_objective']:.2f}")
        print(f"  Improvement: {comparison['improvement']:.2f}%")
        print("  Cost Breakdown:")
        print(f"    Queue Cost: {results['queue_cost']:.2f}")
        print(f"    Switch Cost: {results['switch_cost']:.2f}")
        print(f"    Separation Cost: {results['sep_cost']:.2f}")
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame."""
        # Placeholder for rendering implementation
        pass
