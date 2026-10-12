import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable

class RunwayEnv(gym.Env):
    """
    Runway Configuration and Capacity Optimization Environment for Air Traffic Management.
    
    This environment simulates runway operations at an airport with different configurations,
    arrival/departure queues, and weather conditions. The agent's goal is to optimize
    the runway configuration and service rates to minimize delays while respecting
    operational constraints.
    
    State space:
        - Runway configuration (RC): 0=None, 1=10R_10L, 2=28L_28R
        - Arrival queue length
        - Departure queue length
        - Weather condition: 0=VMC, 1=IMC, 2=LIMC
        - Wind state: 0=Both avail, 1=10R_10L avail, 2=28L_28R avail, 3=Both unavail
    
    Action space:
        - Discrete actions representing different combinations of:
          (runway configuration, arrivals to serve, departures to serve)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        alpha: float,
        delta: float,
        data: Any,
        training_days: List[int],
        u: np.ndarray,
        u_max: int,
        v: np.ndarray,
        v_max: int,
        possible_actions: List[List[int]],
        valid_bool: np.ndarray,
        wind_states: List[List[int]],
        observation_low: List[float],
        observation_high: List[float],
        render_mode: Optional[str] = None
    ):
        """
        Initialize the runway environment.
        
        Args:
            alpha: Weight parameter for arrival queue in reward function
            delta: Weight parameter for configuration switches in reward function
            data: DataFrame containing weather and traffic data for simulation
            training_days: List of day indices available for training episodes
            u: Arrival capacity matrix indexed by [weather_condition, runway_config]
            v: Departure capacity matrix indexed by [weather_condition, runway_config, arrival_rate]
            possible_actions: List of all possible actions as [RC, arr_rate, dep_rate]
            valid_bool: Boolean mask for valid actions by [wind_state, weather_cond, arr_queue, dep_queue]
            wind_states: Available runway configurations for each wind state
            observation_low: Lower bounds for observation space
            observation_high: Upper bounds for observation space
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        super().__init__()
        
        # Environment parameters
        self.window_size = 512  # Size of PyGame window if rendering enabled
        
        # Store simulation data
        self.data = data
        self.training_days = training_days
        self.u = u
        self.u_max = u_max
        self.v = v
        self.v_max = v_max
        self.possible_actions = np.array(possible_actions)
        self.wind_states = wind_states
        self.valid_bool = valid_bool
        
        # Reward function parameters
        self.alpha = alpha  # Weight for arrival queue
        self.delta = delta  # Weight for configuration switches
        
        # Define observation space: (RC_(t-1), a_t, d_t, wc_t, ws_t)
        self.observation_space = spaces.Box(
            low=np.array(observation_low, dtype=np.float32),
            high=np.array(observation_high, dtype=np.float32)
        )
        
        # Define action space
        self.action_space = spaces.Discrete(len(self.possible_actions))
        
        # Verify render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Initialize rendering
        self.window = None
        self.clock = None
        
        # Episode state
        self.day = None
        self.day_data = None
        self.time_step = None
        self.episode_reward = None
        self.obs = None

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.obs
    
    def _get_action_mask(self) -> np.ndarray:
        """Get the action mask for the current state."""
        return self.action_masks()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary with action mask and episode stats."""
        info = {
            'action_mask': self._get_action_mask(),
        }
        
        # Add episode info if we're tracking rewards
        # if hasattr(self, 'episode_reward'):
        #     info['episode'] = {
        #         'r': self.episode_reward,
        #         'l': len(self.day_data)
        #     }
        
        return info
    
    def compute_reward(
        self, 
        RC_prev: int, 
        RC_current: int, 
        arrivals_prev: int, 
        departures_prev: int, 
        arrivals_current: int, 
        departures_current: int
    ) -> float:
        """
        Compute reward for current state and action.
        
        The reward includes penalties for:
        1. Queue lengths (weighted by alpha for arrivals)
        2. Configuration switches (weighted by delta)
        
        Args:
            RC_prev: Previous runway configuration
            RC_current: Current runway configuration
            arrivals_prev: Previous arrival queue
            departures_prev: Previous departure queue
            arrivals_current: Current arrival queue
            departures_current: Current departure queue
            
        Returns:
            Reward value (negative cost)
        """
        # Determine if runway configuration changed
        config_switch = int(RC_current != RC_prev)
        
        # Calculate queue penalties
        queue_penalty = -(self.alpha * arrivals_current**2 + departures_current**2)
        
        # Calculate switch penalty if applicable
        switch_penalty = -config_switch * self.delta * (self.alpha * arrivals_prev**2 + departures_prev**2)
        
        return queue_penalty + switch_penalty
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None,
        day: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            day: Specific day to use for the episode, or None for random
            
        Returns:
            Initial observation and info dictionary
        """
        # Set seed for reproducibility
        super().reset(seed=seed)
        
        # Initialize episode state
        self.time_step = 0
        self.episode_reward = 0
        
        # Select day for the episode
        if day is None:
            self.day = self.np_random.choice(self.training_days)
        else:
            self.day = day
            
        # Get data for selected day
        self.day_data = self.data.loc[self.data['day'] == self.day, :].reset_index(drop=True)
            
        # Set initial state
        self.obs = np.array([
            0,  # RC: Runway configuration (0 = None, 1 = 10R_10L, 2 = 28L_28R)
            self.day_data.loc[self.time_step, 'arrival_demand_40'],  # Arrival queue
            self.day_data.loc[self.time_step, 'departure_demand'],  # Departure queue
            self.day_data.loc[self.time_step, 'weather_condition'],  # Weather condition
            self.day_data.loc[self.time_step, 'wind_state'],  # Wind state
        ], dtype=np.float32)
        
        # Get observation and action mask
        observation = self._get_obs()
        info = {'action_mask': self._get_action_mask()}

        # Render if enabled
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action_index: Index of the selected action
            
        Returns:
            tuple of (observation, reward, terminated, truncated, info)
        """
        # Get the action from the action index
        action = self.possible_actions[action_index]
        
        # Extract current state variables
        RC_prev = int(self.obs[0])  # Previous runway configuration
        arrivals_queue = int(self.obs[1])  # Current arrival queue
        departures_queue = int(self.obs[2])  # Current departure queue
        
        # Apply action
        RC_current = action[0]  # New runway configuration
        arrivals_served = action[1]  # Arrivals to serve
        departures_served = action[2]  # Departures to serve
        
        # Calculate resulting queues
        arrivals_remaining = arrivals_queue - arrivals_served #max(0, arrivals_queue - arrivals_served)
        departures_remaining = departures_queue - departures_served #max(0, departures_queue - departures_served)
        
        if arrivals_remaining < 0 or departures_remaining < 0:
            raise ValueError('Invalid action value detected.')
        
        # Calculate reward
        reward = self.compute_reward(
            RC_prev, RC_current, 
            arrivals_queue, departures_queue, 
            arrivals_remaining, departures_remaining
        )
        self.episode_reward += reward
        
        # Check if episode is done
        terminated = (self.time_step == len(self.day_data) - 1)
        truncated = False
        
        if not terminated:
            # Advance to next time step
            self.time_step += 1
        
            # Add new demand to remaining queues
            new_arrivals = arrivals_remaining + self.day_data.loc[self.time_step, 'arrival_demand_40']
            new_departures = departures_remaining + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Update observation
            self.obs = np.array([
                RC_current,  # New runway configuration
                new_arrivals,  # Updated arrival queue
                new_departures,  # Updated departure queue
                self.day_data.loc[self.time_step, 'weather_condition'],  # Weather condition
                self.day_data.loc[self.time_step, 'wind_state'],  # Wind state
            ], dtype=np.float32)
            
            # Basic info
            info = {'action_mask': self._get_action_mask()}
        else:
            
            # Episode complete - include episode stats
            info = {
                'action_mask': self._get_action_mask(),
                'episode': {
                    'r': self.episode_reward,
                    'l': len(self.day_data) #self.time_step + 1
                }
            }
        
        # Get final observation
        observation = self._get_obs()
        
        # Render if enabled
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def set_of_actions(self) -> np.ndarray:
        """
        Get the set of valid actions for the current state.
        
        Returns:
            Array of valid actions
        """
        # Extract current state
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Validate queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
            
        # Return valid actions using precomputed boolean mask
        return self.possible_actions[self.valid_bool[wind_state, 
                                                     weather_condition,
                                                     arrivals_queue, 
                                                     departures_queue]]
    
    def action_masks(self) -> np.ndarray:
        """
        Get the action mask for the current state.
        
        Returns:
            Boolean array indicating which actions are valid
        """
        # Extract current state
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        
        # Limit queues by maximum possible action values
        arrivals_queue = min(arrivals_queue, self.u_max)
        departures_queue = min(departures_queue, self.v_max)
        
        # Validate queue values
        if arrivals_queue < 0 or departures_queue < 0:
            raise ValueError('Invalid queue value detected.')
        
        # Return boolean mask for valid actions
        return self.valid_bool[wind_state, weather_condition, arrivals_queue, departures_queue]

    def naive_action(self) -> int:
        """
        Determine the greedy action based on immediate rewards.
        
        Returns:
            Index of the greedy action
        """
        
        # Extract current state
        RC_prev = int(self.obs[0])
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        
        # Get valid actions for current state
        valid_actions = self.set_of_actions()
        
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for current state")
        
        # Calculate immediate rewards for each valid action
        rewards = []
        for action in valid_actions:
            RC_current = action[0]
            arrivals_remaining = arrivals_queue - action[1]
            departures_remaining = departures_queue - action[2]
            
            reward = self.compute_reward(
                RC_prev, RC_current, 
                arrivals_queue, departures_queue, 
                arrivals_remaining, departures_remaining
            )
            rewards.append(reward)
        
        # Select action with highest reward
        greedy_action = valid_actions[np.argmax(rewards)]
        
        # Find index of greedy action in possible_actions
        for i, action in enumerate(self.possible_actions):
            if np.array_equal(action, greedy_action):
                return i
                
        # Should never reach here if valid_actions is a subset of possible_actions
        raise ValueError("Selected greedy action not found in possible_actions")
    
    def base_action(self) -> np.ndarray:
        """
        Get the baseline action based on historical data.
        
        Returns:
            The baseline action as [RC, arrivals, departures]
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
        
        return np.array([rc, arrivals, departures], dtype=np.int64)
    
    def base_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step using provided base action.
        
        Args:
            action: The baseline action [RC, arrivals, departures]
            
        Returns:
            tuple of (observation, reward, terminated, truncated, info)
        """
        # Extract current state variables
        RC_prev = int(self.obs[0])
        arrivals_queue = int(self.obs[1])
        departures_queue = int(self.obs[2])
        
        # Apply action
        RC_current = action[0]
        arrivals_served = action[1]
        departures_served = action[2]
        
        # Calculate resulting queues
        arrivals_remaining = arrivals_queue - arrivals_served #max(0, arrivals_queue - arrivals_served)
        departures_remaining = departures_queue - departures_served #max(0, departures_queue - departures_served)
        
        if arrivals_remaining < 0 or departures_remaining < 0:
            raise ValueError('Invalid action value detected.')
        
        # Calculate reward
        reward = self.compute_reward(
            RC_prev, RC_current, 
            arrivals_queue, departures_queue, 
            arrivals_remaining, departures_remaining
        )
        self.episode_reward += reward
        
        # Check if episode is done
        terminated = (self.time_step == len(self.day_data) - 1)
        truncated = False
        
        if not terminated:
            # Advance to next time step
            self.time_step += 1
        
            # Add new demand to remaining queues
            new_arrivals = arrivals_remaining + self.day_data.loc[self.time_step, 'arrival_demand_40']
            new_departures = departures_remaining + self.day_data.loc[self.time_step, 'departure_demand']
            
            # Update observation
            self.obs = np.array([
                RC_current,  # New runway configuration
                new_arrivals,  # Updated arrival queue
                new_departures,  # Updated departure queue
                self.day_data.loc[self.time_step, 'weather_condition'],  # Weather condition
                self.day_data.loc[self.time_step, 'wind_state'],  # Wind state
            ], dtype=np.float32)
            
            # Basic info
            info = {'action_mask': self._get_action_mask()}
        else:
            # Episode complete - include episode stats
            info = {
                'action_mask': self._get_action_mask(),
                'episode': {
                    'r': self.episode_reward,
                    'l': self.time_step + 1
                }
            }
        
        # Get final observation
        observation = self._get_obs()
        
        # Render if enabled
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", otherwise None
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Create and return a rendered frame of the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", otherwise None
        """
        # For now, rendering is a stub that can be implemented later
        # Use the template below as a starting point

        # Import pygame only when rendering is required
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install it with `pip install pygame`")
            
        # Initialize pygame window if needed
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Runway Environment")
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create surface
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # Draw current state
        font = pygame.font.SysFont(None, 24)
        
        # Display runway configuration
        rc_text = f"RC: {int(self.obs[0])}"
        rc_surf = font.render(rc_text, True, (0, 0, 0))
        canvas.blit(rc_surf, (20, 20))
        
        # Display queues
        arr_text = f"Arrivals: {int(self.obs[1])}"
        arr_surf = font.render(arr_text, True, (0, 0, 200))
        canvas.blit(arr_surf, (20, 50))
        
        dep_text = f"Departures: {int(self.obs[2])}"
        dep_surf = font.render(dep_text, True, (200, 0, 0))
        canvas.blit(dep_surf, (20, 80))
        
        # Display weather and wind
        wc_text = f"Weather: {int(self.obs[3])}"
        wc_surf = font.render(wc_text, True, (0, 0, 0))
        canvas.blit(wc_surf, (20, 110))
        
        ws_text = f"Wind State: {int(self.obs[4])}"
        ws_surf = font.render(ws_text, True, (0, 0, 0))
        canvas.blit(ws_surf, (20, 140))
        
        # Display time
        time_text = f"Time Step: {self.time_step}"
        time_surf = font.render(time_text, True, (0, 0, 0))
        canvas.blit(time_surf, (20, 170))
        
        # Update display for human rendering
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        return None
        
    def close(self) -> None:
        """Clean up resources."""
        if self.window is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass  # pygame may not be installed
