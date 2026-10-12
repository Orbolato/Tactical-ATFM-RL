import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import random
from deap import base, creator, tools, algorithms
import math

class RunwayOptimizerGA:
    """
    Runway Configuration, Capacity, and Separation Optimization using Genetic Algorithms.
    
    This class implements a meta-heuristic approach to the airport runway
    configuration problem with separation optimization, solving for the optimal 
    runway configurations, service rates, and separation standards across a 
    planning horizon to minimize total costs (queue lengths, configuration
    switches, and separation penalties).
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        data: pd.DataFrame,
        u: np.ndarray,
        u_max: int,
        v: np.ndarray,
        v_max: int,
        possible_actions: List[List[int]],
        valid_bool: np.ndarray,
        sep_min: int,
        sep_max: int,
        sep_step: int,
        wind_states: List[List[int]]
    ):
        """
        Initialize the runway and separation optimizer with genetic algorithm approach.
        
        Args:
            alpha: Weight parameter for arrival queue in cost function
            beta: Weight parameter for separation penalty in cost function
            eta: Weight parameter for separation switch penalty in cost function
            delta: Weight parameter for configuration switches in cost function
            data: DataFrame containing weather and traffic data for optimization
            u: Arrival capacity matrix indexed by [weather_condition, runway_config]
            v: Departure capacity matrix indexed by [weather_condition, runway_config, arrival_rate]
            possible_actions: List of all possible actions as [RC, arr_rate, dep_rate, separation]
            valid_bool: Boolean mask for valid actions by [wind_state, weather_cond, arr_queue, dep_queue]
            sep_min: Minimum separation value
            sep_max: Maximum separation value
            wind_states: Available runway configurations for each wind state
        """
        # Store parameters
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.data = data
        self.u = u
        self.u_max = u_max
        self.v = v
        self.v_max = v_max
        self.possible_actions = np.array(possible_actions)
        self.valid_bool = valid_bool
        self.sep_min = sep_min
        self.sep_max = sep_max
        self.sep_range = sep_max - sep_min
        self.sep_step = sep_step
        self.wind_states = wind_states
        self.mean_f_100 = np.mean(self.data['arrival_demand_100']) # Mean flow at 100 NM
        self.max_f_100 = np.max(self.data['arrival_demand_100']) # Max flow at 100 NM
        self.mean_arr = np.mean(self.data['arrival_demand_40']) # Mean flow at 40 NM
        self.max_arr = np.max(self.data['arrival_demand_40']) # Max flow at 40 NM
        self.mean_dep = np.mean(self.data['departure_demand']) # Mean departures
        self.max_dep = np.max(self.data['departure_demand']) # Max departures
        
        # Define discrete separation values
        self.separation_values = [5, 10, 15, 20, 25, 30]
        self.separation_indices = {val: idx for idx, val in enumerate(self.separation_values)}
        
        # Define separation actions: 0 = reduce by 5, 1 = keep same, 2 = increase by 5
        self.separation_actions = [0, 1, 2]  # reduce, keep, increase
        
        # For storing results
        self.results = {}
        
        # GA parameters
        self.population_size = 100
        self.generations = 100
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        
        # Initialize DEAP classes only once
        if not hasattr(creator, "FitnessMin") or not isinstance(creator.FitnessMin, type):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual") or not isinstance(creator.Individual, type):
            creator.create("Individual", list, fitness=creator.FitnessMin)
    
    def get_valid_separation_actions(self, 
                                     prev_separation: int,
                                     current_separation: int) -> List[int]:
       """
       Get valid separation actions based on current separation value.
       
       Args:
           current_separation: Current separation value
           
       Returns:
           List of valid separation actions
       """
       if prev_separation < current_separation:
           if current_separation == self.sep_max:  # separation = 30
               return [1]  # reduce or keep same
           else:  # separation in [5, 10, 15, 20, 25]
               return [1, 2]  # keep same or increase
       elif prev_separation > current_separation:
           if current_separation == self.sep_min:  # separation = 5
               return [1]  # keep same
           else:  # separation in [10, 15, 20, 25, 30]
               return [0, 1]  # reduce or keep same
       else:
           if current_separation == self.sep_min:  # separation = 5
               return [1, 2]  # keep same or increase
           elif current_separation == self.sep_max: # separation = 30
               return [0, 1] # keep same or decrease
           else:
               return [0, 1, 2] # all actions possible
           
    def apply_separation_action(self, current_separation: int, action: int) -> int:
           """
           Apply separation action to get new separation value.
           
           Args:
               current_separation: Current separation value
               action: Separation action (0=reduce, 1=keep, 2=increase)
               
           Returns:
               New separation value after applying action
           """
           if action == 0:  # reduce by 5
               new_sep = max(current_separation - self.sep_step, self.sep_min)
           elif action == 1:  # keep same
               new_sep = current_separation
           elif action == 2:  # increase by 5
               new_sep = min(current_separation + self.sep_step, self.sep_max)
           else:
               raise ValueError(f"Invalid separation action: {action}")
           
           return new_sep
    
    def compute_reward(self, 
                      RC_prev: int, 
                      RC_t: int, 
                      a_prev: int, 
                      d_prev: int, 
                      a_t: int, 
                      d_t: int,
                      f_t: int,
                      s_prev: int,
                      s_t: int
                      ) -> float:
        """
        Compute the reward (negative cost) for taking an action from the current state.
        This mirrors the reward function from the RL environment.
        
        Args:
            RC_prev: Previous runway configuration
            RC_t: New runway configuration
            a_prev: Previous arrival queue
            d_prev: Previous departure queue
            a_t: New arrival queue
            d_t: New departure queue
            f_t: Arrival demand
            s_prev: Previous separation value
            s_t: New separation value
            
        Returns:
            float: Calculated reward (negative cost)
        """
        # Determine if runway configuration changed
        config_switch = int(RC_t != RC_prev)
        
        # Calculate queue penalties (negative reward based on squared queue lengths)
        queue_penalty = -(self.alpha * a_t**2 + d_t**2)
        
        # Calculate switch penalty if configuration changed
        # Only penalize switches when there are aircraft in the queue
        switch_penalty = -config_switch * self.delta * (self.alpha * a_prev**2 + d_prev**2)
        
        # Calculate separation penalty (deviation from reference separation)
        sep_penalty = -self.beta * (s_t - self.sep_min) / self.sep_range * f_t / self.max_f_100
        
        # Sum all penalties (negative cost)
        total_reward = queue_penalty + switch_penalty + sep_penalty
        
        return total_reward
    
    def _create_warm_start_individual(self, day_data, time_steps):
        """
        Create a warm start individual from baseline data.
        
        Args:
            day_data: DataFrame with day-specific data
            time_steps: Number of time steps
            
        Returns:
            Individual or None if warm start data is not available
        """
        if 'greedy_configuration' not in day_data.columns:
            return None
            
        base_configs = day_data['greedy_configuration'].tolist()
        base_arrivals = day_data['greedy_arrivals_40'].tolist()
        
        # Ensure configs and arrivals are within valid bounds
        base_configs = [max(0, min(2, int(c))) for c in base_configs]
        base_arrivals = [max(0, min(self.u_max, int(a))) for a in base_arrivals]
        
        # Create separation actions that maintain minimum separation (action 1 = keep same)
        base_separation_actions = [1] * time_steps
        
        # Create and return the warm start individual
        warm_start_ind = creator.Individual(base_configs + base_arrivals + base_separation_actions)
        return warm_start_ind
    
    def _evaluate_individual(self, individual, weather_conditions, wind_states, 
                           arrival_demands, departure_demands, flow, time_steps):
        """
        Evaluate an individual solution and return its fitness.
        
        Args:
            individual: Individual to evaluate
            weather_conditions: Weather conditions for each time step
            wind_states: Wind states for each time step  
            arrival_demands: Arrival demands for each time step
            departure_demands: Departure demands for each time step
            flow: Arriving flow through 100-NM cylinder
            time_steps: Number of time steps
            
        Returns:
            Tuple containing the fitness value (cost)
        """
        # Parse individual: [configs, arrival_rates, separations]
        configs = individual[:time_steps]
        arrival_rates = individual[time_steps:2*time_steps]
        separation_actions = individual[2*time_steps:3*time_steps]
        
        # Validate separation actions are within bounds
        for sep_action in separation_actions:
            if sep_action not in [0, 1, 2]:
                return (1e9,)
        
        # Validate configurations against wind constraints
        for t in range(time_steps):
            ws = wind_states[t]
            if configs[t] not in self.wind_states[ws]:
                # If configuration is invalid for wind state, give a large penalty
                return (1e9,)
        
        # Calculate actual separations from actions
        separations = []
        prev_separation = self.sep_min
        current_separation = self.sep_min  # Start with minimum separation
        
        for t in range(time_steps):
            action = separation_actions[t]
            
            # Check if action is valid for current separation
            valid_actions = self.get_valid_separation_actions(prev_separation, 
                                                              current_separation)
            if action not in valid_actions:
                return (1e9,)  # Invalid action sequence
            
            # Apply action to get new separation
            new_separation = self.apply_separation_action(current_separation, action)
            separations.append(new_separation)
            prev_separation = current_separation
            current_separation = new_separation
        
        # Calculate service rates based on configurations and constraints
        arrivals_served = []
        departures_served = []
        
        for t in range(time_steps):
            wc = weather_conditions[t]
            rc = configs[t]
            
            # Get maximum arrival capacity for this config and weather
            max_arr_capacity = self.u[wc, rc][-1] if len(self.u[wc, rc]) > 0 else 0
            
            # Determine arrival rate (capped by capacity)
            arr_rate = min(arrival_rates[t], max_arr_capacity)
            arrivals_served.append(arr_rate)
            
            # Determine departure rate based on arrival rate
            if arr_rate < len(self.v[wc, rc]):
                dep_capacity = self.v[wc, rc][arr_rate]
            else:
                dep_capacity = 0
            departures_served.append(dep_capacity)
        
        # Calculate queue dynamics with separation effects
        arrival_queue = [arrival_demands[0]]
        departure_queue = [departure_demands[0]]
        
        # Initialize previous values for reward calculation
        prev_config = 0  # Initial configuration
        prev_separation = self.sep_min  # Initial separation
        
        total_cost = 0.0
        
        arrival_demands_copy = arrival_demands.copy()
        
        for t in range(time_steps):
            # Get current queue values
            current_arr_queue = arrival_queue[-1]
            current_dep_queue = departure_queue[-1]
            current_config = configs[t]
            current_separation = separations[t]
            
            # Serve aircraft based on capacity and queue
            arr_served = min(arrivals_served[t], current_arr_queue)
            dep_served = min(departures_served[t], current_dep_queue)
            
            # Calculate queues after service
            arr_queue_after = current_arr_queue - arr_served
            dep_queue_after = current_dep_queue - dep_served
            
            # Calculate cost (negative reward) for this time step
            cost = -self.compute_reward(
                prev_config, current_config,
                current_arr_queue, current_dep_queue,
                arr_queue_after, dep_queue_after, flow[t],
                prev_separation, current_separation
            )
            total_cost += cost
            
            # Calculate next time step's queue if not at the end
            if t < time_steps - 1:
                # Apply separation effect on flow
                flow_factor = self.sep_min / current_separation
                next_arr_demand = int(arrival_demands_copy[t+1] * flow_factor)
                overflow = arrival_demands_copy[t+1] - next_arr_demand
                
                if t < time_steps - 2:
                    arrival_demands_copy[t+2] += overflow
                
                next_arr_queue = arr_queue_after + next_arr_demand
                next_dep_queue = dep_queue_after + departure_demands[t+1]
                
                arrival_queue.append(next_arr_queue)
                departure_queue.append(next_dep_queue)
            
            # Update previous values
            prev_config = current_config
            prev_separation = current_separation
        
        return (total_cost,)
    
    def optimize_day(self, day: int, verbose: bool = False, use_warm_start: bool = False) -> Dict[str, Any]:
        """
        Optimize runway operations for a single day using genetic algorithm.
        
        Args:
            day: Day index to optimize
            verbose: Whether to print optimization progress
            use_warm_start: Whether to use baseline solution as part of initial population
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract data for the specified day
        day_data = self.data.loc[self.data['day'] == day].reset_index(drop=True)
        time_steps = len(day_data)
        
        if verbose:
            print(f"Optimizing day {day} with {time_steps} time steps using Genetic Algorithm")
        
        # Get weather conditions and wind states for this day
        weather_conditions = day_data['weather_condition'].astype(int).tolist()
        wind_states = day_data['wind_state'].astype(int).tolist()
        arrival_demands = day_data['arrival_demand_40'].tolist()
        departure_demands = day_data['departure_demand'].tolist()
        flow = day_data['arrival_demand_100'].tolist()
        
        # Create evaluation function wrapper
        def evaluate_individual(individual):
            return self._evaluate_individual(
                individual, weather_conditions, wind_states,
                arrival_demands, departure_demands, flow, time_steps
            )
        
        # Setup genetic algorithm components
        toolbox = base.Toolbox()
        
        # Define individual structure: [configs, arrival_rates, separations]
        # Gene range for configs: [0, 1, 2] (None, 10R_10L, 28L_28R)
        def config_gene():
            return random.randint(0, 2)
        
        # Gene range for arrival rates: [0, ..., u_max]
        def arrival_rate_gene():
            return random.randint(0, self.u_max)
        
        # Gene range for separation actions: [0, 1, 2] (reduce, keep, increase)
        def separation_action_gene():
            return random.randint(0, 2)
        
        toolbox.register("config_gene", config_gene)
        toolbox.register("arrival_rate_gene", arrival_rate_gene)
        toolbox.register("separation_action_gene", separation_action_gene)
        
        # Define individual creation with proper bounds checking
        def create_individual():
            individual = []
            # Add config genes
            for _ in range(time_steps):
                individual.append(config_gene())
            # Add arrival rate genes
            for _ in range(time_steps):
                individual.append(arrival_rate_gene())
            # Add separation genes
            for _ in range(time_steps):
                individual.append(separation_action_gene())
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        
        # Define population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self._custom_mutate, indpb=0.1, time_steps=time_steps)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=self.population_size)
        
        # Initialize best solution tracking
        best_individual = None
        best_fitness = float('inf')
        warm_start_fitness = None
        
        # Handle warm start solution
        if use_warm_start:
            warm_start_ind = self._create_warm_start_individual(day_data, time_steps)
            
            if warm_start_ind is not None:
                # Evaluate warm start solution
                warm_start_fitness = evaluate_individual(warm_start_ind)[0]
                
                if verbose:
                    print(f"Warm start solution fitness: {warm_start_fitness}")
                
                # Set as initial best if valid
                if warm_start_fitness < 1e8:  # Valid solution
                    best_individual = warm_start_ind.copy()
                    best_fitness = warm_start_fitness
                    
                    if verbose:
                        print(f"Warm start solution set as initial best with fitness: {best_fitness}")
                
                # Add warm start to population (replace first individual)
                if len(pop) > 0:
                    pop[0] = warm_start_ind
                    
            elif verbose:
                print("Warm start data not available, proceeding without warm start")
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
            # Update best solution if this individual is better
            if fit[0] < best_fitness:
                best_individual = ind.copy()
                best_fitness = fit[0]
        
        if verbose:
            initial_best = min(fitnesses, key=lambda x: x[0])[0]
            print(f"Initial population evaluated. Best fitness: {initial_best}")
            if warm_start_fitness is not None:
                improvement = ((warm_start_fitness - initial_best) / warm_start_fitness * 100 
                              if warm_start_fitness > 0 else 0)
                print(f"Initial improvement over warm start: {improvement:.2f}%")
        
        # Setup statistics and hall of fame
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Custom algorithm loop to track best solution throughout evolution
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Evaluate the population with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if hof is not None:
            hof.update(pop)
        
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        if verbose:
            print(logbook.stream)
        
        # Begin the generational process
        for gen in range(1, self.generations + 1):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            
            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, self.crossover_prob, self.mutation_prob)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
                # Update best solution tracking
                if fit[0] < best_fitness:
                    best_individual = ind.copy()
                    best_fitness = fit[0]
                    if verbose and gen % 10 == 0:
                        print(f"New best solution found at generation {gen}: {best_fitness}")
            
            # Update the hall of fame with the generated individuals
            if hof is not None:
                hof.update(offspring)
            
            # Replace the current population by the offspring
            pop[:] = offspring
            
            # Append the current generation statistics to the logbook
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            if verbose and gen % 10 == 0:
                print(logbook.stream)
        
        # Ensure we have the absolute best individual (either from evolution or tracking)
        if best_individual is None or (hof[0].fitness.values[0] < best_fitness):
            best_individual = hof[0]
            best_fitness = hof[0].fitness.values[0]
        
        # Validate the best individual
        def is_valid_individual(individual, wind_states, time_steps):
            configs = individual[:time_steps]
            separation_actions = individual[2*time_steps:3*time_steps]
            
            # Check config validity
            for t, (ws, rc) in enumerate(zip(wind_states, configs)):
                if rc not in self.wind_states[ws]:
                    return False
            
            # Check separation action validity
            prev_separation = self.sep_min
            current_separation = self.sep_min
            for action in separation_actions:
                if action not in [0, 1, 2]:
                    return False
                valid_actions = self.get_valid_separation_actions(prev_separation,
                                                                  current_separation)
                if action not in valid_actions:
                    return False
                prev_separation = current_separation
                current_separation = self.apply_separation_action(current_separation, action)
            
            return True
        
        # If best individual is invalid, find the best valid one in the population
        if not is_valid_individual(best_individual, wind_states, time_steps):
            valid_individuals = [ind for ind in pop if is_valid_individual(ind, wind_states, time_steps)]
            if valid_individuals:
                # Select the valid individual with the best fitness
                best_individual = min(valid_individuals, key=lambda ind: ind.fitness.values[0])
                best_fitness = best_individual.fitness.values[0]
            else:
                raise ValueError("No valid individuals found in the population.")
        
        # Extract solution components
        best_configs = best_individual[:time_steps]
        best_arrival_rates = best_individual[time_steps:2*time_steps]
        best_separation_actions = best_individual[2*time_steps:3*time_steps]
        
        # Convert separation actions to actual separation values
        best_separations = []
        current_separation = self.sep_min
        for action in best_separation_actions:
           new_separation = self.apply_separation_action(current_separation, action)
           best_separations.append(new_separation)
           current_separation = new_separation
        
        # Calculate derived values based on the best solution
        best_arrivals_served = []
        best_departures_served = []
        best_arrival_queue = [arrival_demands[0]]
        best_departure_queue = [departure_demands[0]]
        
        # Initialize for cost calculation
        prev_config = 0
        prev_separation = self.sep_min
        total_cost = 0.0
        queue_costs = []
        switch_costs = []
        sep_costs = []
        arrival_demands_copy = arrival_demands.copy()
        
        for t in range(time_steps):
            wc = weather_conditions[t]
            rc = best_configs[t]
            sep = best_separations[t]
            
            # Calculate arrival service rate
            max_arr_capacity = self.u[wc, rc][-1] if len(self.u[wc, rc]) > 0 else 0
            arr_rate = min(best_arrival_rates[t], max_arr_capacity)
            arr_served = min(arr_rate, best_arrival_queue[-1])
            best_arrivals_served.append(arr_served)
            
            # Calculate departure service rate based on arrival rate
            if arr_rate < len(self.v[wc, rc]):
                dep_capacity = self.v[wc, rc][arr_rate]
            else:
                dep_capacity = 0
            dep_served = min(dep_capacity, best_departure_queue[-1])
            best_departures_served.append(dep_served)
            
            # Calculate queues after service
            arr_queue_after = best_arrival_queue[-1] - arr_served
            dep_queue_after = best_departure_queue[-1] - dep_served
            
            # Calculate cost components for this time step
            config_switch = int(rc != prev_config)
            queue_cost = self.alpha * arr_queue_after**2 + dep_queue_after**2
            switch_cost = config_switch * self.delta * (self.alpha * best_arrival_queue[-1]**2 + best_departure_queue[-1]**2)
            sep_cost = self.beta * (sep - self.sep_min) / self.sep_range * flow[t] / self.max_f_100
            
            step_cost = queue_cost + switch_cost + sep_cost
            total_cost += step_cost
            
            queue_costs.append(queue_cost)
            switch_costs.append(switch_cost)
            sep_costs.append(sep_cost)
            
            # Calculate next period's queues if not at the end
            if t < time_steps - 1:
                # Apply separation effect
                flow_factor = self.sep_min / sep
                next_arr_demand = int(arrival_demands_copy[t+1] * flow_factor)
                overflow = arrival_demands_copy[t+1] - next_arr_demand
                
                if t < len(day_data) - 2:
                    arrival_demands_copy[t+2] += overflow
                
                next_arr_queue = arr_queue_after + next_arr_demand
                next_dep_queue = dep_queue_after + departure_demands[t+1]
                
                best_arrival_queue.append(next_arr_queue)
                best_departure_queue.append(next_dep_queue)
            
            # Update previous values
            prev_config = rc
            prev_separation = sep
        
        # Calculate configuration switches
        config_switches = [1]  # First period always counted as a switch
        for t in range(1, time_steps):
            config_switches.append(1 if best_configs[t] != best_configs[t-1] else 0)
        
        # Compile solution
        solution = {
            'status': 'optimal',  # We assume GA found a good solution
            'termination_condition': 'maxGenerations',
            'objective': total_cost,
            'best_fitness_tracked': best_fitness,
            'warm_start_fitness': warm_start_fitness,
            'time_steps': time_steps,
            'configs': best_configs,
            'arrivals_served': best_arrivals_served,
            'departures_served': best_departures_served,
            'separations': best_separations,
            'arrival_queue': best_arrival_queue,
            'departure_queue': best_departure_queue,
            'config_switches': config_switches,
            'weather_condition': weather_conditions,
            'wind_state': wind_states,
            'arrival_demand': arrival_demands,
            'departure_demand': departure_demands,
            'queue_cost': sum(queue_costs),
            'switch_cost': sum(switch_costs),
            'sep_cost': sum(sep_costs),
            'queue_costs': queue_costs,
            'switch_costs': switch_costs,
            'sep_costs': sep_costs,
            'ga_logbook': logbook
        }
        
        # Store results
        self.results[day] = solution
        
        if verbose:
            print(f"Optimization completed with objective value: {total_cost}")
            if warm_start_fitness is not None:
                improvement = ((warm_start_fitness - total_cost) / warm_start_fitness * 100 
                              if warm_start_fitness > 0 else 0)
                print(f"Improvement over warm start: {improvement:.2f}%")
            print(f"Queue cost: {sum(queue_costs)}, Switch cost: {sum(switch_costs)}")
            print(f"Separation cost: {sum(sep_costs)}")
            print(f"Total configuration switches: {sum(config_switches)}")
        
        return solution
    
    def _custom_mutate(self, individual, indpb, time_steps):
        """
        Custom mutation operator that respects the structure of individuals.
        
        Args:
            individual: The individual to mutate
            indpb: Independent probability of each gene to be mutated
            time_steps: Number of time steps in the problem
            
        Returns:
            Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                if i < time_steps:  # Config gene
                    individual[i] = random.randint(0, 2)
                elif i < 2 * time_steps:  # Arrival rate gene
                    individual[i] = random.randint(0, self.u_max)
                else:  # Separation action gene
                    individual[i] = random.randint(0, 2)
        
        return (individual,)
    
    def solve_all_days(self, days: List[int], verbose: bool = False, use_warm_start: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Optimize all specified days using genetic algorithm.
        
        Args:
            days: List of days to optimize
            verbose: Whether to print solver output
            use_warm_start: Whether to use baseline solution in initial population
            
        Returns:
            Dictionary mapping day index to optimization results
        """
        results = {}
        for day in days:
            if verbose:
                print(f"\nOptimizing day {day} with GA...")
            
            results[day] = self.optimize_day(day, verbose, use_warm_start)
            
            if verbose:
                print(f"Day {day} completed with objective value: {results[day]['objective']}")
                
        return results
    
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
        day_data = self.data.loc[self.data['day'] == day].reset_index(drop=True).copy()
        
        # Check if baseline columns exist
        if 'greedy_configuration' in day_data.columns:
            base_configs = day_data['greedy_configuration'].tolist()
            base_arrivals = day_data['greedy_arrivals_40'].tolist()
            base_departures = day_data['greedy_departures'].tolist()
        else:
            # Use greedy baseline if base doesn't exist
            base_configs = day_data.get('greedy_configuration', [0] * len(day_data)).tolist()
            base_arrivals = day_data.get('greedy_arrivals_40', [0] * len(day_data)).tolist()
            base_departures = day_data.get('greedy_departures', [0] * len(day_data)).tolist()
        
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
            sep_cost = self.beta * (curr_separation - self.sep_min) / self.sep_range * day_data.loc[t, 'arrival_demand_100'] / self.max_f_100
            
            step_cost = queue_cost + switch_cost + sep_cost
            base_total_cost += step_cost
            
            # Calculate new queues
            if t < len(day_data) - 1:
                # Apply separation effect
                flow_factor = self.sep_min / curr_separation
                next_arr_demand = int(day_data.loc[t+1, 'arrival_demand_40'] * flow_factor)
                overflow = day_data.loc[t+1, 'arrival_demand_40'] - next_arr_demand
                
                if t < len(day_data) - 2:
                    day_data.loc[t+2, 'arrival_demand_40'] += overflow
                
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
            'base_dep_queue': base_departure_queue,
            'opt_switches': sum(opt_results['config_switches']),
            'base_switches': sum(base_config_switches)
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
        axs[0].plot(time_steps, comparison['opt_configs'], 'b-', label='GA Optimized')
        axs[0].plot(time_steps, comparison['base_configs'], 'r--', label='Baseline')
        axs[0].set_ylabel('Runway Config')
        axs[0].set_title(f'Day {day} - Comparison (Improvement: {comparison["improvement"]:.2f}%)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot separations
        axs[1].plot(time_steps, comparison['opt_separations'], 'b-', label='GA Optimized')
        axs[1].plot(time_steps, comparison['base_separations'], 'r--', label='Baseline')
        axs[1].set_ylabel('Separation')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot arrival queues - ensure we only plot the same number of points
        axs[2].plot(time_steps, comparison['opt_arr_queue'][:n_time_steps], 'b-', label='GA Optimized')
        axs[2].plot(time_steps, comparison['base_arr_queue'][:n_time_steps], 'r--', label='Baseline')
        axs[2].set_ylabel('Arrival Queue')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot departure queues - ensure we only plot the same number of points
        axs[3].plot(time_steps, comparison['opt_dep_queue'][:n_time_steps], 'b-', label='GA Optimized')
        axs[3].plot(time_steps, comparison['base_dep_queue'][:n_time_steps], 'r--', label='Baseline')
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
        print(f"  Objective: GA Optimized = {comparison['opt_objective']:.2f}, Baseline = {comparison['base_objective']:.2f}")
        print(f"  Improvement: {comparison['improvement']:.2f}%")
        print(f"  Config Switches: GA Optimized = {comparison['opt_switches']}, Baseline = {comparison['base_switches']}")
        print("  Cost Breakdown:")
        print(f"    Queue Cost: {results['queue_cost']:.2f}")
        print(f"    Switch Cost: {results['switch_cost']:.2f}")
        print(f"    Separation Cost: {results['sep_cost']:.2f}")

    def plot_convergence(self, day: int, figsize=(10, 6)):
        """
        Plot the convergence of the genetic algorithm for a specific day.
        
        Args:
            day: Day index to plot
            figsize: Figure size for the plot
        """
        if day not in self.results or 'ga_logbook' not in self.results[day]:
            raise ValueError(f"Day {day} has not been optimized with GA yet")
        
        # Get logbook data
        logbook = self.results[day]['ga_logbook']
        
        # Extract statistics
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_avgs = logbook.select("avg")
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.plot(gen, fit_mins, 'b-', label='Best Fitness')
        plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Cost)')
        plt.title(f'GA Convergence for Day {day}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale often better for visualizing convergence
        plt.tight_layout()
        plt.show()
    
        