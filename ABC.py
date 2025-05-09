import numpy as np
import time
import argparse
from Optimization_Functions import OptimizationFunctions

class ABCOptimizer:
    def __init__(self, lower_bound, upper_bound, colony_size, dimension, 
                 limit=100, employed_bees=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.colony_size = colony_size
        self.dimension = dimension
        self.employed_bees = employed_bees if employed_bees else colony_size // 2
        self.onlooker_bees = colony_size - self.employed_bees
        self.limit = limit  # Limit for scout bee phase
        
        # Initialize food sources (solutions)
        self.food_sources = self._initialize_food_sources()
        self.fitness_scores = np.full(self.employed_bees, np.inf)
        self.trial_counts = np.zeros(self.employed_bees)  # Counter for abandonment
        
        self.best_solution = None
        self.best_fitness = np.inf
        
        self.convergence_history = []
        self.diversity_history = []
    
    def _initialize_food_sources(self):
        food_sources = np.zeros((self.employed_bees, self.dimension))
        for d in range(self.dimension):
            segment_size = (self.upper_bound - self.lower_bound) / self.employed_bees
            for i in range(self.employed_bees):
                food_sources[i, d] = self.lower_bound + (i + np.random.random()) * segment_size
        return food_sources
    
    def _calculate_fitness(self, objective_value):
        # Convert objective value to fitness (maximization problem)
        if objective_value >= 0:
            return 1 / (1 + objective_value)
        else:
            return 1 + abs(objective_value)
    
    def _calculate_probabilities(self):
        # Calculate selection probability for onlooker bees
        fitness_values = np.array([self._calculate_fitness(score) for score in self.fitness_scores])
        return fitness_values / np.sum(fitness_values)
    
    def _generate_neighbor_solution(self, current_solution, partner_idx):
        # Generate new solution near current solution
        phi = np.random.uniform(-1, 1, self.dimension)
        partner_solution = self.food_sources[partner_idx]
        
        # Generate new solution
        neighbor = current_solution.copy()
        param_to_change = np.random.randint(0, self.dimension)
        neighbor[param_to_change] = (current_solution[param_to_change] + 
                                   phi[param_to_change] * 
                                   (current_solution[param_to_change] - partner_solution[param_to_change]))
        
        # Ensure bounds
        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor
    
    def _scout_bee_phase(self):
        # Identify and replace abandoned solutions
        abandoned = np.where(self.trial_counts >= self.limit)[0]
        for idx in abandoned:
            self.food_sources[idx] = np.random.uniform(
                self.lower_bound, 
                self.upper_bound, 
                self.dimension
            )
            self.trial_counts[idx] = 0
    
    def _calculate_swarm_diversity(self):
        mean_position = np.mean(self.food_sources, axis=0)
        distances = np.linalg.norm(self.food_sources - mean_position, axis=1)
        return np.mean(distances)
    
    def optimize(self, objective_func, max_cycles=1000, tolerance=1e-6, verbose=True):
        if verbose:
            print("\nOptimization Progress:")
            
        for cycle in range(max_cycles):
            # Employed Bees Phase
            for i in range(self.employed_bees):
                # Select random partner (excluding current bee)
                possible_partners = [j for j in range(self.employed_bees) if j != i]
                partner_idx = np.random.choice(possible_partners)
                
                # Generate new solution
                new_solution = self._generate_neighbor_solution(self.food_sources[i], partner_idx)
                
                # Evaluate new solution
                current_fitness = objective_func(self.food_sources[i])
                new_fitness = objective_func(new_solution)
                
                # Greedy selection
                if new_fitness < current_fitness:
                    self.food_sources[i] = new_solution
                    self.fitness_scores[i] = new_fitness
                    self.trial_counts[i] = 0
                else:
                    self.trial_counts[i] += 1
                    self.fitness_scores[i] = current_fitness
            
            # Onlooker Bees Phase
            probabilities = self._calculate_probabilities()
            visited_count = 0
            bee = 0
            
            while visited_count < self.onlooker_bees:
                if np.random.random() < probabilities[bee]:
                    # Select random partner
                    possible_partners = [j for j in range(self.employed_bees) if j != bee]
                    partner_idx = np.random.choice(possible_partners)
                    
                    # Generate new solution
                    new_solution = self._generate_neighbor_solution(self.food_sources[bee], partner_idx)
                    
                    # Evaluate new solution
                    current_fitness = objective_func(self.food_sources[bee])
                    new_fitness = objective_func(new_solution)
                    
                    # Greedy selection
                    if new_fitness < current_fitness:
                        self.food_sources[bee] = new_solution
                        self.fitness_scores[bee] = new_fitness
                        self.trial_counts[bee] = 0
                    else:
                        self.trial_counts[bee] += 1
                        self.fitness_scores[bee] = current_fitness
                    
                    visited_count += 1
                
                bee = (bee + 1) % self.employed_bees
            
            # Scout Bees Phase
            self._scout_bee_phase()
            
            # Update best solution
            min_fitness_idx = np.argmin(self.fitness_scores)
            if self.fitness_scores[min_fitness_idx] < self.best_fitness:
                self.best_fitness = self.fitness_scores[min_fitness_idx]
                self.best_solution = self.food_sources[min_fitness_idx].copy()
            
            # Track metrics
            self.convergence_history.append(self.best_fitness)
            self.diversity_history.append(self._calculate_swarm_diversity())
            
            # Print current cycle results if verbose
            if verbose and (cycle % 100 == 0 or cycle == max_cycles - 1):
                print(f"\nCycle {cycle + 1}:")
                print("Current Best Position:")
                for i, pos in enumerate(self.best_solution):
                    print(f"x{i+1}: {pos:.10f}")
                print(f"Current Best Fitness: {self.best_fitness:.10f}")
                print(f"Current Colony Diversity: {self.diversity_history[-1]:.6f}")
            
            if self.best_fitness <= tolerance:
                if verbose:
                    print(f"\nConvergence achieved at cycle {cycle + 1}")
                break
        
        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness,
            'cycles': cycle + 1,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }

def run_simulation(optimizer, objective_func, num_trials, max_cycles, tolerance):
    all_results = []
    best_fitness_overall = np.inf
    best_position_overall = None
    total_start_time = time.time()
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Reset optimizer for new trial
        optimizer.food_sources = optimizer._initialize_food_sources()
        optimizer.fitness_scores = np.full(optimizer.employed_bees, np.inf)
        optimizer.trial_counts = np.zeros(optimizer.employed_bees)
        optimizer.best_solution = None
        optimizer.best_fitness = np.inf
        optimizer.convergence_history = []
        optimizer.diversity_history = []
        
        # Run optimization with reduced verbosity for all but the first trial
        verbose = (trial == 0)
        result = optimizer.optimize(
            objective_func=objective_func,
            max_cycles=max_cycles,
            tolerance=tolerance,
            verbose=verbose
        )
        
        all_results.append(result)
        
        # Update overall best if necessary
        if result['best_fitness'] < best_fitness_overall:
            best_fitness_overall = result['best_fitness']
            best_position_overall = result['best_position'].copy()
        
        print(f"Trial {trial + 1} completed:")
        print(f"Best Fitness: {result['best_fitness']:.10f}")
        print(f"Cycles: {result['cycles']}")
    
    # Calculate statistics
    fitness_values = [r['best_fitness'] for r in all_results]
    cycles_values = [r['cycles'] for r in all_results]
    
    statistics = {
        'best_fitness_overall': best_fitness_overall,
        'best_position_overall': best_position_overall,
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values),
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'mean_cycles': np.mean(cycles_values),
        'std_cycles': np.std(cycles_values),
        'total_time': time.time() - total_start_time,
        'num_trials': num_trials
    }
    
    return statistics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Artificial Bee Colony Algorithm Parameters')
    parser.add_argument('--function', type=str, required=True, help='Function to optimize')
    parser.add_argument('--colony-size', type=int, default=50, help='Colony size')
    parser.add_argument('--dimension', type=int, default=2, help='Problem dimension')
    parser.add_argument('--limit', type=int, default=100, help='Limit for scout bee phase')
    parser.add_argument('--max-cycles', type=int, default=1000, help='Maximum cycles')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--trials', type=int, default=1, help='Number of independent trials to run')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    bounds = {
        'ackley': (-32.768, 32.768),
        'three_hump_camel': (-5, 5),
        'six_hump_camel': (-3, 3),
        'dixon_price': (-10, 10),
        'rosenbrock': (-5, 10),
        'beale': (-4.5, 4.5),
        'branin': (-5, 10),
        'colville': (-10, 10),
        'forrester': (0, 1),
        'goldstein_price': (-2, 2),
        'hartmann_3d': (0, 1),
        'hartmann_4d': (0, 1),
        'hartmann_6d': (0, 1),
        'perm': (-2, 2), 
        'powell': (-4, 5),
        'shekel': (0, 10),
        'styblinski_tang': (-5, 5),
        'bukin': (-15, 15),
        'cross_in_tray': (-10, 10),
        'drop_wave': (-5.12, 5.12),
        'eggholder': (-512, 512),
        'gramacy_lee': (0.5, 2.5),
        'griewank': (-600, 600),
        'holder_table': (-10, 10),
        'langermann': (0, 10),
        'levy': (-10, 10),
        'levy_n13': (-10, 10),
        'rastrigin': (-5.12, 5.12),
        'schaffer_n2': (-100, 100),
        'schaffer_n4': (-100, 100),
        'schwefel': (-500, 500),
        'shubert': (-10, 10),
        'michalewicz': (0, np.pi),
        'easom': (-100, 100),
        'booth': (-10, 10),
        'matyas': (-10, 10),
        'zakharov': (-5, 10),
        'bohachevsky': (-100, 100),
        'perm_0': (-2, 2), 
        'rotated_hyper_ellipsoid': (-65.536, 65.536),
        'sphere': (-5.12, 5.12),
        'sum_of_different_powers': (-1, 1), 
        'mccormick': (-1.5, 4),
        'trid': (-2, 2), 
        'power_sum': (0, 2), 
        'cde_jong': (-65.536, 65.536),
        'sum_squares': (-10, 10),
        'ackley2': (-32.768, 32.768),
        'alpine': (-10, 10),
        'bukin_n4': (-15, 15),
        'bukin_n5': (-15, 15),
        'cosine_mixture': (-1, 1)
    }
    dimension_requirements = {
        'sphere': 2,
        'ackley': 2,
        'three_hump_camel': 2,
        'six_hump_camel': 2,
        'dixon_price': 2,
        'rosenbrock': 2,
        'beale': 2,
        'branin': 2,
        'colville': 4,
        'forrester': 1,
        'goldstein_price': 2,
        'hartmann_3d': 3,
        'hartmann_4d': 4,
        'hartmann_6d': 6,
        'perm': 2,
        'powell': 4,
        'shekel': 4,
        'styblinski_tang': 2,
        'bukin': 2,
        'cross_in_tray': 2,
        'drop_wave': 2,
        'eggholder': 2,
        'gramacy_lee': 1,
        'griewank': 2,
        'holder_table': 2,
        'langermann': 2,
        'levy': 2,
        'levy_n13': 2,
        'rastrigin': 2,
        'schaffer_n2': 2,
        'schaffer_n4': 2,
        'schwefel': 2,
        'shubert': 2,
        'michalewicz': 2,
        'easom': 2,
        'booth': 2,
        'matyas': 2,
        'zakharov': 2,
        'bukin_n5': 2,
        'schwefel_226': 2,
        'sinc': 2,
        'ackley2': 2,
        'sum_squares': 2,
        'step': 2,
        'alpine': 2,
        'bukin_n4': 2,
        'cosine_mixture': 2
    }
    
    if args.function not in bounds:
        raise ValueError(f"Function '{args.function}' is not implemented or bounds are not defined.")
    
    lower_bound, upper_bound = bounds[args.function]
    
    # Initialize optimizer
    optimizer = ABCOptimizer(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        colony_size=args.colony_size,
        dimension=args.dimension,
        limit=args.limit
    )
    
    # Initialize optimization functions
    opt_functions = OptimizationFunctions()
    objective_func = getattr(opt_functions, f"{args.function}_function")
    
    # Run simulation if multiple trials requested
    if args.trials > 1:
        statistics = run_simulation(
            optimizer=optimizer,
            objective_func=objective_func,
            num_trials=args.trials,
            max_cycles=args.max_cycles,
            tolerance=args.tolerance
        )
        
        # Print statistical results
        print("\nStatistical Results:")
        print(f"Number of Trials: {statistics['num_trials']}")
        print(f"Best Fitness Overall: {statistics['best_fitness_overall']:.10f}")
        print(f"Best Position Overall: {statistics['best_position_overall']}")
        print(f"Mean Fitness: {statistics['mean_fitness']:.10f}")
        print(f"Std. Dev. Fitness: {statistics['std_fitness']:.10f}")
        print(f"Min Fitness: {statistics['min_fitness']:.10f}")
        print(f"Max Fitness: {statistics['max_fitness']:.10f}")
        print(f"Mean Cycles: {statistics['mean_cycles']:.2f}")
        print(f"Std. Dev. Cycles: {statistics['std_cycles']:.2f}")
        print(f"Total Execution Time: {statistics['total_time']:.2f} seconds")
    else:
        # Run single optimization
        start_time = time.time()
        result = optimizer.optimize(
            objective_func=objective_func,
            max_cycles=args.max_cycles,
            tolerance=args.tolerance,
            verbose=True
        )
        end_time = time.time()
        
        # Print results
        print("\nOptimization Results:")
        print(f"Function: {args.function}")
        print(f"Best Position: {result['best_position']}")
        print(f"Best Fitness: {result['best_fitness']:.10f}")
        print(f"Cycles: {result['cycles']}")
        print(f"Execution Time: {end_time - start_time:.6f} seconds")
        print(f"Final Colony Diversity: {result['diversity_history'][-1]:.6f}")