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
    
    def optimize(self, objective_func, max_cycles=1000, tolerance=1e-6):
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
            
            if self.best_fitness <= tolerance:
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
        # Reset optimizer for new trial
        optimizer.food_sources = optimizer._initialize_food_sources()
        optimizer.fitness_scores = np.full(optimizer.employed_bees, np.inf)
        optimizer.trial_counts = np.zeros(optimizer.employed_bees)
        optimizer.best_solution = None
        optimizer.best_fitness = np.inf
        optimizer.convergence_history = []
        optimizer.diversity_history = []
        
        # Run optimization
        result = optimizer.optimize(
            objective_func=objective_func,
            max_cycles=max_cycles,
            tolerance=tolerance
        )
        
        all_results.append(result)
        
        # Update overall best if necessary
        if result['best_fitness'] < best_fitness_overall:
            best_fitness_overall = result['best_fitness']
            best_position_overall = result['best_position'].copy()
    
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
    parser = argparse.ArgumentParser(description='Artificial Bee Colony Optimization')
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
        'sphere': (-5.12, 5.12),
        'ackley': (-32.768, 32.768),
        'three_hump_camel': (-5, 5),
        'six_hump_camel': ((-3, 3),(-2, 2)),
        'dixon_price': (-10, 10),
        'rosenbrock': (-5, 10),
        'beale': (-4.5, 4.5),
        'branin': ((-5, 10),(0, 15)),
        'colville': (-10, 10),
        'forrester': (0, 1),
        'goldstein_price': (-2, 2),
        'hartmann_3d': (0, 1),
        'hartmann_4d': (0, 1),
        'hartmann_6d': (0, 1),
        'perm': (-4, 4), ##bu kodun değerleri aynı olmak şartı ile eksi ile artı aralığında değiştirilebilir.(-d,d)
        'powell': (-4, 5),
        'shekel': (0, 10),
        'styblinski_tang': (-5, 5),
        'bukin': ((-15, 15),(-3, 3)),
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
        'bohachevsky': (-100, 100), ## after this spot addings are made by yasin
        'perm_0': (-10, 10), ##bu kodun değerleri aynı olmak şartı ile eksi ile artı aralığında değiştirilebilir.(-d,d)
        'rotated_hyper-ellipsoid': (-65.536, 65.536),
        'sum of different powers': (-1, 1), 
        'mccormick': ((-1.5, 4),(-3,4)),
        'trid': (-10, 10), ## bu kodun değerleri değiştirilecek
        'power sum': (0, 10), #bu kodun değerleri 0 ile herhangi bir eksi olmayan değerler arasında değiştirilebilir.(0, d)
        'cde jong': (-65.536, 65.536)
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
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        objective_func=objective_func,
        max_cycles=args.max_cycles,
        tolerance=args.tolerance
    )
    end_time = time.time()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Function: {args.function}")
    print(f"Best Position: {result['best_position']}")
    print(f"Best Fitness: {result['best_fitness']:.10f}")
    print(f"Cycles: {result['cycles']}")
    print(f"Execution Time: {end_time - start_time:.6f} seconds")  # Time in milliseconds
    if args.trials > 1:
        print(f"Total Time: {statistics['total_time']:.2f} seconds")
    print(f"Final Colony Diversity: {result['diversity_history'][-1]:.6f}")