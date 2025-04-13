import numpy as np
import time
import argparse
from Optimization_Functions import OptimizationFunctions

class GENOptimizer:
    def __init__(self, lower_bound, upper_bound, population_size, dimension, mutation_rate=0.1, crossover_rate=0.9):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.dimension = dimension
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.population_size, np.inf)
        
        self.best_solution = None
        self.best_fitness = np.inf
        
        self.convergence_history = []
        self.diversity_history = []
    
    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
    
    def _calculate_fitness(self, objective_func):
        return np.array([objective_func(ind) for ind in self.population])
    
    def _mutate(self, parent):
        mutant = parent + self.mutation_rate * np.random.uniform(-1, 1, self.dimension)
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def _crossover(self, parent, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dimension)] = True
        offspring = np.where(crossover_mask, mutant, parent)
        return offspring
    
    def _calculate_population_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_position, axis=1)
        return np.mean(distances)
    
    def optimize(self, objective_func, max_generations=1000, tolerance=1e-6, verbose=True):
        if verbose:
            print("\nOptimization Progress:")
            
        for generation in range(max_generations):
            fitness = self._calculate_fitness(objective_func)
            
            for i in range(self.population_size):
                parent = self.population[i]
                mutant = self._mutate(parent)
                offspring = self._crossover(parent, mutant)
                offspring_fitness = objective_func(offspring)
                
                if offspring_fitness < fitness[i]:
                    self.population[i] = offspring
                    fitness[i] = offspring_fitness
            
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.population[min_fitness_idx].copy()
            
            self.convergence_history.append(self.best_fitness)
            self.diversity_history.append(self._calculate_population_diversity())
            
            # Print current generation results if verbose
            if verbose and (generation % 100 == 0 or generation == max_generations - 1):
                print(f"\nGeneration {generation + 1}:")
                print("Current Best Position:")
                for i, pos in enumerate(self.best_solution):
                    print(f"x{i+1}: {pos:.10f}")
                print(f"Current Best Fitness: {self.best_fitness:.10f}")
                print(f"Current Population Diversity: {self.diversity_history[-1]:.6f}")
            
            if self.best_fitness <= tolerance:
                if verbose:
                    print(f"\nConvergence achieved at generation {generation + 1}")
                break
        
        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness,
            'generations': generation + 1,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }

def run_simulation(optimizer, objective_func, num_trials, max_generations, tolerance):
    all_results = []
    best_fitness_overall = np.inf
    best_position_overall = None
    total_start_time = time.time()
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Reset optimizer for new trial
        optimizer.population = optimizer._initialize_population()
        optimizer.fitness_scores = np.full(optimizer.population_size, np.inf)
        optimizer.best_solution = None
        optimizer.best_fitness = np.inf
        optimizer.convergence_history = []
        optimizer.diversity_history = []
        
        # Run optimization with reduced verbosity for all but the first trial
        verbose = (trial == 0)
        result = optimizer.optimize(
            objective_func=objective_func,
            max_generations=max_generations,
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
        print(f"Generations: {result['generations']}")
    
    # Calculate statistics
    fitness_values = [r['best_fitness'] for r in all_results]
    generations_values = [r['generations'] for r in all_results]
    
    statistics = {
        'best_fitness_overall': best_fitness_overall,
        'best_position_overall': best_position_overall,
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values),
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'mean_generations': np.mean(generations_values),
        'std_generations': np.std(generations_values),
        'total_time': time.time() - total_start_time,
        'num_trials': num_trials
    }
    
    return statistics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')
    parser.add_argument('--function', type=str, required=True, help='Function to optimize')
    parser.add_argument('--population-size', type=int, default=50, help='Population size')
    parser.add_argument('--dimension', type=int, default=2, help='Problem dimension')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.9, help='Crossover rate')
    parser.add_argument('--max-generations', type=int, default=1000, help='Maximum generations')
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

    # Get optimization function
    opt_functions = OptimizationFunctions()
    objective_func = getattr(opt_functions, f"{args.function}_function")

    # Create optimizer
    optimizer = GENOptimizer(
        lower_bound=bounds[args.function][0],
        upper_bound=bounds[args.function][1],
        population_size=args.population_size,
        dimension=args.dimension,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )

    if args.trials > 1:
        results = run_simulation(
            optimizer=optimizer,
            objective_func=objective_func,
            num_trials=args.trials,
            max_generations=args.max_generations,
            tolerance=args.tolerance
        )
        
        # Print simulation results
        print("\nSimulation Results:")
        print(f"Best Fitness Overall: {results['best_fitness_overall']:.10f}")
        print(f"Mean Fitness: {results['mean_fitness']:.10f}")
        print(f"Std. Dev. Fitness: {results['std_fitness']:.10f}")
        print(f"Mean Generations: {results['mean_generations']:.1f}")
        print(f"Total Time: {results['total_time']:.2f} seconds")
    else:
        # Run single optimization
        results = optimizer.optimize(
            objective_func=objective_func,
            max_generations=args.max_generations,
            tolerance=args.tolerance
        )
        
        # Print single run results
        print("\nOptimization Results:")
        print("Best Position:")
        for i, pos in enumerate(results['best_position']):
            print(f"x{i+1}: {pos:.10f}")
        print(f"Best Fitness: {results['best_fitness']:.10f}")
        print(f"Generations: {results['generations']}")