import numpy as np
import time
import argparse
from Optimization_Functions import OptimizationFunctions

class GeneticOptimizer:
    def __init__(self, lower_bound, upper_bound, population_size, dimension,
                 mutation_rate=0.1, crossover_rate=0.8, elite_size=2):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.dimension = dimension
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.population_size, np.inf)
        self.best_individual = None
        self.best_fitness = np.inf
        
        self.convergence_history = []
        self.diversity_history = []
    
    def _initialize_population(self):
        population = np.zeros((self.population_size, self.dimension))
        for d in range(self.dimension):
            segment_size = (self.upper_bound - self.lower_bound) / self.population_size
            for i in range(self.population_size):
                population[i, d] = self.lower_bound + (i + np.random.random()) * segment_size
        return population
    
    def _tournament_selection(self, tournament_size=3):
        tournament_indices = np.random.choice(self.population_size, tournament_size, replace=False)
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            alpha = np.random.random(self.dimension)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        return parent1, parent2
    
    def _mutation(self, individual):
        mutation_mask = np.random.random(self.dimension) < self.mutation_rate
        if np.any(mutation_mask):
            mutation_values = np.random.uniform(
                self.lower_bound, 
                self.upper_bound, 
                self.dimension
            )
            individual[mutation_mask] = mutation_values[mutation_mask]
        return individual
    
    def _calculate_population_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_position, axis=1)
        return np.mean(distances)
    
    def optimize(self, objective_func, max_generations=1000, tolerance=1e-6):
        for generation in range(max_generations):
            # Evaluate fitness
            self.fitness_scores = np.array([objective_func(ind) for ind in self.population])
            
            # Update best solution
            min_fitness_idx = np.argmin(self.fitness_scores)
            if self.fitness_scores[min_fitness_idx] < self.best_fitness:
                self.best_fitness = self.fitness_scores[min_fitness_idx]
                self.best_individual = self.population[min_fitness_idx].copy()
            
            # Track metrics
            self.convergence_history.append(self.best_fitness)
            self.diversity_history.append(self._calculate_population_diversity())
            
            if self.best_fitness <= tolerance:
                break
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(self.fitness_scores)[:self.elite_size]
            new_population = [self.population[idx].copy() for idx in elite_indices]
            
            # Create new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                # Add to new population
                new_population.extend([child1, child2])
            
            # Trim if necessary and convert to numpy array
            self.population = np.array(new_population[:self.population_size])
            
            # Ensure bounds
            np.clip(self.population, self.lower_bound, self.upper_bound, out=self.population)
        
        return {
            'best_position': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations': generation + 1,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Optimization')
    parser.add_argument('--function', type=str, required=True, help='Function to optimize')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--dimension', type=int, default=2, help='Problem dimension')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('--max-generations', type=int, default=1000, help='Maximum generations')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance')
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
        'perm': (-2, 2), 
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
        'Bohachevsky': (-100, 100),
        'Perm_0': (-2, 2), 
        'Rotated_Hyper-Ellipsoid': (-65.536, 65.536),
        'Sum of Different Powers': (-1, 1), 
        'McCormick': ((-1.5, 4),(-3,4)),
        'Trid': (-2, 2), 
        'Power Sum': (0, 2), 
        'cDe Jong': (-65.536, 65.536)
    }
    
    if args.function not in bounds:
        raise ValueError(f"Function '{args.function}' is not implemented or bounds are not defined.")
    
    lower_bound, upper_bound = bounds[args.function]
    
    # Initialize optimizer
    optimizer = GeneticOptimizer(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        population_size=args.population,
        dimension=args.dimension,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    
    # Initialize optimization functions
    opt_functions = OptimizationFunctions()
    objective_func = getattr(opt_functions, f"{args.function}_function")
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        objective_func=objective_func,
        max_generations=args.max_generations,
        tolerance=args.tolerance
    )
    end_time = time.time()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Function: {args.function}")
    print(f"Best Position: {result['best_position']}")
    print(f"Best Fitness: {result['best_fitness']:.10f}")
    print(f"Generations: {result['generations']}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Final Population Diversity: {result['diversity_history'][-1]:.6f}")