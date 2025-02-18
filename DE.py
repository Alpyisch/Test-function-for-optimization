import numpy as np
import time
import argparse
from Optimization_Functions import OptimizationFunctions

class DEOptimizer:
    def __init__(self, lower_bound, upper_bound, population_size, dimension,
                 F=0.5, CR=0.7, strategy='DE/rand/1/bin'):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.dimension = dimension
        self.F = F  # Scaling factor
        self.CR = CR  # Crossover rate
        self.strategy = strategy
        
        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.population_size, np.inf)
        
        self.best_solution = None
        self.best_fitness = np.inf
        
        self.convergence_history = []
        self.diversity_history = []
        
        # Dictionary of mutation strategies
        self.mutation_strategies = {
            'DE/rand/1/bin': self._mutation_rand_1,
            'DE/best/1/bin': self._mutation_best_1,
            'DE/current-to-best/1/bin': self._mutation_current_to_best_1,
            'DE/best/2/bin': self._mutation_best_2,
            'DE/rand/2/bin': self._mutation_rand_2
        }
    
    def _initialize_population(self):
        population = np.zeros((self.population_size, self.dimension))
        for d in range(self.dimension):
            segment_size = (self.upper_bound - self.lower_bound) / self.population_size
            for i in range(self.population_size):
                population[i, d] = self.lower_bound + (i + np.random.random()) * segment_size
        return population
    
    def _mutation_rand_1(self, target_idx):
        # Select three random vectors, different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        # DE/rand/1 mutation
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        return mutant
    
    def _mutation_best_1(self, target_idx):
        # Select two random vectors, different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        
        # DE/best/1 mutation
        best_idx = np.argmin(self.fitness_scores)
        mutant = self.population[best_idx] + self.F * (self.population[r1] - self.population[r2])
        return mutant
    
    def _mutation_current_to_best_1(self, target_idx):
        # Select two random vectors, different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        
        # DE/current-to-best/1 mutation
        best_idx = np.argmin(self.fitness_scores)
        mutant = (self.population[target_idx] + 
                 self.F * (self.population[best_idx] - self.population[target_idx]) +
                 self.F * (self.population[r1] - self.population[r2]))
        return mutant
    
    def _mutation_best_2(self, target_idx):
        # Select four random vectors, different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
        
        # DE/best/2 mutation
        best_idx = np.argmin(self.fitness_scores)
        mutant = (self.population[best_idx] + 
                 self.F * (self.population[r1] + self.population[r2] - 
                          self.population[r3] - self.population[r4]))
        return mutant
    
    def _mutation_rand_2(self, target_idx):
        # Select five random vectors, different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        
        # DE/rand/2 mutation
        mutant = (self.population[r1] + 
                 self.F * (self.population[r2] + self.population[r3] - 
                          self.population[r4] - self.population[r5]))
        return mutant
    
    def _crossover(self, target, mutant):
        # Binomial crossover
        crossover_mask = np.random.random(self.dimension) < self.CR
        # Ensure at least one parameter is changed
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dimension)] = True
        
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def _ensure_bounds(self, vector):
        return np.clip(vector, self.lower_bound, self.upper_bound)
    
    def _calculate_population_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_position, axis=1)
        return np.mean(distances)
    
    def optimize(self, objective_func, max_generations=1000, tolerance=1e-6):
        # Initial evaluation
        self.fitness_scores = np.array([objective_func(ind) for ind in self.population])
        best_idx = np.argmin(self.fitness_scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_scores[best_idx]
        
        for generation in range(max_generations):
            for i in range(self.population_size):
                # Mutation
                mutant = self.mutation_strategies[self.strategy](i)
                mutant = self._ensure_bounds(mutant)
                
                # Crossover
                trial = self._crossover(self.population[i], mutant)
                trial = self._ensure_bounds(trial)
                
                # Selection
                trial_fitness = objective_func(trial)
                if trial_fitness < self.fitness_scores[i]:
                    self.population[i] = trial
                    self.fitness_scores[i] = trial_fitness
                    
                    # Update best solution
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial.copy()
                        self.best_fitness = trial_fitness
            
            # Track metrics
            self.convergence_history.append(self.best_fitness)
            self.diversity_history.append(self._calculate_population_diversity())
            
            if self.best_fitness <= tolerance:
                break
        
        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness,
            'generations': generation + 1,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Differential Evolution Optimization')
    parser.add_argument('--function', type=str, required=True, help='Function to optimize')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--dimension', type=int, default=2, help='Problem dimension')
    parser.add_argument('--F', type=float, default=0.5, help='Scaling factor')
    parser.add_argument('--CR', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--strategy', type=str, default='DE/rand/1/bin', 
                       choices=['DE/rand/1/bin', 'DE/best/1/bin', 'DE/current-to-best/1/bin',
                               'DE/best/2/bin', 'DE/rand/2/bin'],
                       help='DE strategy')
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
        'Bohachevsky': (-100, 100),
        'Perm_0': (-10, 10), ##bu kodun değerleri aynı olmak şartı ile eksi ile artı aralığında değiştirilebilir.(-d,d)
        'Rotated_Hyper-Ellipsoid': (-65.536, 65.536),
        'Sum of Different Powers': (-1, 1), 
        'McCormick': ((-1.5, 4),(-3,4)),
        'Trid': (-10, 10), ## bu kodun değerleri değiştirilecek
        'Power Sum': (0, 10), #bu kodun değerleri 0 ile herhangi bir eksi olmayan değerler arasında değiştirilebilir.(0, d)
        'cDe Jong': (-65.536, 65.536)
    }
    
    if args.function not in bounds:
        raise ValueError(f"Function '{args.function}' is not implemented or bounds are not defined.")
    
    lower_bound, upper_bound = bounds[args.function]
    
    # Initialize optimizer
    optimizer = DEOptimizer(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        population_size=args.population,
        dimension=args.dimension,
        F=args.F,
        CR=args.CR,
        strategy=args.strategy
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