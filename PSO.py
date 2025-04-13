import numpy as np
import time
import argparse 
from Optimization_Functions import OptimizationFunctions

class PSOOptimizer:
    def __init__(self, lower_bound, upper_bound, particle_count, dimension, c1=2.0, c2=2.0,
                 w_max=0.9, w_min=0.4):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.particle_count = particle_count
        self.dimension = dimension
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        
        self.particles = self._initialize_particles()
        self.velocities = self._initialize_velocities()
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.particle_count, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        
        self.convergence_history = []
        self.diversity_history = []

    def _initialize_particles(self):
        particles = np.zeros((self.particle_count, self.dimension))
        for d in range(self.dimension):
            segment_size = (self.upper_bound - self.lower_bound) / self.particle_count
            for i in range(self.particle_count):
                particles[i, d] = self.lower_bound + (i + np.random.random()) * segment_size
        return particles

    def _initialize_velocities(self):
        velocity_range = 0.1 * (self.upper_bound - self.lower_bound)
        return np.random.uniform(-velocity_range, velocity_range, 
                               (self.particle_count, self.dimension))

    def _calculate_inertia_weight(self, current_iter, max_iter):
        return self.w_max - (self.w_max - self.w_min) * (current_iter / max_iter)

    def _update_velocity(self, current_iter, max_iter):
        w = self._calculate_inertia_weight(current_iter, max_iter)
        r1 = np.random.random((self.particle_count, self.dimension))
        r2 = np.random.random((self.particle_count, self.dimension))
        
        cognitive = self.c1 * r1 * (self.pbest_positions - self.particles)
        social = self.c2 * r2 * (self.gbest_position - self.particles)
        
        self.velocities = w * self.velocities + cognitive + social
        v_max = 0.2 * (self.upper_bound - self.lower_bound)
        np.clip(self.velocities, -v_max, v_max, out=self.velocities)

    def _update_position(self):
        self.particles += self.velocities
        np.clip(self.particles, self.lower_bound, self.upper_bound, out=self.particles)

    def _calculate_swarm_diversity(self):
        mean_position = np.mean(self.particles, axis=0)
        distances = np.linalg.norm(self.particles - mean_position, axis=1)
        return np.mean(distances)

    def optimize(self, objective_func, max_iterations=1000, tolerance=1e-6, verbose=True):
        if verbose:
            print("\nOptimization Progress:")
            
        for iteration in range(max_iterations):
            # Evaluate fitness
            fitness = np.array([objective_func(p) for p in self.particles])
            
            # Update personal bests
            improved_mask = fitness < self.pbest_scores
            self.pbest_scores[improved_mask] = fitness[improved_mask]
            self.pbest_positions[improved_mask] = self.particles[improved_mask]
            
            # Update global best
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.gbest_score:
                self.gbest_score = fitness[min_fitness_idx]
                self.gbest_position = self.particles[min_fitness_idx].copy()
            
            # Track metrics
            self.convergence_history.append(self.gbest_score)
            self.diversity_history.append(self._calculate_swarm_diversity())
            
            # Print current iteration results if verbose
            if verbose and (iteration % 100 == 0 or iteration == max_iterations - 1):
                print(f"\nIteration {iteration + 1}:")
                print("Current Best Position:")
                for i, pos in enumerate(self.gbest_position):
                    print(f"x{i+1}: {pos:.10f}")
                print(f"Current Best Fitness: {self.gbest_score:.10f}")
                print(f"Current Swarm Diversity: {self.diversity_history[-1]:.6f}")
            
            if self.gbest_score <= tolerance:
                if verbose:
                    print(f"\nConvergence achieved at iteration {iteration + 1}")
                break
                
            self._update_velocity(iteration, max_iterations)
            self._update_position()
        
        return {
            'best_position': self.gbest_position,
            'best_fitness': self.gbest_score,
            'iterations': iteration + 1,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history
        }

def run_simulation(optimizer, objective_func, num_trials, max_iterations, tolerance):
    all_results = []
    best_fitness_overall = np.inf
    best_position_overall = None
    total_start_time = time.time()
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Reset optimizer for new trial
        optimizer.particles = optimizer._initialize_particles()
        optimizer.velocities = optimizer._initialize_velocities()
        optimizer.pbest_positions = np.copy(optimizer.particles)
        optimizer.pbest_scores = np.full(optimizer.particle_count, np.inf)
        optimizer.gbest_position = None
        optimizer.gbest_score = np.inf
        optimizer.convergence_history = []
        optimizer.diversity_history = []
        
        # Run optimization with reduced verbosity for all but the first trial
        verbose = (trial == 0)
        result = optimizer.optimize(
            objective_func=objective_func,
            max_iterations=max_iterations,
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
        print(f"Iterations: {result['iterations']}")
    
    # Calculate statistics
    fitness_values = [r['best_fitness'] for r in all_results]
    iterations_values = [r['iterations'] for r in all_results]
    
    statistics = {
        'best_fitness_overall': best_fitness_overall,
        'best_position_overall': best_position_overall,
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values),
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'mean_iterations': np.mean(iterations_values),
        'std_iterations': np.std(iterations_values),
        'total_time': time.time() - total_start_time,
        'num_trials': num_trials
    }
    
    return statistics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Particle Swarm Optimization Algorithm Parameters')
    parser.add_argument('--function', type=str, required=True, help='Function to optimize')
    parser.add_argument('--particles', type=int, default=50, help='Number of particles')
    parser.add_argument('--dimension', type=int, default=2, help='Problem dimension')
    parser.add_argument('--inertia', type=float, default=0.7, help='Inertia weight')
    parser.add_argument('--cognitive', type=float, default=1.5, help='Cognitive coefficient')
    parser.add_argument('--social', type=float, default=1.5, help='Social coefficient')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum iterations')
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
    optimizer = PSOOptimizer(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        particle_count=args.particles,
        dimension=args.dimension,
        c1=args.cognitive,
        c2=args.social
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
            max_iterations=args.max_iterations,
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
        print(f"Mean Iterations: {statistics['mean_iterations']:.2f}")
        print(f"Std. Dev. Iterations: {statistics['std_iterations']:.2f}")
        print(f"Total Execution Time: {statistics['total_time']:.2f} seconds")
    else:
        # Run single optimization
        start_time = time.time()
        result = optimizer.optimize(
            objective_func=objective_func,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            verbose=True
        )
        end_time = time.time()
        
        # Print results
        print("\nOptimization Results:")
        print(f"Function: {args.function}")
        print(f"Best Position: {result['best_position']}")
        print(f"Best Fitness: {result['best_fitness']:.10f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Execution Time: {end_time - start_time:.6f} seconds")
        print(f"Final Swarm Diversity: {result['diversity_history'][-1]:.6f}")