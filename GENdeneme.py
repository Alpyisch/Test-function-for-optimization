import numpy as np
import time
import argparse
import sys
from Optimization_Functions import OptimizationFunctions

class GENOptimizer:
    def __init__(self, lower_bound, upper_bound, population_size, dimension,
                 mutation_rate=0.05, crossover_rate=0.95):

        self.lower_bound = np.array(lower_bound) if isinstance(lower_bound, (list, np.ndarray)) else lower_bound
        self.upper_bound = np.array(upper_bound) if isinstance(upper_bound, (list, np.ndarray)) else upper_bound

        self.population_size = population_size
        self.dimension = dimension
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = np.inf

    # ---------------- INITIALIZATION ----------------
    def _initialize_population(self):
        pop = np.zeros((self.population_size, self.dimension))
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            pop[:, d] = np.random.uniform(lb, ub, self.population_size)
        return pop

    # ---------------- SELECTION (Tournament) ----------------
    def _tournament_selection(self, fitness, k=2):
        idxs = np.random.choice(self.population_size, k, replace=False)
        best = idxs[np.argmin(fitness[idxs])]
        return self.population[best]

    # ---------------- CROSSOVER ----------------
    def _crossover(self, parent1, parent2):
        child = parent1.copy()
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dimension)
            child[point:] = parent2[point:]
        return child

    # ---------------- MUTATION ----------------
    def _mutation(self, individual):
        if np.random.rand() < self.mutation_rate:
            if isinstance(self.upper_bound, np.ndarray):
                sigma = 0.1 * (self.upper_bound - self.lower_bound)
            else:
                sigma = 0.1 * (self.upper_bound - self.lower_bound)
            individual += np.random.normal(0, sigma, self.dimension)
        return individual

    # ---------------- BOUND CONTROL ----------------
    def _ensure_bounds(self, individual):
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            individual[d] = np.clip(individual[d], lb, ub)
        return individual

    # ---------------- OPTIMIZATION ----------------
    def optimize(self, objective_func, max_generations=3000, tolerance=1e-6):
        try:
            fitness = np.array([objective_func(ind) for ind in self.population])
        except:
            return None

        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()

        for _ in range(max_generations):
            new_population = []

            # Elitism (Best 2)
            elite_idx = np.argsort(fitness)[:2]
            new_population.extend(self.population[elite_idx])

            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(fitness)
                p2 = self._tournament_selection(fitness)

                child = self._crossover(p1, p2)
                child = self._mutation(child)
                child = self._ensure_bounds(child)

                new_population.append(child)

            self.population = np.array(new_population)
            fitness = np.array([objective_func(ind) for ind in self.population])

            curr_best = np.argmin(fitness)
            if fitness[curr_best] < self.best_fitness:
                self.best_fitness = fitness[curr_best]
                self.best_solution = self.population[curr_best].copy()

            if self.best_fitness <= tolerance:
                break

        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness
        }

def get_theoretical_values(func_name, dim):
    val = 0.0
    if func_name == 'eggholder': return -959.6407, [512, 404.2319]
    return 0.0, [val] * dim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--max-generations', type=int, default=3000)
    args = parser.parse_args()
    target_func = args.function.lower()
    
    # --- GÜNCELLENMİŞ BOUNDS LİSTESİ ---
    bounds = {
        'ackley': (-32.768, 32.768), 
        'three_hump_camel': (-5, 5), 
        'six_hump_camel': ([-3, -2], [3, 2]),
        'dixon_price': (-10, 10), 
        'rosenbrock': (-5, 10), 
        'beale': (-4.5, 4.5),
        'branin': ([-5, 0], [10, 15]), 
        'colville': (-10, 10), 
        'forrester': (0, 1),
        'goldstein_price': (-2, 2), 
        'hartmann_3d': (0, 1), 
        'hartmann_4d': (0, 1),
        'hartmann_6d': (0, 1), 
        'perm': (-2, 2), # bu kısmın dimeition'a göre ayarlanması gerekli!
        'powell': (-4, 5),
        'shekel': (0, 10), 
        'styblinski_tang': (-5, 5), 
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
        'shubert': (-10, 10), #[-5.12, 5.12] arlığında da denenebilir
        'michalewicz': (0, np.pi),
        'easom': (-100, 100), 
        'booth': (-10, 10), 
        'matyas': (-10, 10),
        'zakharov': (-5, 10), 
        'bohachevsky': (-100, 100), 
        'perm_0': (-2, 2), #bu kısmın dimeition'a göre ayarlanması gerekli!
        'rotated_hyper_ellipsoid': (-65.536, 65.536), 
        'sphere': (-5.12, 5.12),
        'sum_of_different_powers': (-1, 1), 
        'mccormick': ([-1.5, -3], [4, 4]),
        'trid': (-2, 2), #[-dkare, dkare] arlığında da denenebilir
        'power_sum': (0, 2), #[0, d] arlığında da denenebilir
        'dejong_n5': (-65.536, 65.536),
        'sum_squares': (-10, 10), #[-5.12, 5.12] arlığında da denenebilir
        'bukin_n6': ([-15, -3], [-5, 3])
    }
    
    fixed_dims = {'colville':4,'hartmann_3d':3,'hartmann_4d':4,'hartmann_6d':6,'powell':4,'shekel':4,'eggholder':2,'goldstein_price':2,'six_hump_camel':2,'beale':2,'branin':2,'bukin_n6':2, 'mccormick':2}
    function_map = {'bohachevsky':'bohacevsky_function','perm_0':'perm_0_d_beta_function','bukin':'bukin_n6_function','bukin_n6':'bukin_n6_function', 'dejong_n5': 'dejong_n5_function'}

    if target_func not in bounds: sys.exit(f"Error: {target_func} not found.")
    current_dim = fixed_dims.get(target_func, args.dim)
    bound_data = bounds[target_func]

    if isinstance(bound_data[0], (list, tuple)):
        lb = np.array(bound_data[0])
        ub = np.array(bound_data[1])
    else:
        lb = np.full(current_dim, bound_data[0])
        ub = np.full(current_dim, bound_data[1])

    opt_funcs = OptimizationFunctions()
    m_name = function_map.get(target_func, f"{target_func}_function")
    if not hasattr(opt_funcs, m_name): m_name = target_func
    obj_func = getattr(opt_funcs, m_name)
    theo_fit, theo_pos = get_theoretical_values(target_func, current_dim)

    print("\n"+"="*85)
    print(f"ALGORİTMA: GEN | FONKSİYON: {target_func.upper()} | DIM: {current_dim}")
    print(f"POPULASYON: {args.population_size} | GENERATION: {args.max_generations}")
    print(f"HEDEF: {theo_fit}")
    print("=" * 85)
    print(f"{'No':<4} {'Best Fitness':<20} {'Time(s)':<10} {'Parametre 1':<15} {'Parametre 2':<15}")
    print("-" * 85)

    fits, times = [], []
    for i in range(1, args.trials+1):
        opt = GENOptimizer(lb, ub, args.population_size, current_dim)
        st = time.time()
        res = opt.optimize(obj_func, args.max_generations)
        et = time.time()
        
        run_time = et - st
        fit, pos = res['best_fitness'], res['best_position']
        fits.append(fit); times.append(run_time)
        
        p1 = pos[0] if current_dim>=1 else 0
        p2 = pos[1] if current_dim>=2 else 0
        print(f"{i:<4} {fit:<20.8f} {run_time:<10.4f} {p1:<15.4f} {p2:<15.4f}")

    print("="*85)
    print(f"ORTALAMA FITNESS: {np.mean(fits):.10f}")
    print(f"EN İYİ FITNESS  : {np.min(fits):.10f}")
    print(f"ORTALAMA SÜRE   : {np.mean(times):.4f} sn")
    print("="*85)