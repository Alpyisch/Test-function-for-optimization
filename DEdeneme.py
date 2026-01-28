import numpy as np
import time
import argparse
import sys
from Optimization_Functions import OptimizationFunctions

class DEOptimizer:
    def __init__(self, lower_bound, upper_bound, population_size, dimension,
                 F=0.9, CR=0.9, strategy='DE/rand/1/bin'):

        self.lower_bound = np.array(lower_bound) if isinstance(lower_bound, (list, np.ndarray)) else lower_bound
        self.upper_bound = np.array(upper_bound) if isinstance(upper_bound, (list, np.ndarray)) else upper_bound

        self.population_size = population_size
        self.dimension = dimension
        self.F = F
        self.CR = CR
        self.strategy = strategy

        self.population = self._initialize_population()
        self.fitness_scores = np.full(self.population_size, np.inf)

        self.best_solution = None
        self.best_fitness = np.inf

        self.mutation_strategies = {
            'DE/rand/1/bin': self._mutation_rand_1,
            'DE/best/1/bin': self._mutation_best_1,
            'DE/current-to-best/1/bin': self._mutation_current_to_best_1,
            'DE/best/2/bin': self._mutation_best_2,
            'DE/rand/2/bin': self._mutation_rand_2
        }

    def _initialize_population(self):
        pop = np.zeros((self.population_size, self.dimension))
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            pop[:, d] = np.random.uniform(lb, ub, self.population_size)
        return pop

    def _mutation_rand_1(self, idx):
        r1, r2, r3 = np.random.choice(
            [i for i in range(self.population_size) if i != idx], 3, replace=False
        )
        return self.population[r1] + self.F * (self.population[r2] - self.population[r3])

    def _mutation_best_1(self, idx):
        best = np.argmin(self.fitness_scores)
        r1, r2 = np.random.choice(
            [i for i in range(self.population_size) if i != idx], 2, replace=False
        )
        return self.population[best] + self.F * (self.population[r1] - self.population[r2])

    def _mutation_current_to_best_1(self, idx):
        best = np.argmin(self.fitness_scores)
        r1, r2 = np.random.choice(
            [i for i in range(self.population_size) if i != idx], 2, replace=False
        )
        return (self.population[idx]
                + self.F * (self.population[best] - self.population[idx])
                + self.F * (self.population[r1] - self.population[r2]))

    def _mutation_best_2(self, idx):
        best = np.argmin(self.fitness_scores)
        r1, r2, r3, r4 = np.random.choice(
            [i for i in range(self.population_size) if i != idx], 4, replace=False
        )
        return self.population[best] + self.F * (
            self.population[r1] + self.population[r2]
            - self.population[r3] - self.population[r4]
        )

    def _mutation_rand_2(self, idx):
        r1, r2, r3, r4, r5 = np.random.choice(
            [i for i in range(self.population_size) if i != idx], 5, replace=False
        )
        return self.population[r1] + self.F * (
            self.population[r2] + self.population[r3]
            - self.population[r4] - self.population[r5]
        )


    def _crossover(self, target, mutant):
        mask = np.random.rand(self.dimension) < self.CR
        if not np.any(mask):
            mask[np.random.randint(0, self.dimension)] = True
        return np.where(mask, mutant, target)

    def _ensure_bounds(self, vec):
        res = vec.copy()
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            res[d] = np.clip(res[d], lb, ub)
        return res

    def optimize(self, objective_func, max_generations=3000, tolerance=1e-6):
        try:
            self.fitness_scores = np.array([objective_func(ind) for ind in self.population])
        except:
            return None

        best_idx = np.argmin(self.fitness_scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_scores[best_idx]

        for _ in range(max_generations):
            for i in range(self.population_size):
                mutant = self.mutation_strategies[self.strategy](i)
                mutant = self._ensure_bounds(mutant)

                trial = self._crossover(self.population[i], mutant)
                trial = self._ensure_bounds(trial)

                fit = objective_func(trial)
                if fit < self.fitness_scores[i]:
                    self.population[i] = trial
                    self.fitness_scores[i] = fit
                    if fit < self.best_fitness:
                        self.best_fitness = fit
                        self.best_solution = trial.copy()

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
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--max-generations', type=int, default=1000)
    parser.add_argument('--F', type=float, default=0.9)
    parser.add_argument('--CR', type=float, default=0.9)
    args = parser.parse_args()
    target_func = args.function.lower()
    
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


    if target_func not in bounds: sys.exit(f"Error: {target_func} not found in bounds.")
    
    if target_func in fixed_dims:
        current_dim = fixed_dims[target_func]
        dim_info = "(SABİT)"
    else:
        current_dim = args.dim
        dim_info = "(ESNEK)"

    bound_data = bounds[target_func]
    if isinstance(bound_data[0], (list, tuple, np.ndarray)):
        lb, ub = np.array(bound_data[0]), np.array(bound_data[1])
    else:
        lb, ub = np.full(current_dim, bound_data[0]), np.full(current_dim, bound_data[1])

    opt_funcs = OptimizationFunctions()
    m_name = function_map.get(target_func, f"{target_func}_function")
    if not hasattr(opt_funcs, m_name): m_name = target_func
    obj_func = getattr(opt_funcs, m_name)
    theo_fit, theo_pos = get_theoretical_values(target_func, current_dim)

    print("\n"+"="*145)
    print(f"ALGORİTMA: DE | FONKSİYON: {target_func.upper()} | DIM: {current_dim} {dim_info}")
    print(f"POPULASYON: {args.population} | GENERATION: {args.max_generations}")
    print(f"HEDEF: {theo_fit}")
    print("=" * 145)
    print(f"{'No':<4} {'Best Fitness':<18} {'Time(s)':<10} {'Parametre 1':<15} {'Parametre 2':<15} {'Parametre 3':<15} {'Parametre 4':<15} {'Parametre 5':<15} {'Parametre 6':<15}")
    print("-" * 145)

    fits, times = [], []
    for i in range(1, args.trials+1):
        opt = DEOptimizer(lb, ub, args.population, current_dim, args.F, args.CR)
        st = time.time()
        res = opt.optimize(obj_func, args.max_generations)
        et = time.time()
        
        run_time = et - st
        fit, pos = res['best_fitness'], res['best_position']
        fits.append(fit); times.append(run_time)
        
        params = []
        for j in range(6):
            if j < len(pos): params.append(f"{pos[j]:<15.4f}")
            else: params.append(f"{'-':<15}")
            
        print(f"{i:<4} {fit:<18.8f} {run_time:<10.4f} {params[0]}{params[1]}{params[2]}{params[3]}{params[4]}{params[5]}")

    print("="*145)
    print(f"ORTALAMA FITNESS: {np.mean(fits):.10f}")
    print(f"EN İYİ FITNESS  : {np.min(fits):.10f}")
    print(f"ORTALAMA SÜRE   : {np.mean(times):.4f} sn")
    print("="*145)