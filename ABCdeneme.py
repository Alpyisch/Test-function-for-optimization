import numpy as np
import time
import argparse
import sys
from Optimization_Functions import OptimizationFunctions

class ABCOptimizer:
    def __init__(self, lower_bound, upper_bound, colony_size, dimension, limit=50):
        self.lower_bound = np.array(lower_bound) if isinstance(lower_bound, (list, np.ndarray)) else lower_bound
        self.upper_bound = np.array(upper_bound) if isinstance(upper_bound, (list, np.ndarray)) else upper_bound
        self.colony_size = colony_size
        self.dimension = dimension
        self.employed_bees = colony_size // 2
        self.onlooker_bees = colony_size - self.employed_bees
        self.limit = limit

        self.food_sources = self._initialize_food_sources()
        self.fitness_scores = np.full(self.employed_bees, np.inf)
        self.trial_counts = np.zeros(self.employed_bees, dtype=int)

        self.best_solution = None
        self.best_fitness = np.inf

    # ---------------- INITIALIZATION ----------------
    def _initialize_food_sources(self):
        food = np.zeros((self.employed_bees, self.dimension))
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            food[:, d] = np.random.uniform(lb, ub, self.employed_bees)
        return food

    # ---------------- FITNESS ----------------
    @staticmethod
    def _fitness_transform(value):
        return 1 / (1 + value) if value >= 0 else 1 + abs(value)

    def _calculate_probabilities(self):
        transformed = np.array([self._fitness_transform(v) for v in self.fitness_scores])
        total = np.sum(transformed)
        return transformed / total if total != 0 else np.full(len(transformed), 1 / len(transformed))

    # ---------------- NEIGHBOR ----------------
    def _generate_neighbor(self, idx):
        partner_idx = np.random.choice([i for i in range(self.employed_bees) if i != idx])
        phi = np.random.uniform(-1, 1, self.dimension)

        current = self.food_sources[idx]
        partner = self.food_sources[partner_idx]

        neighbor = current + phi * (current - partner)

        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
            neighbor[d] = np.clip(neighbor[d], lb, ub)

        return neighbor

    # ---------------- SCOUT ----------------
    def _scout_phase(self):
        abandoned = np.where(self.trial_counts >= self.limit)[0]
        for idx in abandoned:
            for d in range(self.dimension):
                lb = self.lower_bound[d] if isinstance(self.lower_bound, np.ndarray) else self.lower_bound
                ub = self.upper_bound[d] if isinstance(self.upper_bound, np.ndarray) else self.upper_bound
                self.food_sources[idx, d] = np.random.uniform(lb, ub)

            self.trial_counts[idx] = 0
            self.fitness_scores[idx] = np.inf

    # ---------------- OPTIMIZATION ----------------
    def optimize(self, objective_func, max_cycles=3000, tolerance=1e-6):
        # Initial evaluation
        for i in range(self.employed_bees):
            try:
                fit = objective_func(self.food_sources[i])
            except:
                fit = np.inf

            self.fitness_scores[i] = fit
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = self.food_sources[i].copy()

        # Main loop
        for _ in range(max_cycles):

            # Employed Bees
            for i in range(self.employed_bees):
                candidate = self._generate_neighbor(i)
                fit = objective_func(candidate)

                if fit < self.fitness_scores[i]:
                    self.food_sources[i] = candidate
                    self.fitness_scores[i] = fit
                    self.trial_counts[i] = 0
                else:
                    self.trial_counts[i] += 1

            # Onlooker Bees
            probs = self._calculate_probabilities()
            count = 0
            i = 0
            while count < self.onlooker_bees:
                if np.random.rand() < probs[i]:
                    candidate = self._generate_neighbor(i)
                    fit = objective_func(candidate)

                    if fit < self.fitness_scores[i]:
                        self.food_sources[i] = candidate
                        self.fitness_scores[i] = fit
                        self.trial_counts[i] = 0
                    else:
                        self.trial_counts[i] += 1
                    count += 1
                i = (i + 1) % self.employed_bees

            # Scout + Global Best
            self._scout_phase()
            idx = np.argmin(self.fitness_scores)
            if self.fitness_scores[idx] < self.best_fitness:
                self.best_fitness = self.fitness_scores[idx]
                self.best_solution = self.food_sources[idx].copy()

            if self.best_fitness <= tolerance:
                break

        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness
        }

def get_theoretical_values(func_name, dim):
    val = 0.0
    if func_name == 'eggholder': return -959.6407, [512, 404.2319]
    elif func_name == 'styblinski_tang': return -39.166 * dim, [-2.9035]*dim
    return 0.0, [val] * dim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--colony-size', type=int, default=100)
    parser.add_argument('--max-cycles', type=int, default=3000)
    
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

    fixed_dims = {
        'colville':4,'hartmann_3d':3,'hartmann_4d':4,'hartmann_6d':6,
        'powell':4,'shekel':4,'eggholder':2,'goldstein_price':2,
        'six_hump_camel':2,'beale':2,'branin':2,'bukin_n6':2,
        'mccormick':2,'drop_wave':2,'three_hump_camel':2
    }
    
    function_map = {
        'bohachevsky':'bohacevsky_function',
        'perm_0':'perm_0_d_beta_function',
        'bukin':'bukin_n6_function',
        'bukin_n6':'bukin_n6_function',
        'dejong_n5': 'dejong_n5_function'
    }

    if target_func not in bounds: sys.exit(f"Error: {target_func} not found.")
    current_dim = fixed_dims.get(target_func, args.dim)
    bound_data = bounds[target_func]

    # BOUNDS İŞLEME MANTIĞI
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
    print(f"ALGORİTMA: ABC | FONKSİYON: {target_func.upper()} | DIM: {current_dim}")
    print(f"KOLONİ: {args.colony_size} | DÖNGÜ: {args.max_cycles}")
    print(f"HEDEF: {theo_fit}")
    print("=" * 85)
    print(f"{'No':<4} {'Best Fitness':<20} {'Time(s)':<10} {'Param 1':<15} {'Param 2':<15}")
    print("-" * 85)

    fits, times = [], []
    for i in range(1, args.trials+1):
        opt = ABCOptimizer(lb, ub, args.colony_size, current_dim)
        st = time.time()
        res = opt.optimize(obj_func, args.max_cycles)
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