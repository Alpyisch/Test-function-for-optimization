import numpy as np
import time
import argparse
import sys
from Optimization_Functions import OptimizationFunctions

class PSOOptimizer:
    def __init__(self, lower_bound, upper_bound, particle_count, dimension, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
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

    def _initialize_particles(self):
        particles = np.zeros((self.particle_count, self.dimension))
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, (list, np.ndarray)) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, (list, np.ndarray)) else self.upper_bound
            particles[:, d] = np.random.uniform(lb, ub, self.particle_count)
        return particles

    def _initialize_velocities(self):
        velocities = np.zeros((self.particle_count, self.dimension))
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, (list, np.ndarray)) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, (list, np.ndarray)) else self.upper_bound
            v_range = 0.1 * (ub - lb)
            velocities[:, d] = np.random.uniform(-v_range, v_range, self.particle_count)
        return velocities

    def _update_velocity(self, current_iter, max_iter):
        w = self.w_max - (self.w_max - self.w_min) * (current_iter / max_iter)
        r1 = np.random.random((self.particle_count, self.dimension))
        r2 = np.random.random((self.particle_count, self.dimension))
        
        cognitive = self.c1 * r1 * (self.pbest_positions - self.particles)
        social = self.c2 * r2 * (self.gbest_position - self.particles)
        
        self.velocities = w * self.velocities + cognitive + social
        
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, (list, np.ndarray)) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, (list, np.ndarray)) else self.upper_bound
            v_max = 0.2 * (ub - lb)
            np.clip(self.velocities[:, d], -v_max, v_max, out=self.velocities[:, d])

    def _update_position(self):
        self.particles += self.velocities
        for d in range(self.dimension):
            lb = self.lower_bound[d] if isinstance(self.lower_bound, (list, np.ndarray)) else self.lower_bound
            ub = self.upper_bound[d] if isinstance(self.upper_bound, (list, np.ndarray)) else self.upper_bound
            np.clip(self.particles[:, d], lb, ub, out=self.particles[:, d])

    def optimize(self, objective_func, max_iterations=3000, tolerance=1e-6):
        for iteration in range(max_iterations):
            try:
                fitness = np.array([objective_func(p) for p in self.particles])
            except: return None

            improved = fitness < self.pbest_scores
            self.pbest_scores[improved] = fitness[improved]
            self.pbest_positions[improved] = self.particles[improved]
            
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.gbest_score:
                self.gbest_score = fitness[min_idx]
                self.gbest_position = self.particles[min_idx].copy()
            
            if self.gbest_score <= -999999: 
                break
                
            self._update_velocity(iteration, max_iterations)
            self._update_position()
            
        return {'best_position': self.gbest_position, 'best_fitness': self.gbest_score}

def get_theoretical_values(func_name, dim):
    theo_pos = None 
    val = 0.0
    
    
    if func_name == 'power_sum':
        val = 0.0
        if dim == 4:
            theo_pos = [1.0, 2.0, 2.0, 3.0] 
        elif dim == 2:
            theo_pos = [1.0, 2.0]
        elif dim == 3:
            theo_pos = [1.0, 2.0, 3.0]
        else:
            theo_pos = [i+1 for i in range(dim)] 
    
    
    elif func_name == 'eggholder':
        val = -959.6407
        theo_pos = [512.0, 404.2319] 
        
    
    elif func_name == 'styblinski_tang':
        val = -39.16617 * dim
        theo_pos = [-2.903534] * dim
        
    
    elif func_name == 'michalewicz':
        if dim == 2:
            val = -1.8013
            theo_pos = [2.2029, 1.5708]
        elif dim == 5:
            val = -4.6877
            theo_pos = [2.2029, 1.5708, 1.2850, 1.9231, 1.7205]
        else:
             val = -0.966 * dim 
             theo_pos = [0.0] * dim 
             

    elif func_name == 'rosenbrock':
        val = 0.0
        theo_pos = [1.0] * dim
        
    
    elif func_name == 'shekel':
        val = -10.5364
        theo_pos = [4.0] * 4
        
    
    elif func_name == 'easom':
        val = -1.0
        theo_pos = [3.14159, 3.14159]
        
    
    elif func_name == 'powell':
        val = 0.0
        theo_pos = [0.0] * dim


    return val, theo_pos
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--dim', type=int, default=2)
    # UPDATED DEFAULTS: Population=100, Iterations=500
    parser.add_argument('--particle-count', type=int, default=100) 
    parser.add_argument('--max-iterations', type=int, default=500) 
    args = parser.parse_args()
    target_func = args.function.lower()
    
    # --- Bounds ---
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
        'perm': (-4, 4), 
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
        'perm_0': (-2, 2), 
        'rotated_hyper_ellipsoid': (-65.536, 65.536), 
        'sphere': (-5.12, 5.12),
        'sum_of_different_powers': (-1, 1), 
        'mccormick': ([-1.5, -3], [4, 4]),
        'trid': (-4, 4), #[-dkare, dkare] arlığında da denenebilir
        'power_sum': (0, 2), #[0, d] arlığında da denenebilir
        'dejong_n5': (-65.536, 65.536),
        'sum_squares': (-10, 10), #[-5.12, 5.12] arlığında da denenebilir
        'bukin_n6': ([-15, -3], [-5, 3])
    }
    
    fixed_dims = {
        'colville':4,'hartmann_3d':3,'hartmann_4d':4,'hartmann_6d':6,
        'shekel':4,'eggholder':2,'goldstein_price':2,
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

    if target_func not in bounds: 
        sys.exit(f"Error: {target_func} not found in bounds dictionary.")
    
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
    print(f"ALGORİTMA: PSO | FONKSİYON: {target_func.upper()} | DIM: {current_dim} {dim_info}")
    print(f"PARÇACIK: {args.particle_count} | İTERASYON: {args.max_iterations}")
    print(f"HEDEF: {theo_fit}")
    print("="*145)
    print(f"{'No':<4} {'Best Fitness':<18} {'Time(s)':<10} {'Parametre 1':<15} {'Parametre 2':<15} {'Parametre 3':<15} {'Parametre 4':<15} {'Parametre 5':<15} {'Parametre 6':<15}")
    print("-" * 145)

    fits, times = [], []

    for i in range(1, args.trials + 1):
        opt = PSOOptimizer(lb, ub, args.particle_count, current_dim)
        st = time.time()
        res = opt.optimize(obj_func, args.max_iterations)
        et = time.time()

        run_time = et - st
        fit, pos = res['best_fitness'], res['best_position']

        fits.append(fit)
        times.append(run_time)

        params = []
        for j in range(6):
            if j < len(pos):
                params.append(f"{pos[j]:<15.4f}")
            else:
                params.append(f"{'-':<15}")

        print(f"{i:<4} {fit:<18.8f} {run_time:<10.4f} {params[0]}{params[1]}{params[2]}{params[3]}{params[4]}{params[5]}")

    print("="*145)
    print(f"ORTALAMA FITNESS: {np.mean(fits):.10f}")
    print(f"EN İYİ FITNESS  : {np.min(fits):.10f}")
    print(f"ORTALAMA SÜRE   : {np.mean(times):.4f} sn")
    print("="*145)