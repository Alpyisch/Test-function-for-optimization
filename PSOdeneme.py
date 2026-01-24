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
            # Sınırları esnek al (Tek sayı veya Liste olabilir)
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
    # Basit bir eşleştirme (Detaylandırılabilir)
    val = 0.0
    if func_name == 'eggholder': return -959.6407, [512, 404.2319]
    elif func_name == 'styblinski_tang': return -39.166 * dim, [-2.9035]*dim
    elif func_name == 'michalewicz': return -1.8013 if dim==2 else -4.687, [2.20, 1.57]
    elif func_name == 'rosenbrock': return 0.0, [1.0]*dim
    return 0.0, [val] * dim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--particle-count', type=int, default=100)
    parser.add_argument('--max-iterations', type=int, default=3000)
    
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

    if target_func not in bounds: 
        sys.exit(f"Error: {target_func} not found in bounds dictionary.")
    
    current_dim = fixed_dims.get(target_func, args.dim)
    bound_data = bounds[target_func]

    # BOUNDS İŞLEME MANTIĞI (LİSTE veya TEK SAYI)
    if isinstance(bound_data[0], (list, tuple)):
        # Eğer ([-15, -3], [-5, 3]) gibi geldiyse
        lb = np.array(bound_data[0])
        ub = np.array(bound_data[1])
    else:
        # Eğer (-5, 5) gibi geldiyse
        lb = np.full(current_dim, bound_data[0])
        ub = np.full(current_dim, bound_data[1])

    opt_funcs = OptimizationFunctions()
    m_name = function_map.get(target_func, f"{target_func}_function")
    if not hasattr(opt_funcs, m_name): m_name = target_func
    
    try:
        obj_func = getattr(opt_funcs, m_name)
    except AttributeError:
        sys.exit(f"Error: Function '{m_name}' not found in OptimizationFunctions class.")

    theo_fit, theo_pos = get_theoretical_values(target_func, current_dim)

    print("\n"+"="*85)
    print(f"ALGORİTMA: PSO | FONKSİYON: {target_func.upper()} | DIM: {current_dim}")
    print(f"PARÇACIK: {args.particle_count} | İTERASYON: {args.max_iterations}")
    print(f"HEDEF: {theo_fit}")
    print("=" * 85)
    print(f"{'No':<4} {'Best Fitness':<20} {'Time(s)':<10} {'Parametre 1':<15} {'Parametre 2':<15} {'Parametre 3':<15} {'Parametre 4':<15}")
    print("-" * 85)

    fits, times = [], []
    for i in range(1, args.trials+1):
        opt = PSOOptimizer(lb, ub, args.particle_count, current_dim)
        st = time.time()
        res = opt.optimize(obj_func, args.max_iterations)
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