# Description: This file contains the implementation of 47 optimization functions.
import numpy as np
class OptimizationFunctions:
    def sphere_function(self, x):
        return np.sum(x**2)

    def ackley_function(self, x):
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.e

    def three_hump_camel_function(self, x):
        x1, x2 = x[0], x[1]
        return 2 * x1**2 - 1.05 * x1**4 + (x1**6 / 6) + x1 * x2 + x2**2

    def six_hump_camel_function(self, x):
        x1, x2 = x[0], x[1]
        return (4 - 2.1 * x1**2 + (x1**4 / 3)) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

    def dixon_price_function(self, x):
        d = len(x)
        return (x[0] - 1)**2 + np.sum([(i+1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, d)])

    def rosenbrock_function(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

    def beale_function(self, x):
        x1, x2 = x[0], x[1]
        return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

    def branin_function(self, x):
        x1, x2 = x[0], x[1]
        a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
        r, s, t = 6, 10, 1/(8*np.pi)
        return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s

    def colville_function(self, x):
        x1, x2, x3, x4 = x
        return (100*(x2 - x1**2)**2 + (1 - x1)**2 + 
                90*(x4 - x3**2)**2 + (1 - x3)**2 + 
                10.1*((x2 - 1)**2 + (x4 - 1)**2) + 
                19.8*(x2 - 1)*(x4 - 1))

    def forrester_function(self, x):
        return ((6*x[0] - 2)**2) * np.sin(12*x[0] - 4)

    def goldstein_price_function(self, x):
        x1, x2 = x[0], x[1]
        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return term1 * term2

    def hartmann_3d_function(self, x):
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array([
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0]
        ])
        P = 1e-4 * np.array([
            [3689, 1170, 2673],
            [4699, 4333, 1091],
            [1091, 8732, 4303],
            [381, 5743, 3125]
        ])
        return -sum(alpha[j] * np.exp(-sum(A[j,i] * (x[i] - P[j,i])**2 for i in range(3))) for j in range(4))

    def hartmann_4d_function(self, x):
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array([
            [10.0, 3.0, 17.0, 3.5],
            [0.05, 10.0, 17.0, 0.1],
            [3.0, 3.5, 1.7, 10.0],
            [17.0, 8.0, 0.05, 10.0]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124],
            [2329, 4135, 8307, 3736],
            [2348, 1451, 3522, 2883],
            [4047, 8828, 8732, 5743]
        ])
        return -sum(alpha[j] * np.exp(-sum(A[j,i] * (x[i] - P[j,i])**2 for i in range(4))) for j in range(4))

    def hartmann_6d_function(self, x):
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array([
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])
        return -sum(alpha[j] * np.exp(-sum(A[j,i] * (x[i] - P[j,i])**2 for i in range(6))) for j in range(4))

    def perm_function(self, x, beta=0.5):
        d = len(x)
        return sum(
            sum((j**k + beta) * ((x[j-1]/j)**k - 1)**2 
                for j in range(1, d+1))
            for k in range(1, d+1)
        )

    def powell_function(self, x):
        n = len(x)
        result = 0
        for i in range(0, n, 4):
            if i+3 >= n:
                break  # Avoid index out of range
            result += (x[i] + 10*x[i+1])**2
            result += 5 * (x[i+2] - x[i+3])**2
            result += (x[i+1] - 2*x[i+2])**4
            result += 10 * (x[i] - x[i+3])**4
        return result

    def shekel_function(self, x):
        """Shekel Function"""
        m, beta = 10, 0.1
        A = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6]
        ])
        return -sum(1 / (sum((x[j] - A[i,j])**2 for j in range(4)) + beta) for i in range(m))

    def styblinski_tang_function(self, x):
        """Styblinski-Tang Function"""
        return 0.5 * sum(x_i**4 - 16*x_i**2 + 5*x_i for x_i in x)

    # Additional functions to reach 47
    def michalewicz_function(self, x, m=10):
        """Michalewicz Function"""
        return -sum(np.sin(x[i]) * (np.sin((i+1)*x[i]**2 / np.pi))**(2*m) for i in range(len(x)))

    def easom_function(self, x):
        """Easom Function"""
        x1, x2 = x
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

    def booth_function(self, x):
        """Booth Function"""
        x1, x2 = x
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

    def matyas_function(self, x):
        """Matyas Function"""
        x1, x2 = x
        return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

    def zakharov_function(self, x):
        """Zakharov Function"""
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * (np.arange(1, len(x)+1)) * x)
        return sum1 + sum2**2 + sum2**4

    def bukin_n5_function(self, x):
        """Bukin N.5 Function"""
        x1, x2 = x[0], x[1]
        term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * abs(x1 + 10)
        return term1 + term2

    def schwefel_226_function(self, x):
        """Schwefel 2.26 Function"""
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    def sinc_function(self, x):
        """Sinc Function"""
        return np.prod([np.sinc(x_i) for x_i in x])

    def ackley2_function(self, x):
        """Modified Ackley Function"""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e

    def sum_squares_function(self, x):
        """Sum Squares Function"""
        return np.sum(x**2)

    def step_function(self, x):
        """Step Function"""
        return np.sum(np.floor(x + 0.5)**2)

    def alpine_function(self, x):
        """Alpine Function"""
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

    def bukin_n4_function(self, x):
        """Bukin N.4 Function"""
        x1, x2 = x[0], x[1]
        return np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2) * np.exp((1 - np.sin(x1))**2)

    def cosine_mixture_function(self, x):
        """Cosine Mixture Function"""
        return np.sum(np.cos(x) + 0.5 * x)

    # Update velocity with inertia weight
    def velocity_update(self):
        r1 = np.random.rand(self.particle_count, self.dimension)
        r2 = np.random.rand(self.particle_count, self.dimension)
        
        cognitive = self.c1 * r1 * (self.pbest_positions - self.particles)
        social = self.c2 * r2 * (self.gbest_position - self.particles)
        
        inertia_weight = 0.5  # Can be tuned or made dynamic
        self.velocities = inertia_weight * self.velocities + cognitive + social

    def position_update(self):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

    def optimize(self, func_name, max_iterations=1000, tolerance=1e-6):
        functions = {
            'sphere': self.sphere_function,
            'ackley': self.ackley_function,
            'three_hump_camel': self.three_hump_camel_function,
            'six_hump_camel': self.six_hump_camel_function,
            'dixon_price': self.dixon_price_function,
            'rosenbrock': self.rosenbrock_function,
            'beale': self.beale_function,
            'branin': self.branin_function,
            'colville': self.colville_function,
            'forrester': self.forrester_function,
            'goldstein_price': self.goldstein_price_function,
            'hartmann_3d': self.hartmann_3d_function,
            'hartmann_4d': self.hartmann_4d_function,
            'hartmann_6d': self.hartmann_6d_function,
            'perm': self.perm_function,
            'powell': self.powell_function,
            'shekel': self.shekel_function,
            'styblinski_tang': self.styblinski_tang_function,
            'bukin': self.bukin_n6_function,
            'cross_in_tray': self.cross_in_tray_function,
            'drop_wave': self.drop_wave_function,
            'eggholder': self.eggholder_function,
            'gramacy_lee': self.gramacy_lee_function,
            'griewank': self.griewank_function,
            'holder_table': self.holder_table_function,
            'langermann': self.langermann_function,
            'levy': self.levy_function,
            'levy_n13': self.levy_n13_function,
            'rastrigin': self.rastrigin_function,
            'schaffer_n2': self.schaffer_n2_function,
            'schaffer_n4': self.schaffer_n4_function,
            'schwefel': self.schwefel_function,
            'shubert': self.shubert_function,
            'michalewicz': self.michalewicz_function,
            'easom': self.easom_function,
            'booth': self.booth_function,
            'matyas': self.matyas_function,
            'zakharov': self.zakharov_function,
            'bukin_n5': self.bukin_n5_function,
            'schwefel_226': self.schwefel_226_function,
            'sinc': self.sinc_function,
            'ackley2': self.ackley2_function,
            'sum_squares': self.sum_squares_function,
            'step': self.step_function,
            'alpine': self.alpine_function,
            'bukin_n4': self.bukin_n4_function,
            'cosine_mixture': self.cosine_mixture_function
        }

        if func_name not in functions:
            raise ValueError(f"Function '{func_name}' is not implemented.")

        func = functions[func_name]

        for iteration in range(max_iterations):
            fitness = np.apply_along_axis(func, 1, self.particles)

            for i in range(self.particle_count):
                if fitness[i] < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness[i]
                    self.pbest_positions[i] = self.particles[i]

            min_fitness_idx = np.argmin(self.pbest_scores)
            if self.pbest_scores[min_fitness_idx] < self.gbest_score:
                self.gbest_score = self.pbest_scores[min_fitness_idx]
                self.gbest_position = self.pbest_positions[min_fitness_idx]

            if self.gbest_score <= tolerance:
                break

            self.velocity_update()
            self.position_update()

        return {
            'best_position': self.gbest_position,
            'best_fitness': self.gbest_score,
            'iterations': iteration + 1
        }

    def bukin_n6_function(self, x):
        """Bukin N.6 Function"""
        x1, x2 = x[0], x[1]
        term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * abs(x1 + 10)
        return term1 + term2

    def cross_in_tray_function(self, x):
        """Cross-in-Tray Function"""
        x1, x2 = x[0], x[1]
        return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2)/np.pi))) + 1)**0.1

    def drop_wave_function(self, x):
        """Drop-Wave Function"""
        x1, x2 = x[0], x[1]
        frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
        frac2 = 0.5 * (x1**2 + x2**2) + 2
        return -frac1 / frac2

    def eggholder_function(self, x):
        """Eggholder Function"""
        x1, x2 = x[0], x[1]
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        return term1 + term2

    def gramacy_lee_function(self, x):
        """Gramacy & Lee Function"""
        return np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1)**4

    def griewank_function(self, x):
        """Griewank Function"""
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq / 4000 - prod_cos

    def holder_table_function(self, x):
        """Holder Table Function"""
        x1, x2 = x[0], x[1]
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)))

    def langermann_function(self, x):
        """Langermann Function"""
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        c = np.array([1, 2, 5, 2, 3])
        m = 5
        d = 2
        sum_val = 0
        for i in range(m):
            term1 = c[i] * np.exp(-1/np.pi * np.sum((x - A[i])**2))
            term2 = np.cos(np.pi * np.sum((x - A[i])**2))
            sum_val += term1 * term2
        return sum_val

    def levy_function(self, x):
        """Levy Function"""
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return term1 + term2 + term3

    def levy_n13_function(self, x):
        """Levy N.13 Function"""
        x1, x2 = x[0], x[1]
        return (np.sin(3 * np.pi * x1)**2 + (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2) +
                (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2))

    def rastrigin_function(self, x):
        """Rastrigin Function"""
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def schaffer_n2_function(self, x):
        """Schaffer N.2 Function"""
        x1, x2 = x[0], x[1]
        num = np.sin(x1**2 - x2**2)**2 - 0.5
        den = (1 + 0.001*(x1**2 + x2**2))**2
        return 0.5 + num / den

    def schaffer_n4_function(self, x):
        """Schaffer N.4 Function"""
        x1, x2 = x[0], x[1]
        num = np.cos(np.sin(np.abs(x1**2 - x2**2)))**2 - 0.5
        den = (1 + 0.001*(x1**2 + x2**2))**2
        return 0.5 + num / den

    def schwefel_function(self, x):
        """Schwefel Function"""
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def shubert_function(self, x):
        """Shubert Function"""
        x1, x2 = x[0], x[1]
        sum1 = sum(i * np.cos((i+1)*x1 + i) for i in range(1, 6))
        sum2 = sum(i * np.cos((i+1)*x2 + i) for i in range(1, 6))
        return sum1 * sum2

    def michalewicz_function(self, x, m=10):
        """Michalewicz Function"""
        return -sum(np.sin(x[i]) * (np.sin((i+1)*x[i]**2 / np.pi))**(2*m) for i in range(len(x)))

    def easom_function(self, x):
        """Easom Function"""
        x1, x2 = x
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

    def booth_function(self, x):
        """Booth Function"""
        x1, x2 = x
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

    def matyas_function(self, x):
        """Matyas Function"""
        x1, x2 = x
        return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

    def zakharov_function(self, x):
        """Zakharov Function"""
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * (np.arange(1, len(x)+1)) * x)
        return sum1 + sum2**2 + sum2**4

    def bukin_n5_function(self, x):
        """Bukin N.5 Function"""
        x1, x2 = x[0], x[1]
        term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * abs(x1 + 10)
        return term1 + term2

    def schwefel_226_function(self, x):
        """Schwefel 2.26 Function"""
        return np.sum(np.abs(x)) + np.prod(np.abs(x))

    def sinc_function(self, x):
        """Sinc Function"""
        return np.prod([np.sinc(x_i) for x_i in x])

    def ackley2_function(self, x):
        """Modified Ackley Function"""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e

    def sum_squares_function(self, x):
        """Sum Squares Function"""
        return np.sum(x**2)

    def step_function(self, x):
        """Step Function"""
        return np.sum(np.floor(x + 0.5)**2)

    def alpine_function(self, x):
        """Alpine Function"""
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

    def bukin_n4_function(self, x):
        """Bukin N.4 Function"""
        x1, x2 = x[0], x[1]
        return np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2) * np.exp((1 - np.sin(x1))**2)

    def cosine_mixture_function(self, x):
        """Cosine Mixture Function"""
        return np.sum(np.cos(x) + 0.5 * x)
    