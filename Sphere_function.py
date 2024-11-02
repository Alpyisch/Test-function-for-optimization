import numpy as np

# Define the Sphere function
def sphere(x):
    return np.sum(x**2)

# Define the Artificial Bee Colony (ABC) algorithm
class ABC:
    def __init__(self, func, bounds, num_bees=30, max_iter=100, limit=100):
        self.func = func
        self.bounds = bounds
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.limit = limit
        
        self.dim = len(bounds)
        self.population = np.random.rand(self.num_bees, self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.fitness = np.apply_along_axis(self.func, 1, self.population)
        self.trial = np.zeros(self.num_bees)
    
    def optimize(self):
        best_solution = self.population[np.argmin(self.fitness)]
        best_fitness = np.min(self.fitness)
        
        for iteration in range(self.max_iter):
            # Employed bee phase
            for i in range(self.num_bees):
                k = np.random.randint(self.num_bees)
                while k == i:
                    k = np.random.randint(self.num_bees)
                
                phi = np.random.uniform(-1, 1, self.dim)
                candidate_solution = self.population[i] + phi * (self.population[i] - self.population[k])
                candidate_solution = np.clip(candidate_solution, self.bounds[:, 0], self.bounds[:, 1])
                
                candidate_fitness = self.func(candidate_solution)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate_solution
                    self.fitness[i] = candidate_fitness
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
            
            # Onlooker bee phase
            # Shift fitness to ensure all values are positive
            shifted_fitness = self.fitness - np.min(self.fitness) + 1e-8  # Shift to make all fitness values positive
            inverse_fitness = 1 / shifted_fitness  # Inverse fitness for maximization-like behavior
            fitness_prob = inverse_fitness / np.sum(inverse_fitness)  # Normalize to get probabilities

            for i in range(self.num_bees):
                j = np.random.choice(range(self.num_bees), p=fitness_prob)
                k = np.random.randint(self.num_bees)
                while k == j:
                    k = np.random.randint(self.num_bees)

                phi = np.random.uniform(-1, 1, self.dim)
                candidate_solution = self.population[j] + phi * (self.population[j] - self.population[k])
                candidate_solution = np.clip(candidate_solution, self.bounds[:, 0], self.bounds[:, 1])

                candidate_fitness = self.func(candidate_solution)
                if candidate_fitness < self.fitness[j]:
                    self.population[j] = candidate_solution
                    self.fitness[j] = candidate_fitness
                    self.trial[j] = 0
                else:
                    self.trial[j] += 1
            
            # Scout bee phase
            for i in range(self.num_bees):
                if self.trial[i] > self.limit:
                    self.population[i] = np.random.rand(self.dim) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
                    self.fitness[i] = self.func(self.population[i])
                    self.trial[i] = 0
            
            # Track the best solution
            current_best_fitness = np.min(self.fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = self.population[np.argmin(self.fitness)]
            
            print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")
        
        return best_solution, best_fitness

# Define the problem bounds for the Sphere function (10 dimensions)
bounds = np.array([[-5.12, 5.12]] *2)  # 10-dimensional problem

# Instantiate and run the ABC algorithm
abc_optimizer = ABC(sphere, bounds, num_bees=50, max_iter=100, limit=50)
best_solution, best_fitness = abc_optimizer.optimize()

print(f"\nBest Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
