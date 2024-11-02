import numpy as np

def beale_f(x):
    """Beale function."""
    c1 = 1.5
    c2 = 2.25
    c3 = 2.625

    fx1 = c1 - x[0] * (1.0 - x[1])
    fx2 = c2 - x[0] * (1.0 - x[1] ** 2)
    fx3 = c3 - x[0] * (1.0 - x[1] ** 3)

    value = fx1 ** 2 + fx2 ** 2 + fx3 ** 2
    return value

class ArtificialBeeColony:
    def __init__(self, num_bees, max_iterations, bounds):
        self.num_bees = num_bees
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self):
        # Initialize the positions of the bees
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_bees, len(self.bounds[0])))
        fitness = np.array([beale_f(pos) for pos in positions])

        for iteration in range(self.max_iterations):
            for i in range(self.num_bees):
                # Select a random bee (food source) to share information with
                partner_index = np.random.randint(0, self.num_bees)
                partner = positions[partner_index]

                # Generate a new solution
                new_solution = positions[i] + np.random.uniform(-1, 1) * (positions[i] - partner)
                new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])  # Ensure within bounds

                # Evaluate the new solution
                new_fitness = beale_f(new_solution)

                # Apply the greedy selection process
                if new_fitness < fitness[i]:
                    positions[i] = new_solution
                    fitness[i] = new_fitness

            # Update the best solution found
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.best_fitness:
                self.best_fitness = fitness[best_index]
                self.best_solution = positions[best_index]

        return self.best_solution, self.best_fitness

def beale_test():
    """Test the Beale function using Artificial Bee Colony optimization."""
    print('BEALE_TEST')
    num_bees = 50
    max_iterations = 100
    bounds = np.array([[-5, -5], [5, 5]])  # Search bounds for x[0] and x[1]

    abc_optimizer = ArtificialBeeColony(num_bees, max_iterations, bounds)
    best_solution, best_fitness = abc_optimizer.optimize()

    print('Computed minimizer:', best_solution)
    print('Function value:', best_fitness)

if __name__ == '__main__':
    beale_test()
