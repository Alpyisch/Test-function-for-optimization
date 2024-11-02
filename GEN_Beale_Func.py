import time

# Başlangıç zamanı
start_time = time.time()

# Kodunuzu buraya yazın
# Örneğin:
for i in range(1000000):
    pass

# Bitiş zamanı
end_time = time.time()

# Çalışma süresi
print("Çalışma süresi:", end_time - start_time, "saniye")

import numpy as np

class Beale_GA:
    def __init__(self, lower_bound, upper_bound, population_size, variable_count, crossover_rate, mutation_rate):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.variable_count = variable_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.population_size, self.variable_count))
        self.fitness = np.zeros(self.population_size)

    # Beale fonksiyonu
    def beale_function(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        fitness = term1 + term2 + term3
        return fitness

    # Fitness hesaplama
    def evaluate_fitness(self):
        self.fitness = self.beale_function(self.population)

    # Seçim fonksiyonu (Turnuva Seçimi)
    def selection(self):
        selected_parents = []
        for _ in range(self.population_size):
            i, j = np.random.randint(0, self.population_size, 2)
            if self.fitness[i] < self.fitness[j]:  # Düşük fitness daha iyi
                selected_parents.append(self.population[i])
            else:
                selected_parents.append(self.population[j])
        return np.array(selected_parents)

    # Çaprazlama fonksiyonu (Tek Nokta Çaprazlama)
    def crossover(self, parents):
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            if np.random.rand() < self.crossover_rate:
                point = np.random.randint(1, self.variable_count)
                child1 = np.concatenate((parent1[:point], parent2[point:]))
                child2 = np.concatenate((parent2[:point], parent1[point:]))
            else:
                child1, child2 = parent1, parent2
            offspring.extend([child1, child2])
        return np.array(offspring)

    # Mutasyon fonksiyonu
    def mutate(self, offspring):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_idx = np.random.randint(0, self.variable_count)
                offspring[i, mutation_idx] += np.random.uniform(-1, 1)
                offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)  # Sınır kontrolü
        return offspring

    # Eğitim fonksiyonu
    def train(self, max_generations=1000):
        self.evaluate_fitness()
        best_solution = self.population[self.fitness.argmin()]
        best_fitness = self.fitness.min()
        
        generation = 0
        while generation < max_generations and best_fitness >= 1e-6:
            generation += 1
            parents = self.selection()
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)

            self.population = offspring
            self.evaluate_fitness()
            
            if self.fitness.min() < best_fitness:
                best_solution = self.population[self.fitness.argmin()]
                best_fitness = self.fitness.min()

        return best_solution, best_fitness, generation

# Beale fonksiyonu için Genetik Algoritmayı çalıştırma
ga_object = Beale_GA(-5, 5, 50, 2, 0.8, 0.1)  # 2 boyutlu Beale fonksiyonu
best_solution, best_fitness, generations = ga_object.train()
print("Best x1 value: ", best_solution[0])
print("Best x2 value: ", best_solution[1])
print("Found fitness value: ", best_fitness)
print("This result was found after: ", generations, " generations!")
