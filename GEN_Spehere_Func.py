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

# Parametreler
pop_size = 50       # Popülasyon büyüklüğü
num_generations = 200  # Nesil sayısı
num_genes = 5        # Her bireydeki gen sayısı (problem boyutu)
mutation_rate = 0.1  # Mutasyon oranı
selection_rate = 0.5 # Seçim oranı

# Sphere fonksiyonu
def sphere_function(x):
    return np.sum(x ** 2)

# Popülasyon başlatma
def initialize_population():
    return np.random.uniform(-5.12, 5.12, (pop_size, num_genes))

# Fitness hesaplama (küçük değerler daha iyidir)
def calculate_fitness(population):
    return np.array([sphere_function(individual) for individual in population])

# Seçim (turnuva seçimi)
def selection(population, fitness):
    selected_indices = np.argsort(fitness)[:int(pop_size * selection_rate)]
    return population[selected_indices]

# Çaprazlama (iki noktalı çaprazlama)
def crossover(parent1, parent2):
    crossover_point1, crossover_point2 = np.sort(np.random.choice(num_genes, 2, replace=False))
    child1 = np.concatenate((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
    child2 = np.concatenate((parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
    return child1, child2

# Mutasyon
def mutation(individual):
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(num_genes)
        individual[mutation_index] = np.random.uniform(-5.12, 5.12)
    return individual

# Genetik algoritma
def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        fitness = calculate_fitness(population)
        selected_population = selection(population, fitness)

        # Yeni popülasyon oluşturma
        next_population = []
        while len(next_population) < pop_size:
            parent1, parent2 = selected_population[np.random.choice(len(selected_population), 2, replace=False)]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutation(child1))
            next_population.append(mutation(child2))
        population = np.array(next_population[:pop_size])

        # En iyi bireyi ve fitness değerini yazdır
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        print(f"Nesil {generation+1} - En iyi fitness: {best_fitness} - En iyi birey: {best_individual}")

    # Sonuç
    best_fitness = np.min(fitness)
    best_individual = population[np.argmin(fitness)]
    return best_individual, best_fitness

# Genetik algoritmayı çalıştır
best_solution, best_fitness = genetic_algorithm()
print(f"En iyi çözüm: {best_solution} - En iyi fitness: {best_fitness}")
