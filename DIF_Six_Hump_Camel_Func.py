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

# Six-Hump Camel fonksiyonu
def six_hump_camel(x):
    x1, x2 = x[0], x[1]
    return (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

# Diferansiyel evrim algoritması
def differential_evolution(func, bounds, pop_size=20, max_iter=1000, F=0.5, CR=0.7):
    # Popülasyonu oluştur
    population = np.random.rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    best_solution = None
    best_value = float('inf')

    for iteration in range(max_iter):
        for i in range(pop_size):
            # Rastgele üç farklı birey seç
            indices = [j for j in range(pop_size) if j != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]

            # Farklandırma ve kombinasyon
            mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(len(bounds)) < CR
            trial = np.where(crossover, mutant, population[i])

            # Seçim
            if func(trial) < func(population[i]):
                population[i] = trial

            # En iyi bireyi güncelle
            current_value = func(population[i])
            if current_value < best_value:
                best_value = current_value
                best_solution = population[i]

    return best_solution, best_value

# Fonksiyonun parametre sınırları (x1 ve x2 için)
bounds = np.array([[-3, 3], [-2, 2]])

# Diferansiyel evrim algoritmasını çalıştır
best_solution, best_value = differential_evolution(six_hump_camel, bounds)

print("En iyi çözüm:", best_solution)
print("Fonksiyonun en iyi değeri:", best_value)
