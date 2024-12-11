import time

# Baslangiç zamani
start_time = time.time()

import numpy as np

class Matyas_PSO:
    def __init__(self, lower_bound, upper_bound, particle_count, variable_count, c1, c2):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.particle_count = particle_count
        self.variable_count = variable_count
        self.c1 = c1
        self.c2 = c2
        self.particles = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.particle_count, self.variable_count))
        self.velocities = np.random.uniform(-1, 1, (self.particle_count, self.variable_count))
        self.fitness = np.zeros(self.particle_count)
        self.pbest = np.zeros((self.particle_count, self.variable_count + 1))
        self.gbest = np.zeros(self.variable_count + 1)

    # Matyas fonksiyonu
    def matyas_function(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        fitness = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
        return fitness

    # Hiz guncelleme fonksiyonu
    def velocity_update(self, pbest, gbest):
        rand1 = np.random.rand(self.particle_count, self.variable_count)
        rand2 = np.random.rand(self.particle_count, self.variable_count)
        cognitive = self.c1 * rand1 * (pbest[:, :-1] - self.particles)
        social = self.c2 * rand2 * (gbest[:-1] - self.particles)
        self.velocities = 0.5 * self.velocities + cognitive + social  # Inertia term (0.5)

    # Konum guncelleme fonksiyonu
    def position_update(self):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)  # Sinir kontrolu

    # Egitim fonksiyonu
    def train(self, max_iterations=1000):
        self.fitness = self.matyas_function(self.particles)  # Ilk uygunluk degerleri
        
        self.pbest[:, :-1] = self.particles  # Ilk durumda pbest kendisi
        self.pbest[:, -1] = self.fitness  # Ilk durumda en iyi uygunluk kendi
        self.gbest = self.pbest[self.pbest[:, -1].argmin(), :]  # En düşük uygunluk degerini al
        
        self.iteration = 0
        while self.iteration < max_iterations and self.gbest[-1] >= 1e-6:  # Belirli bir uygunluk degerine ulasana kadar devam et
            self.iteration += 1
            self.fitness = self.matyas_function(self.particles)  # Her iterasyonda uygunluk hesapla
            
            # pbest guncelleme
            for i in range(self.fitness.size):
                if self.pbest[i, -1] > self.fitness[i]:  # Daha iyi uygunluk varsa pbest'i guncelle
                    self.pbest[i, :-1] = self.particles[i, :]
                    self.pbest[i, -1] = self.fitness[i]
            
            # gbest guncelleme
            if self.gbest[-1] > self.pbest[self.pbest[:, -1].argmin(), -1]:
                self.gbest = self.pbest[self.pbest[:, -1].argmin(), :]
            
            # Hiz ve konum guncelleme
            self.velocity_update(self.pbest, self.gbest)
            self.position_update()
        
        return self.gbest

# Matyas fonksiyonu icin PSO'yu calistirma
pso_object = Matyas_PSO(-10, 10, 50, 2, 2, 2)  # 2 boyutlu Matyas fonksiyonu
best_values = pso_object.train()

print("Best x1 value: ", best_values[0])
print("Best x2 value: ", best_values[1])
print("Found fitness value: ", best_values[2])
print("This result was found after: ", pso_object.iteration, " iterations!")

# Bitis zamani
end_time = time.time()

# Calisma suresi
print("Calisma suresi:", end_time - start_time, "saniye")