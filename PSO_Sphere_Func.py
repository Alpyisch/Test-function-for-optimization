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

class Sphere_PSO:
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

    # Sphere fonksiyonu
    def sphere_function(self, x):
        return np.sum(x**2, axis=1)

    # Hız güncelleme fonksiyonu
    def velocity_update(self, pbest, gbest):
        rand1 = np.random.rand(self.particle_count, self.variable_count)
        rand2 = np.random.rand(self.particle_count, self.variable_count)
        cognitive = self.c1 * rand1 * (pbest[:, :-1] - self.particles)
        social = self.c2 * rand2 * (gbest[:-1] - self.particles)
        self.velocities = 0.5 * self.velocities + cognitive + social  # Inertia term (0.5)

    # Konum güncelleme fonksiyonu
    def position_update(self):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)  # Sınır kontrolü

    # Eğitim fonksiyonu
    def train(self):
        self.fitness = self.sphere_function(self.particles)  # İlk uygunluk değerleri
        
        self.pbest[:, :-1] = self.particles  # İlk durumda pbest'ler kendisi
        self.pbest[:, -1] = self.fitness  # İlk durumda en iyi uygunluk da kendisininki
        self.gbest = self.pbest[self.pbest[:, -1].argmin(), :]  # En düşük uygunluk değerini al
        
        self.iteration = 0
        while self.gbest[-1] >= 1e-6:  # Belirli bir uygunluk değerine ulaşılana kadar devam et
            self.iteration += 1
            self.fitness = self.sphere_function(self.particles)  # Her iterasyonda uygunluk hesapla
            
            # pbest güncelleme
            for i in range(self.fitness.size):
                if self.pbest[i, -1] > self.fitness[i]:  # Daha iyi uygunluk varsa pbest'i güncelle
                    self.pbest[i, :-1] = self.particles[i, :]
                    self.pbest[i, -1] = self.fitness[i]
            
            # gbest güncelleme
            if self.gbest[-1] > self.pbest[self.pbest[:, -1].argmin(), -1]:
                self.gbest = self.pbest[self.pbest[:, -1].argmin(), :]
            
            # Hız ve konum güncelleme
            self.velocity_update(self.pbest, self.gbest)
            self.position_update()
        
        return self.gbest

# Sphere fonksiyonu için PSO'yu çalıştırma
pso_object = Sphere_PSO(-5.12, 5.12, 50, 3, 2, 2)  # 3 boyutlu Sphere fonksiyonu
best_values = pso_object.train()

print("Best x1 value: ", best_values[0])
print("Best x2 value: ", best_values[1])
print("Best x3 value: ", best_values[2])
print("Found fitness value: ", best_values[3])
print("This result was found after: ", pso_object.iteration, " iterations!")
