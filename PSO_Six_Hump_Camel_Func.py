import numpy as np

class Six_Hump_Camel:
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

    # Six Hump Camel fonksiyonu
    def six_hump_camel_back(self, x1, x2):
        fitness = (4-2.1*x1**2+x1**4/3)*x1**2+x1*x2+(-4+4*x2**2)*x2**2
        return fitness

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
        self.fitness = self.six_hump_camel_back(self.particles[:, 0], self.particles[:, 1])
        
        self.pbest[:, :-1] = self.particles  # İlk durumda pbest kendisi
        self.pbest[:, -1] = self.fitness  # İlk durumda en iyi uygunluk kendi
        self.gbest = self.pbest[self.pbest[:, -1].argmin(), :]  # En düşük uygunluk değerini al
        
        self.iteration = 0
        while self.gbest[-1] >= 1e-6:  # Belirli bir uygunluk değerine ulaşılana kadar devam et
            self.iteration += 1
            self.fitness = self.six_hump_camel_back(self.particles[:, 0], self.particles[:, 1])  # Her iterasyonda uygunluk hesapla
            
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

# Six Hump Camel fonksiyonu için PSO'yu çalıştırma
pso_object = Six_Hump_Camel(-2, 2, 1550, 2, 2, 2)  # 2 boyutlu Six Hump Camel fonksiyonu
best_values = pso_object.train()

print("Best x1 value: ", best_values[0])
print("Best x2 value: ", best_values[1])
print("Found fitness value: ", best_values[2])
print("This result was found after: ", pso_object.iteration, " iterations!")
