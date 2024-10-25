import numpy as np
from deap import base, creator, tools, algorithms
import random

# Six-Hump Camel Fonksiyonu
def six_hump_camel(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2,

# Genetik algoritma parametreleri
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize edilecek
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -3, 3)  # X1 ve X2'nin tanım aralığı
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # Her bireyin 2 boyutlu olması gerekiyor
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", six_hump_camel)

# Genetik algoritma işlemleri
def main():
    random.seed(42)  # Rastgelelik için sabit bir başlangıç
    
    pop = toolbox.population(n=300)  # 300 bireylik bir popülasyon oluştur
    hof = tools.HallOfFame(1)  # En iyi bireyi kaydetmek için Hall of Fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Genetik algoritma ile evrim
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                       stats=stats, halloffame=hof, verbose=True)
    
    # Sonuçlar
    print("En iyi birey: ", hof[0])
    print("En iyi bireyin fonksiyon değeri: ", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()
