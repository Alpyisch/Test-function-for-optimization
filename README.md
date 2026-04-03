# Metaheuristic Optimization Benchmark Suite

A research benchmark framework implementing and comparing four classical metaheuristic optimization algorithms — **PSO**, **DE**, **ABC**, and **GEN** — across 47 standard test functions from the optimization literature.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Optimization Functions](#optimization-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Output Format](#output-format)
- [Running Tests](#running-tests)
- [Parameter Reference](#parameter-reference)

---

## Overview

This suite is designed for systematic benchmarking of metaheuristic algorithms. Each algorithm is implemented as a standalone class with a unified `optimize()` interface, making it straightforward to swap algorithms and compare results across different test functions and dimensions.

All four algorithms share common design principles:
- Configurable population/colony size
- Flexible lower/upper bounds (scalar or per-dimension arrays)
- Tolerance-based early stopping
- Reproducible trial structure with per-run timing

---

## Project Structure

```
.
├── PSO_Optimizer.py          # Particle Swarm Optimization
├── DE_Optimizer.py           # Differential Evolution
├── ABC_Optimizer.py          # Artificial Bee Colony
├── GEN_Optimizer.py          # Genetic Algorithm
├── Optimization_Functions.py # 47 benchmark test functions
└── test_optimizers.py        # Pytest regression & stability tests
```

---

## Algorithms

### PSO — Particle Swarm Optimization (`PSO_Optimizer.py`)

Simulates a swarm of particles moving through the search space. Each particle tracks its own personal best and is attracted toward the global best.

**Key features:**
- Linear inertia weight decay from `w_max` to `w_min` over iterations
- Per-dimension velocity clamping (`v_max = 0.2 × range`)
- Cognitive and social acceleration coefficients (`c1`, `c2`)

**Default parameters:** `particle_count=100`, `max_iterations=500`, `c1=2.0`, `c2=2.0`, `w_max=0.9`, `w_min=0.4`

```bash
python PSO_Optimizer.py --function ackley --trials 30 --dim 5 --particle-count 100 --max-iterations 500
```

---

### DE — Differential Evolution (`DE_Optimizer.py`)

Evolves a population by generating mutant vectors through linear combinations of existing members, then crossing over with the target vector.

**Key features:**
- Five mutation strategies: `DE/rand/1/bin`, `DE/best/1/bin`, `DE/current-to-best/1/bin`, `DE/best/2/bin`, `DE/rand/2/bin`
- Greedy selection (trial replaces target only if strictly better)
- Configurable scale factor `F` and crossover rate `CR`

**Default parameters:** `population=50`, `max_generations=1000`, `F=0.9`, `CR=0.9`, strategy=`DE/rand/1/bin`

```bash
python DE_Optimizer.py --function rastrigin --trials 30 --dim 10 --population 50 --max-generations 1000 --F 0.8 --CR 0.9
```

---

### ABC — Artificial Bee Colony (`ABC_Optimizer.py`)

Models the foraging behavior of a honey bee colony. The colony is split into employed bees, onlooker bees, and scout bees.

**Key features:**
- Employed bees: local search around each food source
- Onlooker bees: probabilistic selection weighted by fitness quality
- Scout bees: abandon exhausted sources (`trial_count ≥ limit`) and reinitialize randomly
- Custom fitness transform: `1/(1+f)` for `f≥0`, else `1+|f|`

**Default parameters:** `colony_size=50`, `max_cycles=1000`, `limit=50`

```bash
python ABC_Optimizer.py --function schwefel --trials 30 --dim 2 --colony-size 50 --max-cycles 1000
```

---

### GEN — Genetic Algorithm (`GEN_Optimizer.py`)

An evolutionary algorithm using tournament selection, single-point crossover, and Gaussian mutation with elitism.

**Key features:**
- Tournament selection with `k=2`
- Single-point crossover controlled by `crossover_rate`
- Gaussian mutation with `σ = 0.1 × (upper − lower)`
- Elitism: top 2 individuals carried forward each generation

**Default parameters:** `population_size=50`, `max_generations=1000`, `mutation_rate=0.05`, `crossover_rate=0.95`

```bash
python GEN_Optimizer.py --function rosenbrock --trials 30 --dim 5 --population-size 50 --max-generations 1000
```

---

## Optimization Functions

All 47 functions are implemented in `OptimizationFunctions` and accept a NumPy array `x` as input.

### Function Reference

| Function | Dimensions | Search Bounds | Global Minimum |
|---|---|---|---|
| `sphere` | flexible | [−5.12, 5.12] | 0 at **0** |
| `ackley` | flexible | [−32.768, 32.768] | 0 at **0** |
| `rastrigin` | flexible | [−5.12, 5.12] | 0 at **0** |
| `rosenbrock` | flexible | [−5, 10] | 0 at **1** |
| `griewank` | flexible | [−600, 600] | 0 at **0** |
| `schwefel` | flexible | [−500, 500] | 0 at **420.96...** |
| `levy` | flexible | [−10, 10] | 0 at **1** |
| `zakharov` | flexible | [−5, 10] | 0 at **0** |
| `dixon_price` | flexible | [−10, 10] | 0 |
| `styblinski_tang` | flexible | [−5, 5] | −39.166×d at **−2.9035** |
| `michalewicz` | flexible | [0, π] | −1.8013 (d=2) |
| `beale` | 2 | [−4.5, 4.5] | 0 at (3, 0.5) |
| `six_hump_camel` | 2 | [−3,−2]×[3,2] | −1.0316 |
| `branin` | 2 | [−5,0]×[10,15] | 0.397887 |
| `eggholder` | 2 | [−512, 512] | −959.6407 at (512, 404.23) |
| `goldstein_price` | 2 | [−2, 2] | 3 at (0, −1) |
| `cross_in_tray` | 2 | [−10, 10] | −2.0626 |
| `holder_table` | 2 | [−10, 10] | −19.2085 |
| `schaffer_n2` | 2 | [−100, 100] | 0 at **0** |
| `schaffer_n4` | 2 | [−100, 100] | 0.292579 |
| `easom` | 2 | [−100, 100] | −1 at (π, π) |
| `booth` | 2 | [−10, 10] | 0 at (1, 3) |
| `matyas` | 2 | [−10, 10] | 0 at **0** |
| `bukin_n6` | 2 | [−15,−3]×[−5,3] | 0 at (−10, 1) |
| `mccormick` | 2 | [−1.5,−3]×[4,4] | −1.9133 |
| `shubert` | 2 | [−10, 10] | −186.7309 |
| `langermann` | 2 | [0, 10] | negative |
| `drop_wave` | 2 | [−5.12, 5.12] | −1 at **0** |
| `dejong_n5` | 2 | [−65.536, 65.536] | ~0.998 |
| `three_hump_camel` | 2 | [−5, 5] | 0 at **0** |
| `bohachevsky` | 2 | [−100, 100] | 0 at **0** |
| `hartmann_3d` | 3 | [0, 1] | −3.8628 |
| `hartmann_4d` | 4 | [0, 1] | −3.1344 |
| `hartmann_6d` | 6 | [0, 1] | −3.3224 |
| `colville` | 4 | [−10, 10] | 0 at **1** |
| `powell` | 4 | [−4, 5] | 0 at **0** |
| `shekel` | 4 | [0, 10] | −10.5364 |
| `perm` | flexible | [−2, 2] | 0 |
| `perm_0` | flexible | [−2, 2] | 0 |
| `trid` | flexible | [−4, 4] | varies |
| `power_sum` | flexible | [0, 2] | 0 |
| `sum_squares` | flexible | [−10, 10] | 0 at **0** |
| `rotated_hyper_ellipsoid` | flexible | [−65.536, 65.536] | 0 at **0** |
| `sum_of_different_powers` | flexible | [−1, 1] | 0 at **0** |
| `gramacy_lee` | 1 | [0.5, 2.5] | −0.869 |
| `forrester` | 1 | [0, 1] | −6.0217 |

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install numpy pytest
```

No additional dependencies are needed. Clone the repository and run scripts directly.

---

## Usage

### Basic usage

```bash
python <ALGORITHM>_Optimizer.py --function <function_name> [options]
```

### Examples

**Run PSO on Ackley, 30 trials, 5 dimensions:**
```bash
python PSO_Optimizer.py --function ackley --trials 30 --dim 5
```

**Run DE on Schwefel with custom F and CR:**
```bash
python DE_Optimizer.py --function schwefel --trials 20 --dim 2 --F 0.7 --CR 0.85
```

**Run ABC on Eggholder (fixed 2D):**
```bash
python ABC_Optimizer.py --function eggholder --trials 30 --colony-size 100 --max-cycles 2000
```

**Run GEN on Hartmann 6D (fixed dimension, dim flag ignored):**
```bash
python GEN_Optimizer.py --function hartmann_6d --trials 30 --population-size 100
```

### Using the optimizer classes directly

All four optimizers expose the same interface and can be embedded in your own scripts:

```python
import numpy as np
from PSO_Optimizer import PSOOptimizer
from Optimization_Functions import OptimizationFunctions

lb = np.full(5, -32.768)
ub = np.full(5,  32.768)

opt = PSOOptimizer(lb, ub, particle_count=100, dimension=5)

funcs = OptimizationFunctions()
result = opt.optimize(funcs.ackley_function, max_iterations=500)

print(result['best_fitness'])   # scalar fitness value
print(result['best_position'])  # numpy array of length 5
```

The same pattern applies to `DEOptimizer`, `ABCOptimizer`, and `GENOptimizer`, substituting `max_generations` or `max_cycles` as appropriate.

### Comparing all algorithms on one function

```python
import numpy as np
from Optimization_Functions import OptimizationFunctions
from PSO_Optimizer import PSOOptimizer
from DE_Optimizer import DEOptimizer
from ABC_Optimizer import ABCOptimizer
from GEN_Optimizer import GENOptimizer

funcs = OptimizationFunctions()
dim = 2
lb = np.full(dim, -5.12)
ub = np.full(dim,  5.12)

configs = {
    'PSO': (PSOOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_iterations=1000)),
    'DE':  (DEOptimizer(lb, ub, 50, dim),  lambda opt: opt.optimize(funcs.rastrigin_function, max_generations=1000)),
    'ABC': (ABCOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_cycles=1000)),
    'GEN': (GENOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_generations=1000)),
}

for name, (opt, run) in configs.items():
    result = run(opt)
    print(f"{name}: {result['best_fitness']:.6f}")
```

---

## Output Format

Each run prints a table to stdout:

```
=================================================================================
ALGORİTMA: PSO | FONKSİYON: ACKLEY | DIM: 5 (ESNEK)
PARÇACIK: 100 | İTERASYON: 500
HEDEF: 0.0
=================================================================================
No   Best Fitness       Time(s)    Parametre 1    Parametre 2    ...
---------------------------------------------------------------------------------
1    0.00000142         0.3821     0.0001         -0.0000        ...
2    0.00000389         0.3754     0.0000          0.0001        ...
...
=================================================================================
ORTALAMA FITNESS: 0.0000025000
EN İYİ FITNESS  : 0.0000001420
ORTALAMA SÜRE   : 0.3790 sn
=================================================================================
```

The result dictionary returned by `optimize()` always contains:
- `best_position` — NumPy array with the best found solution vector
- `best_fitness` — scalar float, the objective function value at `best_position`

---

## Running Tests

The test suite in `test_optimizers.py` uses `pytest` and covers:

- **Initialization tests** — all four optimizers initialize correctly for any dimension
- **Regression tests (easy + medium)** — fitness must fall below defined thresholds after 1000 steps with population 50
- **Stability tests (hard functions)** — result must be finite for `eggholder` and `schwefel`
- **Dimension regression** — `sphere` converges at d=2, 5, and 10

```bash
# Run all tests
pytest test_optimizers.py -v

# Run only regression tests
pytest test_optimizers.py -v -k "regression"

# Run a specific optimizer
pytest test_optimizers.py -v -k "PSO"

# Run a specific optimizer + function combination
pytest test_optimizers.py -v -k "DE and ackley"
```

### Regression thresholds

| Function | Threshold |
|---|---|
| `sphere` | 1.0 |
| `sum_squares` | 1.0 |
| `booth` | 5.0 |
| `matyas` | 5.0 |
| `ackley` | 50.0 |
| `griewank` | 50.0 |
| `rastrigin` | 100.0 |

---

## Parameter Reference

| Algorithm | Flag | Default | Description |
|---|---|---|---|
| All | `--function` | required | Test function name (lowercase) |
| All | `--trials` | 30 | Number of independent runs |
| All | `--dim` | 2 | Search space dimension (ignored for fixed-dim functions) |
| PSO | `--particle-count` | 100 | Swarm size |
| PSO | `--max-iterations` | 500 | Maximum iterations |
| DE | `--population` | 50 | Population size |
| DE | `--max-generations` | 1000 | Maximum generations |
| DE | `--F` | 0.9 | Mutation scale factor |
| DE | `--CR` | 0.9 | Crossover rate |
| ABC | `--colony-size` | 50 | Total colony size (split 50/50) |
| ABC | `--max-cycles` | 1000 | Maximum foraging cycles |
| GEN | `--population-size` | 50 | Population size |
| GEN | `--max-generations` | 1000 | Maximum generations |

---

## Notes

- Functions with **fixed dimensions** (e.g. `eggholder=2`, `hartmann_6d=6`) ignore the `--dim` flag and always run at their canonical dimensionality. This is indicated in the output header as `(SABİT)`.
- The `tolerance` parameter (default `1e-6`) triggers early stopping when `best_fitness ≤ tolerance`. This is useful for zero-minimum functions; for functions like `eggholder` or `schwefel` where the minimum is not near zero, it effectively never fires.
- All bounds are enforced via clipping. Infeasible solutions produced by mutation or crossover are projected back to the nearest boundary point.

---

---

# Metasezgisel Optimizasyon Kıyaslama Paketi

Optimizasyon literatüründeki 47 standart test fonksiyonu üzerinde dört klasik metasezgisel algoritmayı — **PSO**, **DE**, **ABC** ve **GEN** — uygulayan ve karşılaştıran bir araştırma kıyaslama çerçevesi.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Proje Yapısı](#proje-yapısı)
- [Algoritmalar](#algoritmalar)
- [Optimizasyon Fonksiyonları](#optimizasyon-fonksiyonları)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Çıktı Formatı](#çıktı-formatı)
- [Testleri Çalıştırma](#testleri-çalıştırma)
- [Parametre Referansı](#parametre-referansı)

---

## Genel Bakış

Bu paket, metasezgisel algoritmaların sistematik olarak kıyaslanması amacıyla tasarlanmıştır. Her algoritma, birleşik bir `optimize()` arayüzüne sahip bağımsız bir sınıf olarak uygulanmıştır; bu sayede algoritmalar arasında geçiş yapmak ve farklı test fonksiyonları ile boyutlar üzerindeki sonuçları karşılaştırmak kolaylaşır.

Dört algoritmanın ortak tasarım ilkeleri:
- Yapılandırılabilir popülasyon/koloni boyutu
- Esnek alt/üst sınırlar (skaler veya boyut başına dizi)
- Tolerans tabanlı erken durdurma
- Çalışma başına zamanlama ile tekrarlanabilir deneme yapısı

---

## Proje Yapısı

```
.
├── PSO_Optimizer.py          # Parçacık Sürü Optimizasyonu
├── DE_Optimizer.py           # Diferansiyel Evrim
├── ABC_Optimizer.py          # Yapay Arı Kolonisi
├── GEN_Optimizer.py          # Genetik Algoritma
├── Optimization_Functions.py # 47 kıyaslama test fonksiyonu
└── test_optimizers.py        # Pytest regresyon ve kararlılık testleri
```

---

## Algoritmalar

### PSO — Parçacık Sürü Optimizasyonu (`PSO_Optimizer.py`)

Arama uzayında hareket eden bir parçacık sürüsünü simüle eder. Her parçacık kendi kişisel en iyisini takip eder ve global en iyiye doğru çekilir.

**Temel özellikler:**
- İterasyon boyunca `w_max`'tan `w_min`'e doğrusal atalet ağırlığı azalması
- Boyut başına hız kısıtlaması (`v_max = 0.2 × aralık`)
- Bilişsel ve sosyal ivme katsayıları (`c1`, `c2`)

**Varsayılan parametreler:** `particle_count=100`, `max_iterations=500`, `c1=2.0`, `c2=2.0`, `w_max=0.9`, `w_min=0.4`

```bash
python PSO_Optimizer.py --function ackley --trials 30 --dim 5 --particle-count 100 --max-iterations 500
```

---

### DE — Diferansiyel Evrim (`DE_Optimizer.py`)

Mevcut üyelerin doğrusal kombinasyonları aracılığıyla mutant vektörler oluşturarak bir popülasyonu evrimleştirir, ardından hedef vektörle çaprazlama yapar.

**Temel özellikler:**
- Beş mutasyon stratejisi: `DE/rand/1/bin`, `DE/best/1/bin`, `DE/current-to-best/1/bin`, `DE/best/2/bin`, `DE/rand/2/bin`
- Açgözlü seçim (deneme vektörü yalnızca kesin olarak daha iyiyse hedefin yerini alır)
- Yapılandırılabilir ölçek faktörü `F` ve çaprazlama oranı `CR`

**Varsayılan parametreler:** `population=50`, `max_generations=1000`, `F=0.9`, `CR=0.9`, strateji=`DE/rand/1/bin`

```bash
python DE_Optimizer.py --function rastrigin --trials 30 --dim 10 --population 50 --max-generations 1000 --F 0.8 --CR 0.9
```

---

### ABC — Yapay Arı Kolonisi (`ABC_Optimizer.py`)

Bir bal arısı kolonisinin besin arama davranışını modeller. Koloni işçi arılar, gözetçi arılar ve keşifçi arılar olarak üçe ayrılır.

**Temel özellikler:**
- İşçi arılar: her besin kaynağı etrafında yerel arama
- Gözetçi arılar: fitness kalitesine göre ağırlıklı olasılıksal seçim
- Keşifçi arılar: tükenmiş kaynakları terk eder (`trial_count ≥ limit`) ve rastgele yeniden başlatır
- Özel fitness dönüşümü: `f≥0` için `1/(1+f)`, aksi hâlde `1+|f|`

**Varsayılan parametreler:** `colony_size=50`, `max_cycles=1000`, `limit=50`

```bash
python ABC_Optimizer.py --function schwefel --trials 30 --dim 2 --colony-size 50 --max-cycles 1000
```

---

### GEN — Genetik Algoritma (`GEN_Optimizer.py`)

Turnuva seçimi, tek noktalı çaprazlama ve elitizm içeren Gauss mutasyonunu kullanan evrimsel bir algoritma.

**Temel özellikler:**
- `k=2` ile turnuva seçimi
- `crossover_rate` ile kontrol edilen tek noktalı çaprazlama
- `σ = 0.1 × (üst − alt)` ile Gauss mutasyonu
- Elitizm: her nesilde en iyi 2 birey doğrudan aktarılır

**Varsayılan parametreler:** `population_size=50`, `max_generations=1000`, `mutation_rate=0.05`, `crossover_rate=0.95`

```bash
python GEN_Optimizer.py --function rosenbrock --trials 30 --dim 5 --population-size 50 --max-generations 1000
```

---

## Optimizasyon Fonksiyonları

47 fonksiyonun tamamı `OptimizationFunctions` sınıfında uygulanmıştır ve girdi olarak NumPy dizisi `x` kabul eder.

### Fonksiyon Referansı

| Fonksiyon | Boyut | Arama Sınırları | Global Minimum |
|---|---|---|---|
| `sphere` | esnek | [−5.12, 5.12] | **0**'da 0 |
| `ackley` | esnek | [−32.768, 32.768] | **0**'da 0 |
| `rastrigin` | esnek | [−5.12, 5.12] | **0**'da 0 |
| `rosenbrock` | esnek | [−5, 10] | **1**'de 0 |
| `griewank` | esnek | [−600, 600] | **0**'da 0 |
| `schwefel` | esnek | [−500, 500] | **420.96...**'da 0 |
| `levy` | esnek | [−10, 10] | **1**'de 0 |
| `zakharov` | esnek | [−5, 10] | **0**'da 0 |
| `dixon_price` | esnek | [−10, 10] | 0 |
| `styblinski_tang` | esnek | [−5, 5] | **−2.9035**'te −39.166×d |
| `michalewicz` | esnek | [0, π] | −1.8013 (d=2) |
| `beale` | 2 | [−4.5, 4.5] | (3, 0.5)'te 0 |
| `six_hump_camel` | 2 | [−3,−2]×[3,2] | −1.0316 |
| `branin` | 2 | [−5,0]×[10,15] | 0.397887 |
| `eggholder` | 2 | [−512, 512] | (512, 404.23)'te −959.6407 |
| `goldstein_price` | 2 | [−2, 2] | (0, −1)'de 3 |
| `cross_in_tray` | 2 | [−10, 10] | −2.0626 |
| `holder_table` | 2 | [−10, 10] | −19.2085 |
| `schaffer_n2` | 2 | [−100, 100] | **0**'da 0 |
| `schaffer_n4` | 2 | [−100, 100] | 0.292579 |
| `easom` | 2 | [−100, 100] | (π, π)'de −1 |
| `booth` | 2 | [−10, 10] | (1, 3)'te 0 |
| `matyas` | 2 | [−10, 10] | **0**'da 0 |
| `bukin_n6` | 2 | [−15,−3]×[−5,3] | (−10, 1)'de 0 |
| `mccormick` | 2 | [−1.5,−3]×[4,4] | −1.9133 |
| `shubert` | 2 | [−10, 10] | −186.7309 |
| `langermann` | 2 | [0, 10] | negatif |
| `drop_wave` | 2 | [−5.12, 5.12] | **0**'da −1 |
| `dejong_n5` | 2 | [−65.536, 65.536] | ~0.998 |
| `three_hump_camel` | 2 | [−5, 5] | **0**'da 0 |
| `bohachevsky` | 2 | [−100, 100] | **0**'da 0 |
| `hartmann_3d` | 3 | [0, 1] | −3.8628 |
| `hartmann_4d` | 4 | [0, 1] | −3.1344 |
| `hartmann_6d` | 6 | [0, 1] | −3.3224 |
| `colville` | 4 | [−10, 10] | **1**'de 0 |
| `powell` | 4 | [−4, 5] | **0**'da 0 |
| `shekel` | 4 | [0, 10] | −10.5364 |
| `perm` | esnek | [−2, 2] | 0 |
| `perm_0` | esnek | [−2, 2] | 0 |
| `trid` | esnek | [−4, 4] | değişken |
| `power_sum` | esnek | [0, 2] | 0 |
| `sum_squares` | esnek | [−10, 10] | **0**'da 0 |
| `rotated_hyper_ellipsoid` | esnek | [−65.536, 65.536] | **0**'da 0 |
| `sum_of_different_powers` | esnek | [−1, 1] | **0**'da 0 |
| `gramacy_lee` | 1 | [0.5, 2.5] | −0.869 |
| `forrester` | 1 | [0, 1] | −6.0217 |

---

## Kurulum

**Gereksinimler:** Python 3.8+

```bash
pip install numpy pytest
```

Ek bağımlılık gerekmez. Depoyu klonlayın ve betikleri doğrudan çalıştırın.

---

## Kullanım

### Temel kullanım

```bash
python <ALGORİTMA>_Optimizer.py --function <fonksiyon_adı> [seçenekler]
```

### Örnekler

**PSO ile Ackley, 30 deneme, 5 boyut:**
```bash
python PSO_Optimizer.py --function ackley --trials 30 --dim 5
```

**DE ile Schwefel, özel F ve CR değerleri:**
```bash
python DE_Optimizer.py --function schwefel --trials 20 --dim 2 --F 0.7 --CR 0.85
```

**ABC ile Eggholder (sabit 2 boyutlu):**
```bash
python ABC_Optimizer.py --function eggholder --trials 30 --colony-size 100 --max-cycles 2000
```

**GEN ile Hartmann 6D (sabit boyut, dim bayrağı yok sayılır):**
```bash
python GEN_Optimizer.py --function hartmann_6d --trials 30 --population-size 100
```

### Optimizer sınıflarını doğrudan kullanma

Dört optimizer de aynı arayüzü sunar ve kendi betiklerinize gömülebilir:

```python
import numpy as np
from PSO_Optimizer import PSOOptimizer
from Optimization_Functions import OptimizationFunctions

lb = np.full(5, -32.768)
ub = np.full(5,  32.768)

opt = PSOOptimizer(lb, ub, particle_count=100, dimension=5)

funcs = OptimizationFunctions()
result = opt.optimize(funcs.ackley_function, max_iterations=500)

print(result['best_fitness'])   # skaler fitness değeri
print(result['best_position'])  # uzunluğu 5 olan numpy dizisi
```

Aynı yapı `DEOptimizer`, `ABCOptimizer` ve `GENOptimizer` için de geçerlidir; yalnızca `max_generations` veya `max_cycles` parametresi uygun şekilde değiştirilir.

### Tüm algoritmaları tek bir fonksiyon üzerinde karşılaştırma

```python
import numpy as np
from Optimization_Functions import OptimizationFunctions
from PSO_Optimizer import PSOOptimizer
from DE_Optimizer import DEOptimizer
from ABC_Optimizer import ABCOptimizer
from GEN_Optimizer import GENOptimizer

funcs = OptimizationFunctions()
dim = 2
lb = np.full(dim, -5.12)
ub = np.full(dim,  5.12)

configs = {
    'PSO': (PSOOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_iterations=1000)),
    'DE':  (DEOptimizer(lb, ub, 50, dim),  lambda opt: opt.optimize(funcs.rastrigin_function, max_generations=1000)),
    'ABC': (ABCOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_cycles=1000)),
    'GEN': (GENOptimizer(lb, ub, 50, dim), lambda opt: opt.optimize(funcs.rastrigin_function, max_generations=1000)),
}

for name, (opt, run) in configs.items():
    result = run(opt)
    print(f"{name}: {result['best_fitness']:.6f}")
```

---

## Çıktı Formatı

Her çalıştırma standart çıktıya bir tablo yazdırır:

```
=================================================================================
ALGORİTMA: PSO | FONKSİYON: ACKLEY | DIM: 5 (ESNEK)
PARÇACIK: 100 | İTERASYON: 500
HEDEF: 0.0
=================================================================================
No   Best Fitness       Time(s)    Parametre 1    Parametre 2    ...
---------------------------------------------------------------------------------
1    0.00000142         0.3821     0.0001         -0.0000        ...
2    0.00000389         0.3754     0.0000          0.0001        ...
...
=================================================================================
ORTALAMA FITNESS: 0.0000025000
EN İYİ FITNESS  : 0.0000001420
ORTALAMA SÜRE   : 0.3790 sn
=================================================================================
```

`optimize()` tarafından döndürülen sonuç sözlüğü her zaman şunları içerir:
- `best_position` — en iyi bulunan çözüm vektörünü içeren NumPy dizisi
- `best_fitness` — `best_position`'daki amaç fonksiyonu değeri (skaler float)

---

## Testleri Çalıştırma

`test_optimizers.py` dosyasındaki test paketi `pytest` kullanır ve şunları kapsar:

- **Başlatma testleri** — dört optimizer de herhangi bir boyut için doğru şekilde başlatılır
- **Regresyon testleri (kolay + orta)** — popülasyon 50 ile 1000 adımdan sonra fitness, tanımlı eşik değerlerin altına düşmelidir
- **Kararlılık testleri (zor fonksiyonlar)** — `eggholder` ve `schwefel` için sonuç sonlu olmalıdır
- **Boyut regresyonu** — `sphere`, d=2, 5 ve 10'da yakınsır

```bash
# Tüm testleri çalıştır
pytest test_optimizers.py -v

# Yalnızca regresyon testlerini çalıştır
pytest test_optimizers.py -v -k "regression"

# Belirli bir optimizer için çalıştır
pytest test_optimizers.py -v -k "PSO"

# Belirli bir optimizer + fonksiyon kombinasyonu için çalıştır
pytest test_optimizers.py -v -k "DE and ackley"
```

### Regresyon eşik değerleri

| Fonksiyon | Eşik Değeri |
|---|---|
| `sphere` | 1.0 |
| `sum_squares` | 1.0 |
| `booth` | 5.0 |
| `matyas` | 5.0 |
| `ackley` | 50.0 |
| `griewank` | 50.0 |
| `rastrigin` | 100.0 |

---

## Parametre Referansı

| Algoritma | Bayrak | Varsayılan | Açıklama |
|---|---|---|---|
| Tümü | `--function` | zorunlu | Test fonksiyonu adı (küçük harf) |
| Tümü | `--trials` | 30 | Bağımsız çalıştırma sayısı |
| Tümü | `--dim` | 2 | Arama uzayı boyutu (sabit boyutlu fonksiyonlarda yok sayılır) |
| PSO | `--particle-count` | 100 | Sürü boyutu |
| PSO | `--max-iterations` | 500 | Maksimum iterasyon sayısı |
| DE | `--population` | 50 | Popülasyon boyutu |
| DE | `--max-generations` | 1000 | Maksimum nesil sayısı |
| DE | `--F` | 0.9 | Mutasyon ölçek faktörü |
| DE | `--CR` | 0.9 | Çaprazlama oranı |
| ABC | `--colony-size` | 50 | Toplam koloni boyutu (50/50 bölünür) |
| ABC | `--max-cycles` | 1000 | Maksimum besin arama döngüsü |
| GEN | `--population-size` | 50 | Popülasyon boyutu |
| GEN | `--max-generations` | 1000 | Maksimum nesil sayısı |

---

## Notlar

- **Sabit boyutlu** fonksiyonlar (örn. `eggholder=2`, `hartmann_6d=6`) `--dim` bayrağını yok sayar ve her zaman kanonik boyutlarında çalışır. Bu durum çıktı başlığında `(SABİT)` olarak gösterilir.
- `tolerance` parametresi (varsayılan `1e-6`), `best_fitness ≤ tolerance` olduğunda erken durdurmayı tetikler. Bu, sıfır minimumlu fonksiyonlar için kullanışlıdır; `eggholder` veya `schwefel` gibi minimumu sıfıra yakın olmayan fonksiyonlarda pratikte hiç devreye girmez.
- Tüm sınırlar kırpma yoluyla uygulanır. Mutasyon veya çaprazlama sonucu oluşan geçersiz çözümler en yakın sınır noktasına yansıtılır.
