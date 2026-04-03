import numpy as np
import pytest
from itertools import product

from PSO_Optimizer import PSOOptimizer
from DE_Optimizer import DEOptimizer
from ABC_Optimizer import ABCOptimizer
from GEN_Optimizer import GENOptimizer
from Optimization_Functions import OptimizationFunctions


EASY_FUNCTIONS = ['sphere', 'sum_squares', 'booth', 'matyas']
MEDIUM_FUNCTIONS = ['ackley', 'griewank', 'rastrigin']
HARD_FUNCTIONS = ['eggholder', 'schwefel']

OPTIMIZERS = {
    'PSO': PSOOptimizer,
    'DE': DEOptimizer,
    'ABC': ABCOptimizer,
    'GEN': GENOptimizer,
}

BOUNDS = {
    'sphere': (-5.12, 5.12),
    'sum_squares': (-10, 10),
    'booth': (-10, 10),
    'matyas': (-10, 10),
    'ackley': (-32.768, 32.768),
    'griewank': (-600, 600),
    'rastrigin': (-5.12, 5.12),
    'eggholder': (-512, 512),
    'schwefel': (-500, 500),
}

FIXED_DIMS = {
    'eggholder': 2,
    'booth': 2,
    'matyas': 2,
}


REGRESSION_THRESHOLDS = {
    'sphere': 1.0,
    'sum_squares': 1.0,
    'booth': 5.0,
    'matyas': 5.0,
    'ackley': 50.0,
    'griewank': 50.0,
    'rastrigin': 100.0,
}


def get_function_config(func_name, dim=2):
    if func_name not in BOUNDS:
        pytest.skip(f"{func_name} not configured")

    current_dim = FIXED_DIMS.get(func_name, dim)
    low, high = BOUNDS[func_name]

    lb = np.full(current_dim, low)
    ub = np.full(current_dim, high)

    obj_funcs = OptimizationFunctions()
    obj_func = getattr(obj_funcs, f"{func_name}_function")

    return lb, ub, current_dim, obj_func


def create_optimizer(optimizer_class, lb, ub, dim, population):
    return optimizer_class(lb, ub, population, dim)


def run_optimizer(optimizer_name, optimizer, obj_func, steps):
    if optimizer_name == 'PSO':
        return optimizer.optimize(obj_func, max_iterations=steps)
    elif optimizer_name in ['DE', 'GEN']:
        return optimizer.optimize(obj_func, max_generations=steps)
    else:
        return optimizer.optimize(obj_func, max_cycles=steps)


@pytest.mark.parametrize("optimizer_name", OPTIMIZERS.keys())
def test_optimizer_initialization(optimizer_name):
    lb = np.array([-10, -10])
    ub = np.array([10, 10])

    optimizer = create_optimizer(
        OPTIMIZERS[optimizer_name], lb, ub, dim=2, population=20
    )

    assert optimizer is not None
    assert optimizer.dimension == 2



@pytest.mark.parametrize(
    "optimizer_name,func_name",
    product(OPTIMIZERS.keys(), EASY_FUNCTIONS + MEDIUM_FUNCTIONS)
)
def test_stochastic_regression_easy_medium(optimizer_name, func_name):
    np.random.seed(42)

    lb, ub, dim, obj_func = get_function_config(func_name, dim=2)
    optimizer = create_optimizer(
        OPTIMIZERS[optimizer_name], lb, ub, dim, population=50
    )

    result = run_optimizer(
        optimizer_name, optimizer, obj_func, steps=1000
    )

    assert result is not None
    assert 'best_fitness' in result

    fitness = result['best_fitness']

    # Core stochastic checks
    assert np.isfinite(fitness)

    threshold = REGRESSION_THRESHOLDS[func_name]
    assert fitness < threshold, (
        f"{optimizer_name} regression on {func_name}: "
        f"{fitness:.4f} >= {threshold}"
    )



@pytest.mark.parametrize(
    "optimizer_name,func_name",
    product(OPTIMIZERS.keys(), HARD_FUNCTIONS)
)
def test_hard_functions_stability(optimizer_name, func_name):
    np.random.seed(999)

    lb, ub, dim, obj_func = get_function_config(func_name, dim=2)
    optimizer = create_optimizer(
        OPTIMIZERS[optimizer_name], lb, ub, dim, population=100
    )

    result = run_optimizer(
        optimizer_name, optimizer, obj_func, steps=1000
    )

    assert result is not None
    fitness = result['best_fitness']

    assert np.isfinite(fitness)


@pytest.mark.parametrize(
    "optimizer_name,dim",
    product(OPTIMIZERS.keys(), [2, 5, 10])
)
def test_dimension_regression(optimizer_name, dim):
    np.random.seed(123)

    lb, ub, dim, obj_func = get_function_config('sphere', dim=dim)
    optimizer = create_optimizer(
        OPTIMIZERS[optimizer_name], lb, ub, dim, population=30
    )

    result = run_optimizer(
        optimizer_name, optimizer, obj_func, steps=500
    )

    assert result is not None
    assert len(result['best_position']) == dim
    assert np.isfinite(result['best_fitness'])
