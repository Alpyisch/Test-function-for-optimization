import numpy as np
import pytest

from PSOdeneme import PSOOptimizer
from Optimization_Functions import OptimizationFunctions


def test_pso_sphere_function():
    """
    Test PSO on a simple Sphere function (much easier than Eggholder).
    Sphere has a single global minimum at origin.
    """
    lb = np.array([-5.12, -5.12])
    ub = np.array([5.12, 5.12])
    dim = 2

    # Set random seed for reproducibility
    np.random.seed(42)

    optimizer = PSOOptimizer(
        lower_bound=lb,
        upper_bound=ub,
        particle_count=30,
        dimension=dim,
        c1=2.0,
        c2=2.0
    )

    obj_func = OptimizationFunctions().sphere_function

    result = optimizer.optimize(
        objective_func=obj_func,
        max_iterations=1000
    )

    # --- assertions ---
    assert result is not None
    assert "best_fitness" in result
    assert "best_position" in result

    best_fit = result["best_fitness"]
    best_pos = result["best_position"]

    # Sphere function minimum: f(0,0) = 0
    assert best_fit < 0.01, f"Expected fitness < 0.01, got {best_fit}"
    assert np.allclose(best_pos, [0.0, 0.0], atol=0.1), \
        f"Expected position near [0,0], got {best_pos}"


def test_pso_rosenbrock_function():
    """
    Test PSO on Rosenbrock function (moderate difficulty).
    Global minimum at (1,1) with f=0
    """
    lb = np.array([-5, -5])
    ub = np.array([10, 10])
    dim = 2

    np.random.seed(123)

    optimizer = PSOOptimizer(
        lower_bound=lb,
        upper_bound=ub,
        particle_count=50,
        dimension=dim,
        c1=2.0,
        c2=2.0
    )

    obj_func = OptimizationFunctions().rosenbrock_function

    result = optimizer.optimize(
        objective_func=obj_func,
        max_iterations=2000
    )

    assert result is not None
    best_fit = result["best_fitness"]
    best_pos = result["best_position"]

    # Rosenbrock is harder, so allow more tolerance
    assert best_fit < 1.0, f"Expected fitness < 1.0, got {best_fit}"
    assert np.allclose(best_pos, [1.0, 1.0], atol=0.5), \
        f"Expected position near [1,1], got {best_pos}"


def test_pso_eggholder_reasonable_convergence():
    """
    Test PSO on Eggholder with realistic expectations.
    Don't expect global optimum, just reasonable convergence.
    """
    lb = np.array([-512, -512])
    ub = np.array([512, 512])
    dim = 2

    np.random.seed(999)

    optimizer = PSOOptimizer(
        lower_bound=lb,
        upper_bound=ub,
        particle_count=100,  # More particles for harder problem
        dimension=dim,
        c1=2.0,
        c2=2.0
    )

    obj_func = OptimizationFunctions().eggholder_function

    result = optimizer.optimize(
        objective_func=obj_func,
        max_iterations=5000
    )

    assert result is not None
    best_fit = result["best_fitness"]

    # Eggholder global minimum is -959.6407
    # But it's very hard, so just check if we get reasonably low
    assert best_fit < -500, \
        f"Expected fitness < -500 for Eggholder, got {best_fit}"
    
    # Optional: Check if we're in a reasonable region
    assert best_fit > -1000, \
        f"Fitness suspiciously low: {best_fit}"


def test_pso_initialization():
    """
    Test that PSO initializes particles correctly.
    """
    lb = np.array([-10, -10])
    ub = np.array([10, 10])
    
    optimizer = PSOOptimizer(
        lower_bound=lb,
        upper_bound=ub,
        particle_count=20,
        dimension=2
    )

    # Check particles are within bounds
    assert np.all(optimizer.particles >= lb)
    assert np.all(optimizer.particles <= ub)
    
    # Check velocities are initialized
    assert optimizer.velocities.shape == (20, 2)
    
    # Check best scores are initialized to infinity
    assert np.all(optimizer.pbest_scores == np.inf)
    assert optimizer.gbest_score == np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])