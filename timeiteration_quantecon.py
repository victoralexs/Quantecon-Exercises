# Optimal Growth III: Time Iteration

# Here, we will solve the Optimal Growth Problem using the so-called time iteration rather than the VFI.
# The method is based in the Euler Equation and usually is more efficient than the VFI based method.

!pip install interpolation

# Again, we are going to use the Numba library to make the coder faster.

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt

from interpolation import interp
from quantecon.optimize import brentq
from numba import njit, jitclass, float64

# Our objective is to write the Euler Equation and solve a iteration based on the Coleman-Reffett Operator due Euler iteration.

# We are going to continue to use a log utility form with the same structure for shocks and capital accumulation.

# Let us first define our analytical solutions to compare later with our simulations:

def v_star(y, α, β, μ):
    """
    True value function
    """
    c1 = np.log(1 - α * β) / (1 - β)
    c2 = (μ + α * np.log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)
    return c1 + c2 * (c3 - c4) + c4 * np.log(y)

def σ_star(y, α, β):
    """
    True optimal policy
    """
    return (1 - α * β) * y

# Using the Numba compilator, we will create a class OptimalGrowthModel that creates the derivatives of u and f to build our Euler:

opt_growth_data = [
    ('α', float64),  # Production parameter
    ('β', float64),  # Discount factor
    ('μ', float64),  # Shock location parameter
    ('s', float64),  # Shock scale parameter
    ('grid', float64[:]),  # Grid (array)
    ('shocks', float64[:])  # Shock draws (array)
]


@jitclass(opt_growth_data)
class OptimalGrowthModel:

    def __init__(self,
                 α=0.4,
                 β=0.96,
                 μ=0,
                 s=0.1,
                 grid_max=4,
                 grid_size=120,
                 shock_size=250,
                 seed=1234):
        self.α, self.β, self.μ, self.s = α, β, μ, s

        # Set up grid
        self.grid = np.linspace(1e-5, grid_max, grid_size)

        # Store shocks (with a seed, so results are reproducible)
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))

    def f(self, k):
        "The production function"
        return k ** self.α

    def u(self, c):
        "The utility function"
        return np.log(c)

    def f_prime(self, k):
        "Derivative of f"
        return self.α * (k ** (self.α - 1))

    def u_prime(self, c):
        "Derivative of u"
        return 1 / c

    def u_prime_inv(self, c):
        "Inverse of u'"
        return 1 / c

# Now, in order to write the Euler Equation, we set the function euler_diff, which gives us as root the Euler Operator Ksigma(y).

@njit
def euler_diff(c, σ, y, og):
    """
    Set up a function such that the root with respect to c,
    given y and σ, is equal to Kσ(y).

    """

    β, shocks, grid = og.β, og.shocks, og.grid
    f, f_prime, u_prime = og.f, og.f_prime, og.u_prime

    # First turn w into a function via interpolation
    σ_func = lambda x: interp(grid, σ, x)

    # Now set up the function we need to find the root of.
    vals = u_prime(σ_func(f(y - c) * shocks)) * f_prime(y - c) * shocks
    return u_prime(c) - β * np.mean(vals)

# Now we define the operator K that solves the euler_diff above, i.e., the Coleman-Reffett operator

@njit
def K(σ, og):
    """
    The Coleman-Reffett operator

     Here og is an instance of OptimalGrowthModel.
    """

    β = og.β
    f, f_prime, u_prime = og.f, og.f_prime, og.u_prime
    grid, shocks = og.grid, og.shocks

    σ_new = np.empty_like(σ)
    for i, y in enumerate(grid):
        y = grid[i]
        # Solve for optimal c at y
        c_star = brentq(euler_diff, 1e-10, y-1e-10, args=(σ, y, og))[0]     # brenqt is a function to solve equations.
        σ_new[i] = c_star

    return σ_new

# In order to test if the algorith works, let's define an instance and plot some interates of K, starting from sigma(y) = y:

og = OptimalGrowthModel()
grid = og.grid

n = 15
σ = grid.copy()  # Set initial condition

fig, ax = plt.subplots()
lb = 'initial condition $\sigma(y) = y$'
ax.plot(grid, σ, color=plt.cm.jet(0), alpha=0.6, label=lb)

for i in range(n):
    σ = K(σ, og)
    ax.plot(grid, σ, color=plt.cm.jet(i / n), alpha=0.6)

# Update one more time and plot the last iterate in black
σ = K(σ, og)
ax.plot(grid, σ, color='k', alpha=0.8, label='last iterate')

ax.legend()

plt.show()

# We can see that according to the iterates, it converges to the equilibrium.

# Now, defining the function solve_model_time_iter to approximate the solution taking an instance from the OptimalGrowthModel, we can obtain the optimal policy functin by time iteration:

def solve_model_time_iter(model,    # Class with model information
                          σ,        # Initial condition
                          tol=1e-4,
                          max_iter=1000,
                          verbose=True,
                          print_skip=25):

    # Set up loop
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        σ_new = K(σ, model)
        error = np.max(np.abs(σ - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        σ = σ_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return σ_new

# Now, calling the iteration:

σ_init = np.copy(og.grid)
σ = solve_model_time_iter(og, σ_init)

# And plotting it with the analytical solution, we can see it converges perfectly:

fig, ax = plt.subplots()

ax.plot(og.grid, σ, lw=2,
        alpha=0.8, label='approximate policy function')

ax.plot(og.grid, σ_star(og.grid, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()

###########

# Exercise 1: Re-do it with a CRRA utility function.

# First, we need to define a class with CRRA utility function:

opt_growth_data = [
    ('α', float64),  # Production parameter
    ('β', float64),  # Discount factor
    ('μ', float64),  # Shock location parameter
    ('γ', float64),  # Preference parameter
    ('s', float64),  # Shock scale parameter
    ('grid', float64[:]),  # Grid (array)
    ('shocks', float64[:])  # Shock draws (array)
]


@jitclass(opt_growth_data)
class OptimalGrowthModel_CRRA:

    def __init__(self,
                 α=0.4,
                 β=0.96,
                 μ=0,
                 s=0.1,
                 γ=1.5,
                 grid_max=4,
                 grid_size=120,
                 shock_size=250,
                 seed=1234):
        self.α, self.β, self.γ, self.μ, self.s = α, β, γ, μ, s

        # Set up grid
        self.grid = np.linspace(1e-5, grid_max, grid_size)

        # Store shocks (with a seed, so results are reproducible)
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))

    def f(self, k):
        "The production function."
        return k ** self.α

    def u(self, c):
        "The utility function."
        return c ** (1 - self.γ) / (1 - self.γ)

    def f_prime(self, k):
        "Derivative of f."
        return self.α * (k ** (self.α - 1))

    def u_prime(self, c):
        "Derivative of u."
        return c ** (-self.γ)

    def u_prime_inv(c):
        return c ** (-1 / self.γ)

    # And defining a new instance:

    og_CRRA = OptimalGrowthModel_CRRA()

    # Now iterating it:

    %%time
    σ = solve_model_time_iter(og_CRRA, σ_init)

    fig, ax = plt.subplots()

    ax.plot(og.grid, σ, lw=2,
            alpha=0.8, label='approximate policy function')

    ax.legend()
    plt.show()

    # Which works!