# Optimal Growth II: Accelerating the Code with Numba

# In this code we are gonna exercise the Numba method to faster the code for VFI.
# All methods are grom quantecon exercies from T. Sargent website.

# !pip install --upgrade quantecon
# !pip install --upgrade interpolation

import numpy as np
import matplotlib.pyplot as plt
from interpolation import interp
from numba import jit, njit, jitclass, prange, float64, int32
from quantecon.optimize.scalar_maximization import brent_max
import quantecon as qd

# The logic is that, numba make the code faster, but with less flexibility in the algorithm.
# We are going to use the JIT (just-in-time) compilation method to accelerate the code.
# interpolation and brent_max functions are designed for embedding JIT-compiled code.

# Let us start with the basics. A model with a log utility  with capital and a shock with exponential distribution.

# In order to compare our results, first define the true (analytical) solutions for the value and policy function:

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

# In order to compile with the Numba's @jitclass, we need to specify the type of data before define the class of the growth model.
# For this, we define a list called opt_growth_data with the parameters.
# We will use the same utility and production function from the growth code (available in the victoralexs GitHub)'
# This is where flexibility is sacrificed in order to gain more speed.

opt_growth_data = [
    ('α', float64),          # Production parameter
    ('β', float64),          # Discount factor
    ('μ', float64),          # Shock location parameter
    ('s', float64),          # Shock scale parameter
    ('grid', float64[:]),    # Grid (array)
    ('shocks', float64[:])   # Shock draws (array)
]

# Now we are going to define our class compiling with de @jit function by adding the @jitclass on the parameters above.
# Assuming a log utility function:

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

    def u_prime(self, c):                   # We are going to use this later. Not now.
        "Derivative of u"
        return 1 / c

    def u_prime_inv(self, c):
        "Inverse of u'"
        return 1 / c

# Now, defining our Bellman Operator, i.e., a function, compiled with the JIT, that returns the RHS side of the Bellman Equation.
# We are going to define the function state_action_value that returns a value of c given state y:

@njit
def state_action_value(c, y, v_array, og):
    """
    Right hand side of the Bellman equation.

     * c is consumption
     * y is income
     * og is an instance of OptimalGrowthModel
     * v_array represents a guess of the value function on the grid

    """

    u, f, β, shocks = og.u, og.f, og.β, og.shocks

    v = lambda x: interp(og.grid, v_array, x)

    return u(c) + β * np.mean(v(f(y - c) * shocks))

# Now, we can define the method T(v,og), the Bellman operator, that maximizes the RHS of the Bellman Equation and update the value for a defined grid.
# Also, we are using the @jit to compilate the Numba Accelerator:

@jit(nopython=True)
def T(v, og):
    """
    The Bellman operator.

     * og is an instance of OptimalGrowthModel
     * v is an array representing a guess of the value function

    """

    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)

    for i in range(len(og.grid)):
        y = og.grid[i]

        # Maximize RHS of Bellman equation at state y
        result = brent_max(state_action_value, 1e-10, y, args=(y, v, og))
        v_greedy[i], v_new[i] = result[0], result[1]

    return v_greedy, v_new

# Now, we know define the solve_model function to perform iteration until convergence.
# Note that the solve_model code is the same as the growth code without numba @jit:

def solve_model(og,
                tol=1e-4,
                max_iter=1000,
                verbose=True,
                print_skip=25):
    """
    Solve model by iterating with the Bellman operator.

    """

    # Set up loop
    v = og.u(og.grid)  # Initial condition
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_greedy, v_new = T(v, og)
        error = np.max(np.abs(v - v_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        v = v_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return v_greedy, v_new

# Now, let's compute our approximate solution using the @jit compilation.

# First, defining the standard instances for the og model:

og = OptimalGrowthModel()

# Finally, iterating:

%%time
v_greedy, v_solution = solve_model(og)

# Plotting our approximated value function and comparing it with the analytical solution.
# We can see that it matches.

fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2,
        alpha=0.8, label='approximate policy function')

ax.plot(og.grid, σ_star(og.grid, og.α, og.β), 'k--',
        lw=2, alpha=0.8, label='true policy function')

ax.legend()
plt.show()

# If we want to see the absolute deviation from the approximate model and the analytical solution:

np.max(np.abs(v_greedy - σ_star(og.grid, og.α, og.β)))

####################

# Exercise 1: Time how long it takes to iterate the Bellman operator 20 times, starting from initial condition u(y) = v(y)

# The exercise is basic. First, let's guess our value function with the u(y) form defined above, with the log structure for utility function.

# Setting up the initial condition for v:
v = og.u(og.grid)

# Now, iterating the Bellman operator 20 times:

# %%time
for i in range(20):
    v_greedy, v_new = T(v, og)
    v = v_new

#############

# Exercise 2: Compute and estimate the optimal policy the model but using as utility function the CRRA function with parameter gamma = 1.5

# Because hte @jit does not support inheritance, we need to copy the class and change the parameters. I am going to to by myself:

opt_growth_data = [
    ('α', float64),          # Production parameter
    ('β', float64),          # Discount factor
    ('μ', float64),          # Shock location parameter
    ('γ', float64),          # Preference parameter
    ('s', float64),          # Shock scale parameter
    ('grid', float64[:]),    # Grid (array)
    ('shocks', float64[:])   # Shock draws (array)
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

        self.α, self.β, self.μ, self.s, self.γ = α, β, μ, s, γ

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
        return (c**(1 - self.γ) - 1) / (1 - self.γ)

    def f_prime(self, k):
        "Derivative of f"
        return self.α * (k ** (self.α - 1))

    def u_prime(self, c):                   # We are going to use this later. Not now.
        "Derivative of u"
        return c**(-self.γ)

    def u_prime_inv(self, c):
        "Inverse of u'"
        return c**(-1 / self.γ)

    # Now, creating the instance for og_crra:

    og_crra = OptimalGrowthModel_CRRA()

    # Finally, solving the iteration using the solve_model function:

%%time
v_greedy, v_solution = solve_model(og_crra)

# And plotting it:

fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2,
        alpha=0.6, label='Approximate value function')

ax.legend(loc='lower right')
plt.show()

################

# Exercise 3: Return to the log utility function and compute a figure of the optimal policy function for different values of beta, the discount factor.
            # Simulate it for 100 elements with an initial value of y_0 = 0.1, discount values {0.8, 0.9 and 0.98} and shocks s = 0.05

# The intuition is that, more patient agents typically have higher wealth during life cycle. Let's show it:

# The exercise asks to simulate a time-series for the wealth given the consumption policy function sigma for 100 iterations. Thus, we need to create a function called simulate_og that gives us this time series.
# In the function we define the initial value and the time series lenght.
# Also, define the y_t dynamics for the growth model that we are using.

def simulate_og(σ_func, og, y0=0.1, ts_length=100):
    '''
    Compute a time series given consumption policy σ.
    '''
    y = np.empty(ts_length)
    ξ = np.random.randn(ts_length-1)
    y[0] = y0
    for t in range(ts_length-1):
        y[t+1] = (y[t] - σ_func(y[t]))**og.α * np.exp(og.μ + og.s * ξ[t])
    return y

# Now, plotting for different values of beta, using our OptimalGrowthModel with the log utility and defining our shock s:

fig, ax = plt.subplots()

for β in (0.8, 0.9, 0.98):

    og = OptimalGrowthModel(β=β, s=0.05)

    v_greedy, v_solution = solve_model(og)

    # Define an optimal policy function
    σ_func = lambda x: interp(og.grid, v_greedy, x)
    y = simulate_og(σ_func, og)
    ax.plot(y, lw=2, alpha=0.6, label=rf'$\beta = {β}$')

ax.legend(loc='lower right')
plt.show()

# We can see that when beta is higher, the lifetime wealth is also higher.

