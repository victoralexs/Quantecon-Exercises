# Optimal Growth I: The Stochastic Optimal Growth

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# We are gonna assume y_t as the state variable.

# The mechanism follows:
    # 1. Begin with an array of values {v_1,...v_I} representing the values of some initial function v on the grid points {y_1,...,y_I}.
    # 2. Build a function v^ on the state space R+ by linear interpolation, based on these data points.
    # 3. Obtain and record the value of Tv^(y_i) on each grid point y_i by repeatedly solving the Bellman Operator.
    # 4. Unless some stopping function condition is satisfied, set {v_1,...v_I} = {Tv(y_1),...,Tv(y_I)}, i.e., the convergence to a fixed point, return to step 2.

# Scalar Maximization

# Same as former lectures, we will define the maximize function using the scipy minimization of -g, using the minimize_scalar function:

def maximize(g, a, b, args):
    """
    Maximize the function g over the interval [a, b].

    We use the fact that the maximizer of g on any interval is
    also the minimizer of -g.  The tuple args collects any extra
    arguments to g.

    Returns the maximal value and the maximizer.
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum

# Now, we are gonna define a class for the Optimal Growth Model, defining its parameters and grids and initial values.

# Also, defining a method that solves the RHS of the Bellman Equation:

class OptimalGrowthModel:

    def __init__(self,
                 u,            # utility function
                 f,            # production function
                 β=0.96,       # discount factor
                 μ=0,          # shock location parameter
                 s=0.1,        # shock scale parameter
                 grid_max=4,
                 grid_size=120,
                 shock_size=250,
                 seed=1234):

        self.u, self.f, self.β, self.μ, self.s = u, f, β, μ, s

        # Set up grid
        self.grid = np.linspace(1e-4, grid_max, grid_size)

        # Store shocks (with a seed, so results are reproducible)
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))

    def state_action_value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, shocks = self.u, self.f, self.β, self.shocks

        v = interp1d(self.grid, v_array)

        return u(c) + β * np.mean(v(f(y - c) * shocks))

    # Where in the last line above, the expectation if computed by Monte Carlo using the approximation to a discrete value.

    # Now, let's implement the Bellman Operator:

    def T(v, og):
        """
        The Bellman operator.  Updates the guess of the value function
        and also computes a v-greedy policy.

          * og is an instance of OptimalGrowthModel
          * v is an array representing a guess of the value function

        """
        v_new = np.empty_like(v)
        v_greedy = np.empty_like(v)

        for i in range(len(grid)):
            y = grid[i]

            # Maximize RHS of Bellman equation at state y
            c_star, v_max = maximize(og.state_action_value, 1e-10, y, (y, v))
            v_new[i] = v_max
            v_greedy[i] = c_star

        return v_greedy, v_new

    # Now, let's state the analytical solutions for the value function and the policy function in order to compare with our simulation:

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

    # Creating  an instance of the model with the above primitives and assigning it to the variable og, defined above in the Bellman Operator:

    α = 0.4

    def fcd(k):
        return k ** α

    og = OptimalGrowthModel(u=np.log, f=fcd)

# Just for illustration, let us apply the Bellman Operator in the analytical solution v_star.
# Since v_star is the fixed point, i.e., the solution, the result of course will be v_star.
# That's why it must converge:

grid = og.grid

v_init = v_star(grid, α, og.β, og.μ)    # Start at the solution
v_greedy, v = T(v_init, og)             # Apply T once

fig, ax = plt.subplots()
ax.set_ylim(-35, -24)
ax.plot(grid, v, lw=2, alpha=0.6, label='$Tv^*$')
ax.plot(grid, v_init, lw=2, alpha=0.6, label='$v^*$')
ax.legend()
plt.show()

# Now let's iterate the Bellman Operator starting with an arbitrary initial condition.
# Let's say, v(y) = 5 ln(y)

v = 5 * np.log(grid)  # An initial condition
n = 35

fig, ax = plt.subplots()

ax.plot(grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='Initial condition')

for i in range(n):
    v_greedy, v = T(v, og)  # Apply the Bellman operator
    ax.plot(grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax.plot(grid, v_star(grid, α, og.β, og.μ), 'k-', lw=2,
        alpha=0.8, label='True value function')

ax.legend()
ax.set(ylim=(-40, 10), xlim=(np.min(grid), np.max(grid)))
plt.show()

# We can notice that, after 36 iterations, we are getting closer to the analytical solution. But we need more iterations!

# Now, let's create a solver that iterates the value function until the convergence, with 1000 iterations.

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

# Finally, let's compute an approximate solution:

v_greedy, v_solution = solve_model(og)

# Computing and plotting the v_solution with the analytical solution, we can see that it matches:

fig, ax = plt.subplots()

ax.plot(grid, v_solution, lw=2, alpha=0.6,
        label='Approximate value function')

ax.plot(grid, v_star(grid, α, og.β, og.μ), lw=2,
        alpha=0.6, label='True value function')

ax.legend()
ax.set_ylim(-35, -24)
plt.show()

# Now, doing the same with the converged optimal policy function and comparing with the analytical solution defined in the beginning of this code:

fig, ax = plt.subplots()

ax.plot(grid, v_greedy, lw=2,
        alpha=0.6, label='approximate policy function')

ax.plot(grid, σ_star(grid, α, og.β), '--',
        lw=2, alpha=0.6, label='true policy function')

ax.legend()
plt.show()

# It does converge!

###################

# Now, let's make some changes and exercise a little bit more our algorithm.

# Exercise 1: Use the CRRA model as utility function and simulate for the OptimalGrowthModel. Also, compute the time. Assume gamma = 1.5

# We first need to define our CRRA function and state the og instance with this new utility function (remember that, before, we were assuming a simple log form numpy library):

γ = 1.5   # Preference parameter

def u_crra(c):
    return (c**(1 - γ) - 1) / (1 - γ)

og = OptimalGrowthModel(u=u_crra, f=fcd)


# Since we have already defined our og instance, we just need to apply it to the function solve_model:

%%time
v_greedy, v_solution = solve_model(og)

# Plotting it:

fig, ax = plt.subplots()

ax.plot(grid, v_greedy, lw=2,
        alpha=0.6, label='Approximate optimal policy')

ax.legend()
plt.show()




