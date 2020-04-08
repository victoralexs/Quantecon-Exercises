# Lecture 1: OOP 1: Introduction to Object Oriented Programming

import numpy as np
import matplotlib.pyplot as plt

s = 'This is a string'
type(s)

x = 42   # Now let's create an integer
type(x)

int('300') + 400   # To add as numbers, change the string to an integer

y = 2.5
z = 2.5
id(y)
id(z)

x = 42
x

x.imag

x.__class__

# Any name after the dot in a object, as above, is called an attribute of the object.

# Attributes that act like functions are called methods.

x = ['foo', 'bar']
callable(x.append)

callable(x.__doc__)

# Methods act on the data contained in the object they belong to, or combine that data with other data

x = ['a', 'b']
x.append('c')
s = 'This is a string'
s.upper()

s.lower()

s.replace('This', 'That')

# Method calls are common in Python:

x = ['a', 'b']
x[0] = 'aa'  # Item assignment using square bracket notation
x

# Which is the same that call the method 'setitem':

x = ['a', 'b']
x.__setitem__(0, 'aa')  # Equivalent to x[0] = 'aa'
x

# Example: creating functions. When Python reads a function definition, it creates a function object and stores it in memory:

def f(x): return x**2
f

type(f)

id(f)

f.__name__

f.__class__

f(3) # which is the same as

f.__call__(3)

import math

id(math)

###########################################################################

# Lecture 2: OOP II: Building Classes

# We are now work with class definitions. Basically, classes are blueprints that help you to build objects according to your own specifications.

import numpy as np
import matplotlib.pyplot as plt

# Key concepts

x = [1, 5, 4]
x.sort()
x
x.__class__

dir(x) # A command to view all attributes of x.

# Example: A Consumer Class

# The 'class' below indicate that we are building a class, composed by wealth (data) and 3 functions/methods (__init__, earn and spend).

# wealth is called a instance data since each consumer that it will be created will have its own wealth data.

# earn is a method that earn(y) increments consumers' wealth by y

# spend is a method that spend(x) either decreases wealth by x or returns an error if insufficient funds exist.

# __init__ is a method that acts as a constructor method. Basically, whenever we create a class, this method is called.
# It sets up a 'namespace' to hold the instance data.

# self is an argument mandatory to define classes and methods. Also, any referenced within class should be called by the self.method_name

class Consumer:

    def __init__(self, w):
        "Initialize consumer with w dollars of wealth"
        self.wealth = w

    def earn(self, y):
        "The consumer earns y dollars"
        self.wealth += y

    def spend(self, x):
        "The consumer spends x dollars if feasible"
        new_wealth = self.wealth - x
        if new_wealth < 0:
            print("Insufficent funds")
        else:
            self.wealth = new_wealth


# Application

c1 = Consumer(10)  # Create instance with initial wealth 10
c1.spend(5)
c1.wealth

# Now let's see a case that an ind spends more than its wealth.

c1.earn(15)
c1.spend(100)
c1.wealth

# Note that the process is continuous.

# Adding a consumer 2.

c1 = Consumer(10)
c2 = Consumer(12)
c2.spend(4)
c2.wealth

c1.wealth

# Examples

# 1. The Solow Growth Model

class Solow:
    r"""
    Implements the Solow growth model with the update rule

        k_{t+1} = [(s z k^α_t) + (1 - δ)k_t] /(1 + n)

    """
    def __init__(self, n=0.05,  # population growth rate
                       s=0.25,  # savings rate
                       δ=0.1,   # depreciation rate
                       α=0.3,   # share of labor
                       z=2.0,   # productivity
                       k=1.0):  # current capital stock

        self.n, self.s, self.δ, self.α, self.z = n, s, δ, α, z
        self.k = k

    def h(self):
        "Evaluate the h function"
        # Unpack parameters (get rid of self to simplify notation)
        n, s, δ, α, z = self.n, self.s, self.δ, self.α, self.z
        # Apply the update rule
        return (s * z * self.k**α + (1 - δ) * self.k) / (1 + n)

    def update(self):
        "Update the current state (i.e., the capital stock)."
        self.k =  self.h()

    def steady_state(self):
        "Compute the steady state value of capital."
        # Unpack parameters (get rid of self to simplify notation)
        n, s, δ, α, z = self.n, self.s, self.δ, self.α, self.z
        # Compute and return steady state
        return ((s * z) / (n + δ))**(1 / (1 - α))

    def generate_sequence(self, t):
        "Generate and return a time series of length t"
        path = []
        for i in range(t):
            path.append(self.k)
            self.update()
        return path

# Finally, computing time series from two different initial conditions, we can see some charts:

s1 = Solow()
s2 = Solow(k=8.0)

T = 60
fig, ax = plt.subplots(figsize=(9, 6))

# Plot the common steady state value of capital
ax.plot([s1.steady_state()]*T, 'k-', label='steady state')

# Plot time series for each economy
for s in s1, s2:
    lb = f'capital series from initial state {s.k}'
    ax.plot(s.generate_sequence(T), 'o-', lw=2, alpha=0.6, label=lb)

ax.set_xlabel('$k_{t+1}$', fontsize=14)
ax.set_ylabel('$k_t$', fontsize=14)
ax.legend()
plt.show()

###############################################################

# Lecture 3: Dynamics in one dimension

import numpy as np
import matplotlib.pyplot as plt

# Graphical analysis

# Producing 45 degree plots and time series plots (ignore this code)

def subplots(fs):
    "Custom subplots with axes throught the origin"
    fig, ax = plt.subplots(figsize=fs)

    # Set the axes through the origin
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_color('green')
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')

    return fig, ax


def plot45(g, xmin, xmax, x0, num_arrows=6, var='x'):

    xgrid = np.linspace(xmin, xmax, 200)

    fig, ax = subplots((6.5, 6))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    hw = (xmax - xmin) * 0.01
    hl = 2 * hw
    arrow_args = dict(fc="k", ec="k", head_width=hw,
            length_includes_head=True, lw=1,
            alpha=0.6, head_length=hl)

    ax.plot(xgrid, g(xgrid), 'b-', lw=2, alpha=0.6, label='g')
    ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='45')

    x = x0
    xticks = [xmin]
    xtick_labels = [xmin]

    for i in range(num_arrows):
        if i == 0:
            ax.arrow(x, 0.0, 0.0, g(x), **arrow_args) # x, y, dx, dy
        else:
            ax.arrow(x, x, 0.0, g(x) - x, **arrow_args)
            ax.plot((x, x), (0, x), 'k', ls='dotted')

        ax.arrow(x, g(x), g(x) - x, 0, **arrow_args)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i)))

        x = g(x)
        xticks.append(x)
        xtick_labels.append(r'${}_{}$'.format(var, str(i+1)))
        ax.plot((x, x), (0, x), 'k-', ls='dotted')

    xticks.append(xmax)
    xtick_labels.append(xmax)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(xtick_labels)

    bbox = (0., 1.04, 1., .104)
    legend_args = {'bbox_to_anchor': bbox, 'loc': 'upper right'}

    ax.legend(ncol=2, frameon=False, **legend_args, fontsize=14)
    plt.show()

def ts_plot(g, xmin, xmax, x0, ts_length=6, var='x'):
    fig, ax = subplots((7, 5.5))
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel(r'$t$', fontsize=14)
    ax.set_ylabel(r'${}_t$'.format(var), fontsize=14)
    x = np.empty(ts_length)
    x[0] = x0
    for t in range(ts_length-1):
        x[t+1] = g(x[t])
    ax.plot(range(ts_length),
            x,
            'bo-',
            alpha=0.6,
            lw=2,
            label=r'${}_t$'.format(var))
    ax.legend(loc='best', fontsize=14)
    ax.set_xticks(range(ts_length))
    plt.show()

# Fixing parameters for the Solow model

A, s, alpha, delta = 2, 0.3, 0.3, 0.4

# Update function

def g(k):
    return A * s * k**alpha + (1 - delta) * k

# And plotting it

xmin, xmax = 0, 4  # Suitable plotting region.

plot45(g, xmin, xmax, 0, num_arrows=0)

##########################################################

# Lecture 4: Finite Markov Chains

import quantecon as qe
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Let's simulate a markov chain in our own. The algorithm follows:
    # 1. At time t=0, X_o is chosen from psi
    # 2. At each subsequent time t, the new state X_{t+1} is drawn from P(X_t)

# From qe library, we use a function to generating draws from a discrete distribution: random.draw

ψ = (0.3, 0.7)           # probabilities over {0, 1}
cdf = np.cumsum(ψ)       # convert into cummulative distribution
qe.random.draw(cdf, 5)   # generate 5 independent draws from ψ

# The code is composed by
    # A stochastic matrix P (markov matrix)
    # An initial state init
    # A positive integer sample_size representing the lenght of the time series the function should return

def mc_sample_path(P, ψ_0=None, sample_size=1_000):

    # set up
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)

    # Convert each row of P into a cdf
    n = len(P)
    P_dist = [np.cumsum(P[i, :]) for i in range(n)]

    # draw initial state, defaulting to 0
    if ψ_0 is not None:
        X_0 = qe.random.draw(np.cumsum(ψ_0))
    else:
        X_0 = 0

    # simulate
    X[0] = X_0
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])

    return X

P = [[0.4, 0.6],
     [0.2, 0.8]]

X = mc_sample_path(P, ψ_0=[0.1, 0.9], sample_size=100_000)
np.mean(X == 0)

# The same code can be written using qe library:

from quantecon import MarkovChain

mc = qe.MarkovChain(P)
X = mc.simulate(ts_length=1_000_000)
np.mean(X == 0)

# Which gives us the simulated state just using the mc function from qe. And also much faster, since it is used the jit compilation:

%time mc_sample_path(P, sample_size=1_000_000) # Our version

%time mc.simulate(ts_length=1_000_000) # qe version

# Let us add some state values and initial conditions, using the MC function from qe

mc = qe.MarkovChain(P, state_values=('unemployed', 'employed'))
mc.simulate(ts_length=4, init='employed')

mc.simulate(ts_length=4, init='unemployed')

# Simulating with indices instead of states:

mc.simulate_indices(ts_length=4)

########################################################################################

# Lecture 5: Consumption, Savings and Growth

# 5.1. Cake Eating I: Introduction to Optimal Saving

# The idea here is to get use to dynamic programming

import numpy as np
import matplotlib.pyplot as plt

# We have a CRRA utility function

def u(c, γ):

    return c**(1 - γ) / (1 - γ)

# The value function that maximizes the Bellman Eq will be (we will calculate ir later)

def v_star(x, β, γ):

    return (1 - β**(1 / γ))**(-γ) * u(x, γ)

# Plotting the value function with fixed pre-determined parameters:

β, γ = 0.95, 1.2
x_grid = np.linspace(0.1, 5, 100)

fig, ax = plt.subplots()

ax.plot(x_grid, v_star(x_grid, β, γ), label='value function')

ax.set_xlabel('$x$', fontsize=12)
ax.legend(fontsize=12)

plt.show()

# Guessing the optimal policy function for c, we can write c_star:

def c_star(x, β, γ):

    return (1 - β ** (1/γ)) * x

# Plotting the optimal policy function with the state variable time-series, for different parameters:

fig, ax = plt.subplots()
ax.plot(x_grid, c_star(x_grid, β, γ), label='default parameters')
ax.plot(x_grid, c_star(x_grid, β + 0.02, γ), label=r'higher $\beta$')
ax.plot(x_grid, c_star(x_grid, β, γ + 0.2), label=r'higher $\gamma$')
ax.set_ylabel(r'$\sigma(x)$')
ax.set_xlabel('$x$')
ax.legend()

plt.show()

# That is, a higher beta and a higher gamma (more patient and more risk averse), less consumption today, since the agent smooths it as the state changes.

########

# 5.2. Cake Eating II: Numerical Methods

!pip install --upgrade interpolation

import numpy as np
import matplotlib.pyplot as plt

from interpolation import interp
from scipy.optimize import minimize_scalar, bisect

# We will continue the cake eating problem, but now exploring the numerical methods.

# Value Function Iteration (VFI)

# The algorithm follows:
    # 1. Take an arbitrary guess to v
    # 2. Obtain an update w
    # 3. Stop if w approximately equal to v, otherwise set v = w and go back to step 2.

# Defining a maximization function:

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

# Definig a class called CakeEating, its initial values and a method that is called state_action_value,
# that returns the value of consumption choice given a particular state and guess of v.

class CakeEating:

    def __init__(self,
                 β=0.96,           # discount factor
                 γ=1.5,            # degree of relative risk aversion
                 x_grid_min=1e-3,  # exclude zero for numerical stability
                 x_grid_max=2.5,   # size of cake
                 x_grid_size=120): # number of iterations

        self.β, self.γ = β, γ

        # Set up grid
        self.x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)

    # Utility function
    def u(self, c):

        γ = self.γ

        if γ == 1:
            return np.log(c)
        else:
            return (c ** (1 - γ)) / (1 - γ)

    # first derivative of utility function
    def u_prime(self, c):

        return c ** (-self.γ)

    def state_action_value(self, c, x, v_array):
        """
        Right hand side of the Bellman equation given x and c.
        """

        u, β = self.u, self.β
        v = lambda x: interp(self.x_grid, v_array, x)

        return u(c) + β * v(x - c)

# Defining the Bellman operator to update the value function:

def T(v, ce):
    """
    The Bellman operator.  Updates the guess of the value function.

    * ce is an instance of CakeEating
    * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)

    for i, x in enumerate(ce.x_grid):
        # Maximize RHS of Bellman equation at state x
        v_new[i] = maximize(ce.state_action_value, 1e-10, x, (x, v))[1]

    return v_new

# Creating a CakeEating instance:

ce = CakeEating()

# Now, let's see the VFI in action

# Start with a guess v given by v(x) = u(x) for every x grid point

x_grid = ce.x_grid
v = ce.u(x_grid)       # Initial guess
n = 12                 # Number of iterations

fig, ax = plt.subplots()

ax.plot(x_grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='Initial guess')

for i in range(n):
    v = T(v, ce)  # Apply the Bellman operator
    ax.plot(x_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax.legend()
ax.set_ylabel('value', fontsize=12)
ax.set_xlabel('cake size $x$', fontsize=12)
ax.set_title('Value function iterations')

plt.show()

# We can do it more systematically, setting a convergence condition to the VFI stops. We define then the function compute_value_function:

def compute_value_function(ce,
                           tol=1e-4,
                           max_iter=1000,
                           verbose=True,
                           print_skip=25):

    # Set up loop
    v = np.zeros(len(ce.x_grid)) # Initial guess
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(v, ce)

        error = np.max(np.abs(v - v_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        v = v_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return v_new

# Now, computing it, it remains just to call the function, calling "v":

v = compute_value_function(ce)

# Finally, plotting the converged value function:

fig, ax = plt.subplots()

ax.plot(x_grid, v, label='Approximate value function')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_title('Value function')
ax.legend()
plt.show()

# If we want to compare with an analytical solution, we run the following code:

def v_star(ce):

    β, γ = ce.β, ce.γ
    x_grid = ce.x_grid
    u = ce.u

    a = β ** (1 / γ)
    x = 1 - a
    z = u(x_grid)

    return z / x ** γ

v_analytical = v_star(ce)

fig, ax = plt.subplots()

ax.plot(x_grid, v_analytical, label='analytical solution')
ax.plot(x_grid, v, label='numerical solution')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.legend()
ax.set_title('Comparison between analytical and numerical value functions')
plt.show()

# When we can conclude that the quality of approximation is good for large x, but less near the lower boundary.

# Now we can see the policy functions, how they behave in the cake eating model

# For that, let's define a function sigma that computes the policy function for consumption taking as given the approximation of the value function above:

def σ(ce, v):
    """
    The optimal policy function. Given the value function,
    it finds optimal consumption in each state.

    * ce is an instance of CakeEating
    * v is a value function array

    """
    c = np.empty_like(v)

    for i in range(len(ce.x_grid)):
        x = ce.x_grid[i]
        # Maximize RHS of Bellman equation at state x
        c[i] = maximize(ce.state_action_value, 1e-10, x, (x, v))[0]

    return c

# Calling c the policy function

c = σ(ce, v)

# Plotting the numerical and analytical optimal policy function to compare:

def c_star(ce):

    β, γ = ce.β, ce.γ
    x_grid = ce.x_grid

    return (1 - β ** (1/γ)) * x_grid


c_analytical = c_star(ce)

fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label='analytical')
ax.plot(ce.x_grid, c, label='Numerical')
ax.set_ylabel(r'$\sigma(x)$')
ax.set_xlabel('$x$')
ax.legend()

plt.show()

# Both converge, but not perfect. To have more fit, we would need to increase the grid size or reduce the error tolerance.

# However, there is an algorithm that offers the possibility to faster computing time and give more accuracy in the approximation. Let's see it:

# Time Iteration: it uses the FOC and the Euler Equation to iterate time. Usually faster than VFI

def K(σ_array, ce):
    """
    The policy function operator. Given the policy function,
    it updates the optimal consumption using Euler equation.

    * σ_array is an array of policy function values on the grid
    * ce is an instance of CakeEating

    """

    u_prime, β, x_grid = ce.u_prime, ce.β, ce.x_grid
    σ_new = np.empty_like(σ_array)

    σ = lambda x: interp(x_grid, σ_array, x)

    def euler_diff(c, x):
        return u_prime(c) - β * u_prime(σ(x - c))

    for i, x in enumerate(x_grid):

        # handle small x separately --- helps numerical stability
        if x < 1e-12:
            σ_new[i] = 0.0

        # handle other x
        else:
            σ_new[i] = bisect(euler_diff, 1e-10, x - 1e-10, x)

    return σ_new

# Defining the Euler Iteration:

def iterate_euler_equation(ce,
                           max_iter=500,
                           tol=1e-5,
                           verbose=True,
                           print_skip=25):

    x_grid = ce.x_grid

    σ = np.copy(x_grid)        # initial guess

    i = 0
    error = tol + 1
    while i < max_iter and error > tol:

        σ_new = K(σ, ce)

        error = np.max(np.abs(σ_new - σ))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        σ = σ_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return σ

# Iterating

ce = CakeEating(x_grid_min=0.0)
c_euler = iterate_euler_equation(ce)

# Comparing the time iteration with the analytical solution. We can see that the fit is better than before.

fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label='analytical solution')
ax.plot(ce.x_grid, c_euler, label='time iteration solution')

ax.set_ylabel('consumption')
ax.set_xlabel('$x$')
ax.legend(fontsize=12)

plt.show()

####################################################################


