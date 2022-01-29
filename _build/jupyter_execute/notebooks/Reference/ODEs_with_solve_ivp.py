#!/usr/bin/env python
# coding: utf-8

# # Solving ODEs with scipy.integrate.solve_ivp

# ## Solving ordinary differential equations (ODEs)
# 
# Here we will revisit the differential equations solved in 5300_Jupyter_Python_intro_01.ipynb with `odeint`, only now we'll use `solve_ivp` from Scipy.  We'll compare the new and old solutions as we go.

# ### First-order ODE

# In[1]:


# Import the required modules
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp   # Now preferred to odeint


# Let's try a one-dimensional first-order ODE, say:
# 
# $\begin{align}
# \quad 
# \frac{dv}{dt} = -g, \quad \mbox{with} \quad v(0) = 10
# \end{align}$
# 
# in some appropriate units (we'll use MKS units by default).  This ODE can be separated and directly integrated:
# 
# $\begin{align}
#   \int_{v_0=10}^{v} dv' = - g \int_{0}^{t} dt'
#   \quad\Longrightarrow\quad
#     v - v_0 = - g (t - 0)
#   \quad\Longrightarrow\quad
#    v(t) = 10 - gt
# \end{align}$
# 
# 

# The goal is to find the solution $v(t)$ as an array `v_pts` at the times in the array `t_pts`.

# In[2]:


# Define a function which calculates the derivative
def dv_dt_new(t, v, g=9.8):
    """Returns the right side of a simple first-order ODE with default g."""
    return -g   

t_start = 0.
t_end = 10.
t_pts = np.linspace(t_start, t_end, 20)  # 20 points between t=0 and t=10.

v_0 = np.array([10.0])  # initial condition, in form of a list or numpy array

abserr = 1.e-8
relerr = 1.e-8

solution = solve_ivp(dv_dt_new, (t_start, t_end), v_0, t_eval=t_pts,
                     rtol=relerr, atol=abserr)  
    # solve_ivp( function for rhs with (t, v) argument (cf. (v,t) for odeint), 
    #            tspan=(starting t value, ending t value),
    #            initial value of v(t), array of points we want to know v(t),
    #            method='RK45' is the default method,
    #            rtol=1.e-3, atol=1.e-6 are default tolerances
    #          )
v_pts = solution.y  # array of results at t_pts


# In[3]:


v_pts.shape   # 1 x 100 matrix (row vector)


# Here's how we did it before with odeint:

# In[4]:


from scipy.integrate import odeint   

# Define a function which calculates the derivative
def dv_dt(v, t, g=9.8):
    """Returns the right side of a simple first-order ODE with default g."""
    return -g   

t_pts = np.linspace(0., 10., 20)     # 20 points between t=0 and t=10.
v_0 = 10.  # the initial condition
v_pts_odeint = odeint(dv_dt, v_0, t_pts)  # odeint( function for rhs, 
                                          #         initial value of v(t),
                                          #         array of t values )


# In[5]:


v_pts_odeint.shape   # 100 x 1 matrix (column vector)


# Make a table comparing results (using `flatten()` to make the matrices into arrays):

# In[6]:


print('    t     v(t) [solve_ivp]    v(t) [odeint]')
for t, v_solve_ivp, v_odeint in zip(t_pts, 
                                    v_pts.flatten(), 
                                    v_pts_odeint.flatten()):
    print(f' {t:6.3f}   {v_solve_ivp:12.7f}       {v_odeint:12.7f}')


# Differences between `solve_ivp` and `odeint`:
# * `dv_dt(t, v)`  vs.  `dv_dt(v, t)`, i.e., the function definitions have the arguments reversed.
# * With `odeint`, you only specify the full array of $t$ points you want to know $v(t)$ at.  With `solve_ivp`, you first specify the starting $t$ and ending $t$ as a tuple: `(t_start, t_end)` and then (optionally) specify `t_eval=t_pts` to evaluate $v$ at the points in the `t_pts` array.
# * `solve_ivp` returns an object from which $v(t)$ (and other results) can be found, while `ode_int` returns $v(t)$.
# * For this single first-order equation, $v(t)$ is returned for the $N$ requested $t$ points as a $1 \times N$ two-dimensional array by `solve_ivp` and as a $N \times 1$ array by `odeint`.
# * `odeint` has no choice of solver while the `solve_ivp` solver can be set by `method`.  The default is `method='RK45'`, which is good, general-purpose Runge-Kutta solver.  

# ### Second-order ODE

# Suppose we have a second-order ODE such as:
# 
# $$
# \quad y'' + 2 y' + 2 y = \cos(2x), \quad \quad y(0) = 0, \; y'(0) = 0
# $$
# 
# We can turn this into two first-order equations by defining a new dependent variable. For example,
# 
# $$
# \quad z \equiv y' \quad \Rightarrow \quad z' + 2 z + 2y = \cos(2x), \quad z(0)=y(0) = 0.
# $$
# 
# Now introduce the vector 
# 
# $$
#   \mathbf{U}(x) = \left(\begin{array}{c}
#                          y(x) \\
#                          z(x)
#                         \end{array}
#                   \right)
#         \quad\Longrightarrow\quad
#     \frac{d\mathbf{U}}{dx} = \left(\begin{array}{c}
#                                     z \\
#                                     -2 y' - 2 y + \cos(2x)
#                                    \end{array}
#                              \right) 
# $$
# 
# We can solve this system of ODEs using `solve_ivp` with lists, as follows.  We will try it first without specifying the relative and absolute error tolerances rtol and atol.

# In[7]:


# Define a function for the right side
def dU_dx_new(x, U):
    """Right side of the differential equation to be solved.
    U is a two-component vector with y=U[0] and z=U[1]. 
    Thus this function should return [y', z']
    """
    return [U[1], -2*U[1] - 2*U[0] + np.cos(2*x)]

# initial condition U_0 = [y(0)=0, z(0)=y'(0)=0]
U_0 = [0., 0.]

x_pts = np.linspace(0, 15, 20)  # Set up the mesh of x points
result = solve_ivp(dU_dx_new, (0, 15), U_0, t_eval=x_pts)
y_pts = result.y[0,:]   # Ok, this is tricky.  For each x, result.y has two 
                        #  components.  We want the first component for all
                        #  x, which is y(x).  The 0 means the first index and 
                        #  the : means all of the x values.


# Here's how we did it before with `odeint`:

# In[8]:


# Define a function for the right side
def dU_dx(U, x):
    """Right side of the differential equation to be solved.
    U is a two-component vector with y=U[0] and z=U[1]. 
    Thus this function should return [y', z']
    """
    return [U[1], -2*U[1] - 2*U[0] + np.cos(2*x)]

# initial condition U_0 = [y(0)=0, z(0)=y'(0)=0]
U_0 = [0., 0.]

x_pts = np.linspace(0, 15, 20)  # Set up the mesh of x points
U_pts = odeint(dU_dx, U_0, x_pts)  # U_pts is a 2-dimensional array
y_pts_odeint = U_pts[:,0]  # Ok, this is tricky.  For each x, U_pts has two 
                           #  components.  We want the upper component for all
                           #  x, which is y(x).  The : means all of the first 
                           #  index, which is x, and the 0 means the first
                           #  component in the other dimension.


# Make a table comparing results (using `flatten()` to make the matrices into arrays):

# In[9]:


print('    x     y(x) [solve_ivp]    y(x) [odeint]')
for x, y_solve_ivp, y_odeint in zip(x_pts, 
                                    y_pts.flatten(), 
                                    y_pts_odeint.flatten()):
    print(f' {x:6.3f}   {y_solve_ivp:12.7f}       {y_odeint:12.7f}')


# Not very close agreement by the end.  Run both again with greater accuracy.

# In[10]:


relerr = 1.e-10
abserr = 1.e-10

result = solve_ivp(dU_dx_new, (0, 15), U_0, t_eval=x_pts, 
                   rtol=relerr, atol=abserr)
y_pts = result.y[0,:]    

U_pts = odeint(dU_dx, U_0, x_pts, 
               rtol=relerr, atol=abserr)  
y_pts_odeint = U_pts[:,0]   

print('    x     y(x) [solve_ivp]    y(x) [odeint]')
for x, y_solve_ivp, y_odeint in zip(x_pts, 
                                    y_pts.flatten(), 
                                    y_pts_odeint.flatten()):
    print(f' {x:6.3f}   {y_solve_ivp:12.7f}       {y_odeint:12.7f}')


# Comparing the results from when we didn't specify the errors we see that the default error tolerances for solve_ivp were insufficient.  Moral: specify them explicitly.  
