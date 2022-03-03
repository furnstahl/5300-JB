#!/usr/bin/env python
# coding: utf-8

# # Playing with coupled oscillators - v3
# 
# This notebook builds on previous notebooks such as Orbit_games.ipynb to implement the solution of linear coupled differential equations using `solve_ivp` as well as with more analytic methods.  The formalism and notation is based on Taylor, Chapter 11: *Coupled Oscillators and Normal Modes*.

# The general form of the equations to solve is compactly expressed in matrix form:
# 
# $\begin{align}
#   \mathbf{M} \mathbf{\ddot q} = - \mathbf{K} \mathbf{q}
# \end{align}$
# 
# where $\mathbf{M}$ and $\mathbf{K}$ are $N \times N$ symmetric matrices and $\mathbf{q}$ is a vector of generalized coordinates (this is $\mathbf{x}$ in the beginning discussion in Taylor).  
# 
# We can solve this directly as differential equations or solve for normal modes.  For the former, we map the problem onto linear differential equations and the vectors $\mathbf{y}$ and $d\mathbf{y}/dt$, which are each of length $2N$.  Here is the $N=2$ mapping:
# 
# $\begin{align}
#   \mathbf{y} = \left( \begin{array}{c}
#                          q_1 \\
#                          q_2 \\
#                          \dot q_1 \\
#                          \dot q_2
#                       \end{array}   
#                        \right)
#  \qquad\qquad
# \frac{d\mathbf{y}}{dt} = \left( \begin{array}{c}
#                          \dot q_1 \\
#                          \dot q_2 \\
#                          \ddot q_1 \\
#                          \ddot q_2
#                       \end{array}   
#                        \right)
# \end{align}$
# 
# Then for the $\ddot q$ parts we use the matrix solution of our equation:
# 
# $\begin{align}
#   \mathbf{\ddot q} = - \mathbf{M}^{-1} \mathbf{K} \mathbf{q}
# \end{align}$
# 
# where $\mathbf{M}^{-1}$ means the matrix inverse.  Check out the implementation below!
# 
# To solve for normal modes, we solve the generalized eigenvalue problem
# 
# $\begin{align}
#   (\mathbf{K} - \omega^2 \mathbf{M}) \mathbf{a} = 0
# \end{align}$
# 
# using functions from `scipy.linalg` to find the eigenvalues and eigenvectors.  We use these below together with the numerical solutions to make plots of the normal modes.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import linalg as LA 


# In[2]:


# Change the common font size
font_size = 14
plt.rcParams.update({'font.size': font_size})


# In[3]:


class CoupledOscillators:
    """
    Solve the equations for linear coupled oscillators in matrix formulation.
    """
    
    def __init__(self, M_matrix, K_matrix, b_matrix=np.zeros(1)):
        self.M_matrix = M_matrix
        self.K_matrix = K_matrix
        self.q_len = len(self.M_matrix)
        if b_matrix.any():
            self.b_matrix = b_matrix
        else:    
            self.b_matrix = np.zeros((self.q_len)) 
        self.full_matrix = np.zeros((self.q_len, self.q_len))
        
            
    def dy_dt(self, t, y):
        """
        This function returns the right-hand side of the diffeq: 
        [dq_vec/dt d^2q_vec/dt^2]
        
        Parameters
        ----------
        t : float
            time 
        y : float
            vector with y[:q_len] = q_vec, y[q_len:] = dqvec/dt
            
        """
        dy_dt_vec = np.zeros(2*self.q_len)   # Initialize to twice q_vec size
        # Set the upper half [:self.q_len] of dy_dt to the lower half of y
        dy_dt_vec[:self.q_len] = y[self.q_len:]  
        # Set the lower half [self.q_len:] of dy_dt to the vector that
        #  results from M^{-1} K y where @ is matrix multiplication
        dy_dt_vec[self.q_len:] = -1.* LA.inv(self.M_matrix) @                     ( self.K_matrix @ y[:self.q_len] +                       self.b_matrix @ y[self.q_len:] )
        return dy_dt_vec
    
    
    def solve_ode(self, t_pts, q_vec_0, q_vec_dot_0,
                  method='RK45',
                  abserr=1.0e-10, relerr=1.0e-10):
        """
        Solve the ODE given initial conditions.
        Use solve_ivp with the option of specifying the method.
        Specify smaller abserr and relerr to get more precision.
        """
        y = np.concatenate((q_vec_0, q_vec_dot_0))
        solution = solve_ivp(self.dy_dt, (t_pts[0], t_pts[-1]), 
                             y, t_eval=t_pts, method=method, 
                             atol=abserr, rtol=relerr)
        q_vec, q_vec_dot = np.array_split(solution.y, 2)
        return q_vec, q_vec_dot
    

    def find_eigenmodes(self):
        """
        Find the normal modes.
        """
        eig_vals, eig_vecs = LA.eigh(K_matrix, M_matrix)
        self.frequencies = np.sqrt(eig_vals)
        return eig_vals, eig_vecs
    
    def plot_eigenmodes(self, t_pts, output_file=None):
        """
        plot the normal modes separately
        """
        from itertools import cycle
        colors = ['blue', 'green', 'red', 'purple']

        eig_vals, eig_vecs = self.find_eigenmodes()
        for i in np.arange(self.q_len): 
            q_vec_0 = eig_vecs[:,i]   # this is the i'th eigenvector
            q_vec_dot_0 = np.zeros(self.q_len)  # start with all qdots = 0
            q_vec, q_vec_dot = self.solve_ode(t_pts, q_vec_0, q_vec_dot_0)
             
            fig, axes = plt.subplots(self.q_len, 1, 
                                     figsize=(10, 2.*self.q_len))
            color_list = cycle(colors)
            for j in np.arange(self.q_len):
               axes[j].plot(t_pts, q_vec[j], color=next(color_list))
               axes[j].set_xlabel(r'$t$')
               axes[j].set_ylabel(fr'$q_{j+1:d}$')
               axes[j].axvline(2.*np.pi/np.sqrt(eig_vals[i]), 
                               color='black', linestyle=':')
            overall_title = fr'Normal mode {i+1:d}, ' +                             fr'$\omega = {self.frequencies[i]:.2f}$' 
            fig.suptitle(overall_title, va='center')
            fig.tight_layout()
            if output_file:
                 output_file_name = output_file.replace('.', f'_{i:d}.', 1)
                 fig.savefig(output_file_name, dpi=200, bbox_inches='tight')
         


# In[4]:


def start_stop_indices(t_pts, plot_start, plot_stop):
    start_index = (np.fabs(t_pts-plot_start)).argmin()  # index in t_pts array 
    stop_index = (np.fabs(t_pts-plot_stop)).argmin()  # index in t_pts array 
    return start_index, stop_index


# ## Section 11.2

# In[5]:


m = 1.
M_matrix = np.array([ [m, 0.], [0., m] ])
print('M matrix: \n', M_matrix, '\n')

k = 1.
K_matrix = np.array([ [2.*k, -k], [-k, 2.*k]])
print('K matrix: \n', K_matrix, '\n')

# b = 0
# b_matrix = np.array([ [b, 0], [0, b]])
# print('b matrix: \n', b_matrix, '\n')

co1 = CoupledOscillators(M_matrix, K_matrix)

eig_vals, eig_vecs = co1.find_eigenmodes() # LA.eigh(K_matrix, M_matrix)

print(f'omega^2 = {eig_vals}')
print(f'frequencies = {co1.frequencies}')


# In[6]:


print(eig_vecs[:,0])
print(eig_vecs[:,1])


# In[7]:


t_start = 0.
t_end = 10.
delta_t = 0.001

t_pts = np.arange(t_start, t_end+delta_t, delta_t)  


# In[8]:


co1.plot_eigenmodes(t_pts, 'Section11p2.png')


# ### Figure 11.6
# 
# This has $A_1 = 1$, $A_2 = 0.7$, $\delta_1 = 0$ and $\delta_2 = \pi/2$.

# In[9]:


t_start = 0.
t_end = 50.
delta_t = 0.001

t_pts = np.arange(t_start, t_end+delta_t, delta_t)  


# In[10]:


colors = ['blue', 'green', 'red', 'purple']

# changing A_i -> -A_i to match convention in Figure 11.6
A_1 = -1.
delta_1 = 0.
A_2 = -0.7
delta_2 = np.pi/2.

omega_1 = co1.frequencies[0]
omega_2 = co1.frequencies[1]

q_vec_0 = A_1 * eig_vecs[:,0] * np.cos(-delta_1) +           A_2 * eig_vecs[:,1] * np.cos(-delta_2)
q_vec_dot_0 = omega_1 * A_1 * eig_vecs[:,0] * np.sin(delta_1) +               omega_2 * A_2 * eig_vecs[:,1] * np.sin(delta_2)
q_vec_11p6, q_vec_dot_11p6 = co1.solve_ode(t_pts, q_vec_0, q_vec_dot_0)

fig, axes = plt.subplots(co1.q_len, 1, figsize=(10, 2.*co1.q_len))
for j in np.arange(co1.q_len):
    axes[j].plot(t_pts, q_vec_11p6[j], color=colors[j])
    axes[j].set_xlabel(r'$t$')
    axes[j].set_ylabel(fr'$q_{j+1:d}$')
    axes[j].set_xlim(t_pts[0], t_pts[-1])
    axes[j].axhline(0., color='black', alpha=0.3)
overall_title = 'General solution: ' + fr'$A_1 = {A_1:.2f}$, ' +                                        fr'$\delta_1 = {delta_1:.2f}$, ' +                                        fr'$A_2 = {A_2:.2f}$, ' +                                        fr'$\delta_2 = {delta_2:.2f}$'  
fig.suptitle(overall_title, va='center')
fig.tight_layout()


print( q_vec_0, q_vec_dot_0 )  # debugging check


# ## Problem 11.5
# 
# Almost the same as section 11.2, but mass m2 is not attached by a spring to the wall.

# In[11]:


m = 1.
M_matrix = np.array([ [m, 0.], [0., m] ])
print('M matrix: \n', M_matrix, '\n')

k = 1.
K_matrix = np.array([ [2.*k, -k], [-k, k]])
print('K matrix: \n', K_matrix, '\n')

# b = 0
# b_matrix = np.array([ [b, 0], [0, b]])
# print('b matrix: \n', b_matrix, '\n')

co1 = CoupledOscillators(M_matrix, K_matrix)

eig_vals, eig_vecs = co1.find_eigenmodes() # LA.eigh(K_matrix, M_matrix)

print(f'omega^2 = {eig_vals}')
print(f'frequencies = {co1.frequencies}')


# In[12]:


print(eig_vecs[:,0])
print(eig_vecs[:,1])


# In[13]:


t_start = 0.
t_end = 10.
delta_t = 0.001

t_pts = np.arange(t_start, t_end+delta_t, delta_t)  
co1.plot_eigenmodes(t_pts)


# ## Problem 11.7
# 
# Writing the general solution to the cart problem in Section 11.2 in a different form and plot two solutions.

# In[14]:


m = 1.
M_matrix = np.array([ [m, 0.], [0., m] ])
print('M matrix: \n', M_matrix, '\n')

k = 1.
K_matrix = np.array([ [2.*k, -k], [-k, 2.*k]])
print('K matrix: \n', K_matrix, '\n')

# b = 0
# b_matrix = np.array([ [b, 0], [0, b]])
# print('b matrix: \n', b_matrix, '\n')

co1 = CoupledOscillators(M_matrix, K_matrix)

eig_vals, eig_vecs = co1.find_eigenmodes() # LA.eigh(K_matrix, M_matrix)

print(f'omega^2 = {eig_vals}')
print(f'frequencies = {co1.frequencies}')


# In[15]:


t_start = 0.
t_end = 30.
delta_t = 0.001

t_pts = np.arange(t_start, t_end+delta_t, delta_t)  


# In[16]:


colors = ['blue', 'green', 'red', 'purple']

# changing A -> -A to match convention in Taylor
A = -1.
B_1 = A
B_2 = 0.
C_1 = 0.
C_2 = 0.

omega_1 = co1.frequencies[0]
omega_2 = co1.frequencies[1]

q_vec_0 = B_1 * eig_vecs[:,0] +           B_2 * eig_vecs[:,1] 
q_vec_dot_0 = omega_1 * C_1 * eig_vecs[:,0] +               omega_2 * C_2 * eig_vecs[:,1]
q_vec_11p7a, q_vec_dot_11p7a = co1.solve_ode(t_pts, q_vec_0, q_vec_dot_0)

fig, axes = plt.subplots(co1.q_len, 1, figsize=(10, 2.*co1.q_len))
for j in np.arange(co1.q_len):
    axes[j].plot(t_pts, q_vec_11p7a[j], color=colors[j])
    axes[j].set_xlabel(r'$t$')
    axes[j].set_ylabel(fr'$q_{j+1:d}$')
    axes[j].set_xlim(t_pts[0], t_pts[-1])
    axes[j].axhline(0., color='black', alpha=0.3)
overall_title = 'Specific solution 1: ' + fr'$B_1 = {B_1:.2f}$, ' +                                           fr'$C_1 = {C_1:.2f}$, ' +                                           fr'$B_2 = {B_2:.2f}$, ' +                                           fr'$C_2 = {C_2:.2f}$'  
fig.suptitle(overall_title, va='center')
fig.tight_layout()


print( q_vec_0, q_vec_dot_0 )  # debugging check


# In[17]:


colors = ['blue', 'green', 'red', 'purple']

# changing A -> -A to match convention in Taylor
A = -1.
B_1 = A / 2.
B_2 = A / 2.
C_1 = 0.
C_2 = 0.

omega_1 = co1.frequencies[0]
omega_2 = co1.frequencies[1]

q_vec_0 = B_1 * eig_vecs[:,0] +           B_2 * eig_vecs[:,1] 
q_vec_dot_0 = omega_1 * C_1 * eig_vecs[:,0] +               omega_2 * C_2 * eig_vecs[:,1]
q_vec_11p7a, q_vec_dot_11p7a = co1.solve_ode(t_pts, q_vec_0, q_vec_dot_0)

fig, axes = plt.subplots(co1.q_len, 1, figsize=(10, 2.*co1.q_len))
for j in np.arange(co1.q_len):
    axes[j].plot(t_pts, q_vec_11p7a[j], color=colors[j])
    axes[j].set_xlabel(r'$t$')
    axes[j].set_ylabel(fr'$q_{j+1:d}$')
    axes[j].set_xlim(t_pts[0], t_pts[-1])
    axes[j].axhline(0., color='black', alpha=0.3)
overall_title = 'Specific solution 1: ' + fr'$B_1 = {B_1:.2f}$, ' +                                           fr'$C_1 = {C_1:.2f}$, ' +                                           fr'$B_2 = {B_2:.2f}$, ' +                                           fr'$C_2 = {C_2:.2f}$'  
fig.suptitle(overall_title, va='center')
fig.tight_layout()


print( q_vec_0, q_vec_dot_0 )  # debugging check


# ## Double pendulum with small oscillations

# In[18]:


m = 1.
L = 1.
M_matrix = m * L**2 * np.array([ [2., 1.], [1., 1.] ])
print('M matrix: \n', M_matrix, '\n')

k = 1.
omega_0 = 1.
K_matrix = m * L**2 * np.array([ [2.*omega_0**2, 0.], [0., omega_0**2]])
print('K matrix: \n', K_matrix, '\n')

# b = 0
# b_matrix = np.array([ [b, 0], [0, b]])
# print('b matrix: \n', b_matrix, '\n')

double_pendulum = CoupledOscillators(M_matrix, K_matrix)

eig_vals, eig_vecs = double_pendulum.find_eigenmodes() 

print(f'omega^2 = {eig_vals}')
print(f'frequencies = {double_pendulum.frequencies}')
print(f'{2. + np.sqrt(2.):.4f}  {2. - np.sqrt(2.):.4f}')


# In[19]:


print(eig_vecs[:,0])
print(eig_vecs[:,1])


# In[20]:


t_start = 0.
t_end = 10.
delta_t = 0.001

t_pts = np.arange(t_start, t_end+delta_t, delta_t)  
double_pendulum.plot_eigenmodes(t_pts)


# In[ ]:




