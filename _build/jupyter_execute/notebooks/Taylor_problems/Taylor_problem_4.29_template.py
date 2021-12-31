#!/usr/bin/env python
# coding: utf-8

# # Taylor problem 4.29 template
# 
# In this problem the one-dimenstional potential energy is 
# 
# $$\begin{align}
#   U(x) = k x^4  \qquad k > 0 \;.
# \end{align}$$
# 
# **However, at present the graph and calculation is for a different potential.  Your job is to change it everywhere relevant to be the correct potential.  Look for places with ###.**
# 
# We can apply the formula for the time between positions $x_0$ and $x$ given by
# 
# $$\begin{align}
#   t = \sqrt{\frac{m}{2}} \int_{x_0}^{x} \frac{dx'}{\sqrt{E - U(x')}}
# \end{align}$$
# 
# to the interval from $x'=0$ to $x'=x_{max}$, which is one-fourth of the period $\tau$:
# 
# $$\begin{align}
#   \tau = \sqrt{2m} \int_{0}^{x_{max}} \frac{dx'}{\sqrt{E - U(x')}}
# \end{align}$$
# 
# Because we'll be evaluating integrals, we import a numerical integration function called `quad` as well as our standard numpy and matplotlib imports.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Make a plot of the potential with simple values of the constants.

# In[ ]:


# set constants to 1 per instructions in the problem statement
k = m = A = 1.

x_pts = np.arange(-2.,2.,0.01)
U_of_x_pts = k*x_pts**6   ### Use the correct potential here

fig_U = plt.figure(figsize=(5,5))
ax_U = fig_U.add_subplot(1,1,1)

ax_U.plot(x_pts, U_of_x_pts, 'b-', label=r'$k x^6$', lw=2)  ### fix label
ax_U.set_xlabel('x')
ax_U.set_ylabel('U(x)')

# add the harmonic oscillator for comparison
ax_U.plot(x_pts, k*x_pts**2, 'r:', label=r'$k x^2$', lw=3)

ax_U.set_xlim(-1.5, 1.5)
ax_U.set_ylim(-1., 6.)
# draw a black horizontal line at E but with alpha=0.3 so less distracting
ax_U.plot([-2.,2.], [1.,1.], 'k-', label='E', alpha=0.3 )  

ax_U.legend();


# Change the integral here to the relevant one for this problem.

# In[ ]:


def integrand(x):
    """Integrand of a dimensionless integral that is part of an expression
        for the period given a potential energy U.
    """
    return 1./np.sqrt(1.-x**6)     ### fix this expression


# It's always a good idea to plot the integrand before doing the integral.  The integral is the area under this curve.

# In[ ]:


x_pts = np.arange(0., 1., 0.01)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_pts, integrand(x_pts), 'r-')
ax.set_xlabel('x')
ax.set_ylabel('integrand(x)');


# In[ ]:


answer, error = quad(integrand, 0., 1.)
# Use f-strings for formatting (google "python f-string" to learn more)
print(f'The integral is {answer:.10f} with estimated error {error:.4e}.')

