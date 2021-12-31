#!/usr/bin/env python
# coding: utf-8

# # Taylor problem 1.50 supplement
# 
# last revised: 31-Dec-2021 by Dick Furnstahl [furnstahl.1@osu.edu]
# 
# The goal of this notebook is to create an energy diagram graph for problem 1.50 to help explain the observation that the small angle approximation graph completes a period faster than the full solution.
# 
# The equations are:
# 
# $
# \begin{align}
#   \ddot\phi = -\frac{g}{R}\sin\phi \qquad\mbox{and}\qquad \ddot\phi = -\frac{g}{R}\phi
# \end{align}
# $
# 
# The potential energies associated with the small angle case is (with $g = R = 1$) given by $U(\phi)=\phi^2/2$.  For the full
# case, with the same zero of potential energy, $U(x) = 1 - \cos\phi$.  Note that these agree when $\phi$ is sufficiently small.
# 
# Plan: construct the energy diagrams for these two potentials.  Use the fact that they start with the same initial condition:
# $\phi(0) = \phi_0$ for some $\phi_0$ and $\dot\phi(0) = 0$.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


# In[ ]:


x = np.linspace(-3.,3.,200)  # adjust this interval to make a good plot

fig = plt.figure(num='Comparison of full potential and harmonic approximation')
ax = fig.add_subplot(1,1,1)
ax.plot(x, x**2/2, label=r'harmonic: $\phi^2/2$', color='blue')
ax.plot(x, 1.-np.cos(x), label=r'$1 - \cos\ \phi$', color='red')
ax.axhline(1., color='blue', linestyle='--')
ax.axhline(0.846, color='red', linestyle='--')
ax.axvline(np.sqrt(2.), color='green')
ax.set_ylim(-.1,3.)
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$U(\phi)$')
ax.legend(loc='best')

fig.tight_layout()  # separates the plot from the controls


# Now do the analysis!

# In[ ]:




