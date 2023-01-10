#!/usr/bin/env python
# coding: utf-8

# # Taylor problem 2.40

# Consider an object that is coasting horizontally (positive $x$ direction) subject to a drag force $f = -bv -c v^2$.  Your first job is to solve Newton's 2nd law equation for $v(t)$ by separating variables.  You should find:
# 
# $$\begin{align}
#   v(t) &= \frac{b A e^{-bt/m}}{1 - c A e^{-bt/m}} \\
#   A &\equiv \frac{v_0}{b + c v_0}
# \end{align}$$
# 
# Now we want to plot $v(t)$ as analyze the behavior for large $t$.
# 
# **Go through and fill in the blanks where ### appears.**

# In[ ]:


import numpy as np

def v_of_t(t, b, c, v0, m=1):
    A = v0/(b + c*v0)
    return   ### fill in the equation here


# Next we make a plot in the standard way:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

t_pts =  ### determine a set of t points such that you see the decay
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t_pts, v_of_t(t_pts, 1., 1., 1.))
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$v(t)$')


# Now we add another plot and check if it is an exponential decay.  **What kind of plot is this?  (Google the name along with 'matplotlib'.)**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

t_pts = np.arange(0., 3., 0.1)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1)
ax.plot(t_pts, v_of_t(t_pts, 1., 1., 1.))
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$v(t)$')
ax.grid(True)

ax2 = fig.add_subplot(1,2,2)
ax2.semilogy(t_pts, v_of_t(t_pts, 1., 1., 1.))
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$v(t)$')
ax2.grid(True)

fig.tight_layout()  # make the spacing of subplots nicer


# In[ ]:


fig.savefig('Taylor_prob_2.40.png', bbox_inches='tight')
### Find the figure file and display it in your browser, then save or print. 
### What do you learn from the second graph?


# In[ ]:




