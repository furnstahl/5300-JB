#!/usr/bin/env python
# coding: utf-8

# # Taylor problem 4.37 template
# 
# In this problem the potential energy $U(\phi)$ depends on the masses $m$ and $M$, with the qualitative behavior of the system starting at $\phi=0$ depending on the ratio $m/M$. We will determine the critical value of this ratio from the graphs of the potential energy (corresponding to taking $\phi = 0$, meaning $M$ at its lowest, as the zero of potential energy):
# 
# $$\begin{align}
#   U(\phi) = M g R (1-\cos\phi) - m g R \phi  \;.
# \end{align}$$
# 
# **This notebook has a different potential energy function that has the same qualitative behavior.  Your job is to implement the correct formula.  Places you need to change are marked with ###.**
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# Make a plot of the potential with simple values of the constants.

# In[ ]:


def U(phi, M_big, m_small, R=1., g=1.):
    """Potential energy for the system in Taylor, problem 4.37."""
    return M_big*g*R*(1.-np.exp(-phi**4/2.)) - m_small*g*R*phi  ### made up U(\phi)


# In[ ]:


# set non-essential constants to 1 (doesn't change shapes)
R = 1.
g = 1.
M_big = 1.

phi_pts = np.arange(0., 3.5, 0.01)

fig_U = plt.figure()
ax_U = fig_U.add_subplot(1,1,1)

m_small = 0.5  ### different values
U_of_phi_pts = U(phi_pts, M_big, m_small)
ax_U.plot(phi_pts, U_of_phi_pts, 'b-', label=fr'$m/M = {m_small:0.1f}$', lw=2)

m_small = 0.6  ### different values
U_of_phi_pts = U(phi_pts, M_big, m_small)
ax_U.plot(phi_pts, U_of_phi_pts, 'g-', label=fr'$m/M = {m_small:0.1f}$', lw=2)


ax_U.set_xlabel(r'$\phi$')
ax_U.set_ylabel(r'$U(\phi)$')

ax_U.set_xlim(0, 3.5)
#ax_U.set_ylim(-1., 6.)

# draw a black horizontal line at 0 but with alpha=0.3 so less distracting
ax_U.axhline(0., color='black', alpha=0.3 )  
ax_U.legend();


# Let's plot it with a slider to find the critical value (where the curve is tangent to the line at zero. 

# Ok, let's change $m/M$ with a slider:

# In[ ]:


from ipywidgets import interact, fixed
import ipywidgets as widgets


# In[ ]:


# set non-essential constants to 1 (doesn't change shapes)
R = 1.
g = 1.
M_big = 1.

def plot_U_given_m_over_M(m_over_M, M_big=1., R=1., g=1.):
    ""
    phi_pts = np.arange(0., 3.5, 0.01)

    fig_U = plt.figure()
    ax_U = fig_U.add_subplot(1,1,1)

    m_small = m_over_M * M_big
    U_of_phi_pts = U(phi_pts, M_big, m_small)
    ax_U.plot(phi_pts, U_of_phi_pts, 'r-', 
              label=rf'$m/M = {m_over_M:.2f}$', lw=2)

    ax_U.set_xlabel(r'$\phi$')
    ax_U.set_ylabel(r'$U(\phi)$')

    ax_U.set_xlim(0, 3.5)
    #ax_U.set_ylim(-1., 6.)

    # draw a black horizontal line at 0 but with alpha=0.3 so less distracting
    ax_U.axhline(0., color='black', alpha=0.3 )  

    ax_U.legend();

# fix the values of M_big, R, and g
interact(plot_U_given_m_over_M, 
         m_over_M = widgets.FloatSlider(min=0., max=1., step=0.01, 
                                        continuous_update=False,
                                        value=0.5, readout_format='.3f'), 
         M_big=fixed(1.), R=fixed(1.), g=fixed(1.));


# Note that you can type numbers in the box to be more precise.

# In[ ]:




