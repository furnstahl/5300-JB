#!/usr/bin/env python
# coding: utf-8

# # Taylor examples 8.1 and 8.2

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Change the common font size
font_size = 14
plt.rcParams.update({'font.size': font_size})


# In[3]:


class Potential:
    """
    Potential for central force motion.
    """
    
    def __init__(self, ang_mom, gamma=1, mu=1):
        self.ang_mom = ang_mom
        self.gamma = gamma
        self.mu = mu
    
    def U(self, r):
        """Potential energy of the form U = -gamma/r."""
        return -self.gamma / r
    
    def U_deriv(self, r):
        """dU/dr"""
        return self.gamma / r**2
    
    def Ucf(self, r):
        """Centrifugal potential energy"""
        return self.ang_mom**2 / (2. * self.mu * r**2)
    
    def Ucf_deriv(self, r):
        """dU_cf/dr"""
        return -2. * self.ang_mom**2 / (2. * self.mu * r**3)
    
    def Ueff(self, r):
        """effective potential energy"""
        return self.U(r) + self.Ucf(r)
    
    def Ueff_deriv(self, r):
        """dU_eff/dr"""
        return self.U_deriv(r) + self.Ucf_deriv(r)
    


# ## Examples 8.1 and 8.2: Effective potential energy for a comet

# In[4]:


fig_1 = plt.figure(figsize=(12,4), num='Energy diagrams')

gamma = 1.
ang_mom = 2.
p1 = Potential(ang_mom, gamma=gamma, mu=1)
E1 = 3.

ax_1 = fig_1.add_subplot(1,3,1)

r_pts = np.linspace(0.001, 3., 200)
U_pts = p1.U(r_pts)
Ucf_pts = p1.Ucf(r_pts)
Ueff_pts = p1.Ueff(r_pts)

ax_1.plot(r_pts, U_pts, linestyle='dashed', color='blue', label='U(r)')
ax_1.plot(r_pts, Ucf_pts, linestyle='dotted', color='green', label='U_cf(r)')
ax_1.plot(r_pts, Ueff_pts, linestyle='solid', color='red', label='U_eff(r)')

ax_1.set_xlim(0., 3.)
ax_1.set_ylim(-10., 10.)
ax_1.set_xlabel('r')
ax_1.set_ylabel('U(r)')
ax_1.set_title(f'$\gamma = {gamma},\ \ l = {ang_mom}$')
ax_1.legend(loc='upper center')

ax_1.axhline(0.0, color='black', alpha=0.3)

ax_1.axhline(E1, color='red', alpha=0.5)
ax_1.annotate(r'$E_1$', (2.5,3.5), color='red', alpha=0.7)

gamma = 3.
ang_mom = 1.
p2 = Potential(ang_mom, gamma=gamma, mu=1)

E2 = -2.


ax_2 = fig_1.add_subplot(1,3,2)

r_pts = np.linspace(0.001, 3., 200)
U_pts = p2.U(r_pts)
Ucf_pts = p2.Ucf(r_pts)
Ueff_pts = p2.Ueff(r_pts)

ax_2.plot(r_pts, U_pts, linestyle='dashed', color='blue', label='U(r)')
ax_2.plot(r_pts, Ucf_pts, linestyle='dotted', 
          color='green', label='U_cf(r)')
ax_2.plot(r_pts, Ueff_pts, linestyle='solid', color='red', label='U_eff(r)')

ax_2.set_xlim(0., 3.)
ax_2.set_ylim(-10., 10.)
ax_2.set_xlabel('r')
ax_2.set_ylabel('U(r)')
ax_2.set_title(f'$\gamma = {gamma},\ \ l = {ang_mom}$')
ax_2.legend(loc='upper center')

ax_2.axhline(0.0, color='black', alpha=0.3)

ax_2.axhline(E1, color='red', alpha=0.5)
ax_2.axhline(E2, color='red', alpha=0.5)
ax_2.annotate(r'$E_1$', (2.5,3.5), color='red', alpha=0.7)
ax_2.annotate(r'$E_2$', (2.5,-3.2), color='red', alpha=0.7)

fig_1.tight_layout()


# In[ ]:





# In[ ]:




