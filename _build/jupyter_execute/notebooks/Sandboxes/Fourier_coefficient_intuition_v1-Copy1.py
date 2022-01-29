#!/usr/bin/env python
# coding: utf-8

# # Fourier coefficient intuition
# 
# last revised: 18-Jan-2019 by Dick Furnstahl [furnstahl.1@osu.edu]
# 

# Make plots of products of sines and cosines to gain intuition on how Fourier coefficients are projected out.

# The Fourier expansion of a periodic function $f(t+\tau) = f(t)$ takes the form
# 
# $\begin{align}
#   f(t) = \sum_{n=0} a_n \cos(n\omega t) + b_n \sin(n\omega t)
# \end{align}$
# 
# where $\omega = 2\pi/\tau$.
# The coefficients are found from integrals of cosines and sines over $f(t)$:
# 
# $\begin{align}
#   a_0 = \frac{1}{\tau} \int_{-\tau/2}^{\tau/2} f(t)\, dt  \qquad
#   b_0 = 0
# \end{align}$
# 
# $\begin{align}
#   a_m = \frac{2}{\tau} \int_{-\tau/2}^{\tau/2} \cos(m\omega t) f(t)\, dt  \qquad
#   b_m = \frac{2}{\tau} \int_{-\tau/2}^{\tau/2} \sin(m\omega t) f(t)\, dt 
# \end{align}$
# 
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from ipywidgets import interact


# In[4]:


def Fourier_integrand_plot(n, m):
    """Plot the integrands of sine and cosines of different orders"""
    tau = 1.
    omega = 2.*np.pi / tau
    t_pts = np.arange(-tau/2., tau/2., 0.01)
    
    cos_n_pts = np.cos(n * omega * t_pts)
    cos_m_pts = np.cos(m * omega * t_pts)
    sin_n_pts = np.sin(n * omega * t_pts)
    sin_m_pts = np.sin(m * omega * t_pts)
    
    fig = plt.figure(figsize=(12,4))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.plot(t_pts, cos_n_pts*cos_m_pts, color='red')
    ax1.plot(t_pts, cos_n_pts, color='blue', linestyle='dashed', alpha=0.5)
    ax1.plot(t_pts, cos_m_pts, color='green', linestyle='dotted', alpha=0.5)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_title('cosine * cosine')

    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(t_pts, sin_n_pts*sin_m_pts, color='red')
    ax2.plot(t_pts, sin_n_pts, color='blue', linestyle='dashed', alpha=0.5)
    ax2.plot(t_pts, sin_m_pts, color='green', linestyle='dotted', alpha=0.5)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title('sine * sine')

    ax3 = fig.add_subplot(1,3,3)
    ax3.plot(t_pts, cos_n_pts*sin_m_pts, color='red')
    ax3.plot(t_pts, cos_n_pts, color='blue', linestyle='dashed', alpha=0.5)
    ax3.plot(t_pts, sin_m_pts, color='green', linestyle='dotted', alpha=0.8)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_title('cosine * sine')
    


# The plots below show the products of cosines and sines

# In[5]:


interact(Fourier_integrand_plot, m=(0,4), n=(0,4));


# In[ ]:




